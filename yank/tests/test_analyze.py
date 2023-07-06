#!/usr/local/bin/env python

"""
Test multistate.analyzers facility.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import copy
import shutil
import os
import tempfile
import logging

import numpy as np
from nose.tools import assert_raises, assert_equal
import openmmtools as mmtools
import pymbar
from pymbar.utils import ParameterError
from simtk import unit
from openmmtools.multistate import (MultiStateReporter, MultiStateSampler,
                                    ReplicaExchangeSampler, SAMSSampler, utils)

from yank.yank import Topography
from yank.restraints import RestraintState
from yank import analyze
from .test_experiment import solvation_stock

# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # Boltzmann constant
# quiet down some citation spam
MultiStateSampler._global_citation_silence = True


# ==============================================================================
# TEMPLATE PHASE CLASS WITH NO OBSERVABLES
# ==============================================================================

class BlankPhase(analyze.YankPhaseAnalyzer):
    """Create a blank phase class with no get_X (observable) methods for testing the MultiPhaseAnalyzer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze_phase(self, *args, **kwargs):
        pass

    def get_effective_energy_timeseries(self):
        pass

    def _prepare_mbar_input_data(self):
        pass

    @classmethod
    def _get_cache_dependency_graph(cls):
        return {'reporter': []}


class FreeEnergyPhase(BlankPhase):
    """
    Create a phase which can return a free energy estimate

    Tests the Observables depending on 2 phases and has an error
    """
    # Static constants test can access
    fev = 1
    dfev = 0.5

    def get_free_energy(self):
        value = np.ones([2, 2]) * self.fev
        error = np.ones([2, 2]) * self.dfev
        self._computed_observables['free_energy'] = {'value': value, 'error': error}
        return value, error


class FEStandardStatePhase(FreeEnergyPhase):
    """
    Class which also defines a standard state
    Tests Observable with no error depending only on phase
    """
    # Static constant the test can access
    ssc = -1

    def get_standard_state_correction(self):
        self._computed_observables['standard_state_correction'] = self.ssc
        return self._computed_observables['standard_state_correction']


# ==============================================================================
# UTILITY TESTS
# ==============================================================================

def test_timeseries():
    """Test that the timeseries analysis utilities behave as expected"""
    timeseries = pymbar.timeseries
    # Generate some data
    a = 1
    i_max = 20
    n_i = 1000
    i_t = np.linspace(0, i_max, n_i)
    # Boltzmann distribution
    series = np.sqrt(2/np.pi) * i_t**2 * np.exp(-i_t/(2*a**2))/a**3
    scale = np.random.normal(size=n_i)*(1-i_t/i_max)/2  # Noise which decays
    full_series = series + scale
    # Analyze output
    a_i_t, a_g_i, a_n_effective_i = utils.get_equilibration_data_per_sample(full_series, fast=True, max_subset=n_i)
    a_n_effective_max = a_n_effective_i.max()
    a_i_max = a_n_effective_i.argmax()
    a_n_equilibration = a_i_t[a_i_max]
    a_g_t = a_g_i[a_i_max]
    a_equilibration_data = [a_n_equilibration, a_g_t, a_n_effective_max]
    # MBAR output
    m_equilibration_data = timeseries.detectEquilibration(full_series, fast=True)
    assert np.allclose(m_equilibration_data, a_equilibration_data)


def test_auto_analyze():
    """Test that the auto analysis and multi-exp analysis code works"""
    with mmtools.utils.temporary_directory() as tmp_dir:
        # Make the logger less noisy
        logging.disable(21)  # Raise threshold to just above INFO logging
        # Ensure 1 iteration runs to allow analysis
        run_one = {'options': {'default_number_of_iterations': 1}}
        script, builder = solvation_stock(tmp_dir, overwrite_options=run_one)
        output_dir = os.path.normpath(builder._get_experiment_dir(''))
        assert os.path.isdir(output_dir)
        single_auto = analyze.ExperimentAnalyzer(output_dir)
        exp_directories = builder.get_experiment_directories()
        multi_auto = analyze.MultiExperimentAnalyzer(script)
        assert sorted(exp_directories) == sorted(multi_auto.paths)
        payload = multi_auto.run_all_analysis(serial_data_path=os.path.join(tmp_dir, '_analysis.pkl'))
        _, exp_name = os.path.split(output_dir)
        # Check like in the code
        if exp_name == '':
            exp_name = 'experiment'
        np.testing.assert_equal(payload[exp_name], single_auto.auto_analyze())
        # Restore logger
        logging.disable(logging.NOTSET)


# ==============================================================================
# TEST ANALYZER
# ==============================================================================

class TestMultiPhaseAnalyzer(object):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 3
    N_STATES = 5
    SAMPLER = MultiStateSampler
    ANALYZER = analyze.YankMultiStateSamplerAnalyzer
    ONLINE_ANALYSIS = False

    # --------------------------------------
    # Optional helper function to overwrite.
    # --------------------------------------

    @classmethod
    def call_sampler_create(cls, sampler, reporter,
                            thermodynamic_states,
                            sampler_states,
                            unsampled_states,
                            metadata=None):
        """Helper function to call the create method for the sampler"""
        # Allows initial thermodynamic states to be handled by the built in methods
        sampler.create(thermodynamic_states, sampler_states, reporter, unsampled_thermodynamic_states=unsampled_states,
                       metadata=metadata)

    @classmethod
    def setup_class(cls):
        """Shared test cases and variables."""
        cls.checkpoint_interval = 2
        # Make sure we collect the same number of samples for all tests to avoid instabilities in MBAR.
        base_steps = 50
        cls.n_steps = int(np.ceil(base_steps / cls.N_SAMPLERS))

        # Test case with host guest in vacuum at 3 different positions and alchemical parameters.
        # -----------------------------------------------------------------------------------------
        hostguest_test = mmtools.testsystems.HostGuestVacuum()
        factory = mmtools.alchemy.AbsoluteAlchemicalFactory()
        alchemical_region = mmtools.alchemy.AlchemicalRegion(alchemical_atoms=range(126, 156))
        hostguest_alchemical = factory.create_alchemical_system(hostguest_test.system, alchemical_region)

        # Add restraint force between host and guest.
        restraint_force = mmtools.forces.HarmonicRestraintBondForce(
            spring_constant=2.0 * unit.kilojoule_per_mole / unit.angstrom ** 2,
            restrained_atom_index1=10, restrained_atom_index2=16,
        )
        hostguest_alchemical.addForce(copy.deepcopy(restraint_force))

        # Translate the sampler states to be different one from each other.
        positions = hostguest_test.positions
        box_vectors = hostguest_test.system.getDefaultPeriodicBoxVectors()
        hostguest_sampler_states = [mmtools.states.SamplerState(positions=positions + 10*i*unit.nanometers,
                                                                box_vectors=box_vectors)
                                    for i in range(cls.N_SAMPLERS)]

        # Create the basic thermodynamic states.
        hostguest_thermodynamic_states = [mmtools.states.ThermodynamicState(hostguest_alchemical, 300*unit.kelvin)
                                          for _ in range(cls.N_STATES)]

        # Create alchemical states at different parameter values.
        alchemical_states = [mmtools.alchemy.AlchemicalState.from_system(hostguest_alchemical)
                             for _ in range(cls.N_STATES)]
        for i, alchemical_state in enumerate(alchemical_states):
            alchemical_state.set_alchemical_parameters(float(i) / (cls.N_STATES - 1))

        # Create compound states.
        hostguest_compound_states = list()
        for i in range(cls.N_STATES):
            hostguest_compound_states.append(
                mmtools.states.CompoundThermodynamicState(thermodynamic_state=hostguest_thermodynamic_states[i],
                                                          composable_states=[alchemical_states[i]])
            )

        # Unsampled states.
        cls.n_unsampled_states = 2
        nonalchemical_system = copy.deepcopy(hostguest_test.system)
        nonalchemical_system.addForce(copy.deepcopy(restraint_force))
        nonalchemical_state = mmtools.states.ThermodynamicState(nonalchemical_system, 300*unit.kelvin)
        nonalchemical_compound_state = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=nonalchemical_state,
            composable_states=[RestraintState(lambda_restraints=1.0)]
        )
        hostguest_unsampled_states = [copy.deepcopy(nonalchemical_compound_state) for _ in
                                      range(cls.n_unsampled_states)]

        cls.hostguest_test = (hostguest_compound_states, hostguest_sampler_states, hostguest_unsampled_states)

        # Run a quick simulation
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(cls.hostguest_test)
        n_states = len(thermodynamic_states)

        # Prepare metadata for analysis.
        reference_state = mmtools.states.ThermodynamicState(hostguest_test.system, 300*unit.kelvin)
        topography = Topography(hostguest_test.topology, ligand_atoms=range(126, 156))
        metadata = {
            'standard_state_correction': 4.0,
            'reference_state': mmtools.utils.serialize(reference_state),
            'topography': mmtools.utils.serialize(topography)
        }
        analysis_atoms = topography.receptor_atoms

        # Create simulation and storage file.
        cls.tmp_dir = tempfile.mkdtemp()
        storage_path = os.path.join(cls.tmp_dir, 'test_analyze.nc')
        move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
        if cls.ONLINE_ANALYSIS:
            online_analysis_interval = cls.n_steps - 1
        else:
            online_analysis_interval = None
        cls.sampler = cls.SAMPLER(mcmc_moves=move, number_of_iterations=cls.n_steps,
                                  online_analysis_interval=online_analysis_interval,
                                  online_analysis_minimum_iterations=0)
        cls.reporter = MultiStateReporter(storage_path, checkpoint_interval=cls.checkpoint_interval,
                                          analysis_particle_indices=analysis_atoms)
        cls.call_sampler_create(cls.sampler, cls.reporter, thermodynamic_states, sampler_states, unsampled_states,
                                metadata=metadata)

        # Run some iterations.
        cls.n_replicas = cls.N_SAMPLERS
        cls.n_states = n_states
        cls.analysis_atoms = analysis_atoms
        cls.sampler.run(cls.n_steps-1)  # Initial config
        cls.repex_name = "RepexAnalyzer"  # kind of an unused test

        # Debugging Messages to sent to Nose with --nocapture enabled
        online_flag = " "
        if cls.ONLINE_ANALYSIS:
            online_flag += "Online "
        output_descr = "Testing{}Analyzer: {}  -- States: {}  -- Samplers: {}".format(
            online_flag, cls.SAMPLER.__name__, cls.N_STATES, cls.N_SAMPLERS)
        len_output = len(output_descr)
        print("#" * len_output)
        print(output_descr)
        print("#" * len_output)

    @classmethod
    def teardown_class(cls):
        cls.reporter.close()
        shutil.rmtree(cls.tmp_dir)

    def test_phase_initialize(self):
        """Test that the Phase analyzer initializes correctly"""
        phase = self.ANALYZER(self.reporter, name=self.repex_name)
        assert phase.reporter is self.reporter
        assert phase.name == self.repex_name

    def test_mixing_stats(self):
        """Test that the Phase yields mixing stats that make sense"""
        phase = self.ANALYZER(self.reporter, name=self.repex_name)
        t, mu, g = phase.generate_mixing_statistics()
        # Output is the correct number of states
        assert t.shape == (self.n_states, self.n_states)
        # Assert transition matrix values are all 0 <= x <= 1
        assert np.all(np.logical_and(t >= 0, t <= 1))
        # Assert all rows add to 1
        # Floating point error can lead mu[0] to be not exactly 1.0, but like 0.99999998 or something
        for row in range(self.n_states):
            assert np.allclose(t[row, :].sum(), 1)
        # Assert all eigenvalues are all <= 1
        assert np.allclose(mu[0], 1)

    def test_mbar_creation_process(self):
        """Test each of the steps of the MBAR creation process

        We do this in one function since the test for each part would be a bunch of repeated recreation of the phase
        """
        phase = self.ANALYZER(self.reporter, name=self.repex_name, unbias_restraint=False)
        u_sampled, u_unsampled, neighborhood, sampled_states = phase.read_energies()
        # Test energy output matches appropriate MBAR shapes
        assert u_sampled.shape == (self.n_replicas, self.n_states, self.n_steps)
        assert u_unsampled.shape == (self.n_replicas, 2, self.n_steps)
        u_n = phase.get_effective_energy_timeseries()
        assert u_n.shape == (self.n_steps,)

        # This may need to be adjusted from time to time as analyze changes
        discard = 1  # The analysis discards the minimization frame.
        # Generate mbar semi-manually, use phases's static methods
        n_eq, g_t, Neff_max = pymbar.timeseries.detectEquilibration(u_n[discard:])
        u_sampled_sub = mmtools.multistate.remove_unequilibrated_data(u_sampled, n_eq, -1)
        # Make sure output from discarding iterations is what we expect.
        assert u_sampled_sub.shape == (self.n_replicas, self.n_states, phase.n_iterations + 1 - n_eq)

        # Generate MBAR from phase
        phase_mbar = phase.mbar
        # Assert mbar object is formed of nstates + unsampled states, Number of effective samples
        # Round up to nearest int to handle floating point issues in single replica tests
        n_effective_samples = np.ceil(Neff_max - discard)
        assert phase_mbar.u_kn.shape[0] == self.n_states + 2
        assert abs(phase_mbar.u_kn.shape[1]/self.n_replicas - n_effective_samples) <= 1

        # Check that Free energies are returned correctly
        fe, dfe = self.help_fe_calc(phase)
        assert fe.shape == (self.n_states + 2, self.n_states + 2)
        stored_fe_dict = phase._computed_observables['free_energy']
        stored_fe, stored_dfe = stored_fe_dict['value'], stored_fe_dict['error']
        assert np.allclose(stored_fe, fe), "stored_fe = {}, fe = {}".format(stored_fe, fe)
        assert np.allclose(stored_dfe, dfe), "stored_dfe = {}, dfe = {}".format(stored_dfe, dfe)
        # Test reference states and full work up creation
        iinit, jinit = phase.reference_states
        output = phase.analyze_phase()
        fe_out, dfe_out = output['free_energy_diff'], output['free_energy_diff_error']
        assert fe_out == fe[iinit, jinit]
        assert dfe_out == dfe[iinit, jinit]
        inew, jnew = 1, 2
        phase.reference_states = [inew, jnew]
        new_output = phase.analyze_phase()
        new_fe_out, new_dfe_out = new_output['free_energy_diff'], new_output['free_energy_diff_error']
        assert new_fe_out == fe[inew, jnew]
        assert new_dfe_out == dfe[inew, jnew]

    @staticmethod
    def invoke_free_energy(phase):
        """Secondary helper function to try and solve for free energy"""
        fe, dfe = phase.get_free_energy()
        if np.any(np.isnan(dfe)) or np.any(np.isnan(fe)):
            raise RuntimeError("Free energy or its error is NaN, likely due to insufficient run time!")
        return fe, dfe

    def help_fe_calc(self, phase):
        try:
            fe, dfe = self.invoke_free_energy(phase)
        except (ParameterError, RuntimeError) as e:
            # Handle case where MBAR does not have a converged free energy yet by attempting to run longer
            # Only run up until we have sampled every state, or we hit some cycle limit
            self.reporter.open(mode='a')
            cycle_limit = 20  # Put some upper limit of cycles
            cycles = 0
            cycle_steps = 20
            throw = True
            phase.clear()
            while (not np.unique(self.sampler._reporter.read_replica_thermodynamic_states()).size == self.N_STATES
                   and cycles < cycle_limit):
                self.sampler.extend(cycle_steps)
                try:
                    fe, dfe = self.invoke_free_energy(phase)
                except (ParameterError, RuntimeError):
                    # If the max error count internally is reached, its a RuntimeError and won't be trapped
                    # So it will be raised correctly
                    pass
                else:
                    # Test is good, let it pass by returning here
                    throw = False
                    break
                cycles += 1
                phase.clear()
            self.reporter.sync()
            self.reporter.open(mode='r')
            if throw:
                # If we get here, we have not validated, raise original error
                raise e
            self.n_steps = self.n_steps + cycles * cycle_steps
        return fe, dfe

    def test_multi_phase(self):
        """Test MultiPhaseAnalysis"""
        # Create the phases
        full_phase = self.ANALYZER(self.reporter, name=self.repex_name, unbias_restraint=False)
        blank_phase = BlankPhase(self.reporter, name="blank")
        fe_phase = FreeEnergyPhase(self.reporter, name="fe")
        fes_phase = FEStandardStatePhase(self.reporter, name="fes")
        # Test that a full phase can be added to itself
        double = full_phase + full_phase
        triple = full_phase + full_phase - full_phase
        # Tests the __neg__ of phase and not just __sub__ of phase
        quad = - full_phase - full_phase + full_phase + full_phase
        # Assert that names are unique
        for names in [double.names, triple.names, quad.names]:
            # Assert basename in names
            assert self.repex_name in names
            # Assert unique names
            assert len(names) == len(set(names))
        # Check signs
        assert double.signs == ['+', '+']
        assert triple.signs == ['+', '+', '-']
        assert quad.signs == ['-', '-', '+', '+']
        # Check cannot combine things that are blank
        with assert_raises(RuntimeError):
            _ = full_phase + blank_phase
        # Check combine partial data
        full_fe_phase = full_phase + fe_phase
        # Check methods  of full_fe
        assert hasattr(full_fe_phase, 'get_free_energy')
        assert not hasattr(full_fe_phase, 'get_enthalpy')
        # prep final property
        full_fes = full_phase + fes_phase
        # Compute free energy
        free_energy_full_fe, dfree_energy_full_fe = self.help_fe_calc(full_fe_phase)
        full_free_energy, full_dfree_energy = self.help_fe_calc(full_phase)
        i, j = full_phase.reference_states
        full_free_energy, full_dfree_energy = full_free_energy[i, j], full_dfree_energy[i, j]
        # Check free energy values
        assert free_energy_full_fe == full_free_energy + fe_phase.fev
        assert np.allclose(dfree_energy_full_fe, np.sqrt(full_dfree_energy**2 + fe_phase.dfev**2))
        # Check by phase (should only yield 1 value, no error)
        combo_standard_state = full_fes.get_standard_state_correction()
        assert combo_standard_state == full_phase.get_standard_state_correction() + fes_phase.ssc
        # Check compound multiphase stuff
        quad_free_energy, quad_dfree_energy = quad.get_free_energy()
        assert quad_free_energy == 0
        assert np.allclose(quad_dfree_energy, np.sqrt(4*full_dfree_energy**2))
        assert np.allclose(triple.get_standard_state_correction(),
                           (2*full_phase.get_standard_state_correction() - full_phase.get_standard_state_correction()))

    def test_yank_registry(self):
        """Test that observable registry is implemented correctly for a given class"""
        phase = self.ANALYZER(self.reporter, name=self.repex_name)
        observables = set(analyze.yank_registry.observables)
        assert set(phase.observables) == observables

    def test_cache_dependency_graph_generation(self):
        """Test that the dependency graph used to invalidate cached values is generated correctly."""
        cache_dependency_graph = self.ANALYZER._get_cache_dependency_graph()
        test_cases = [
            ('reporter', {'equilibration_data'}),
            ('mbar', {'observables'}),
            ('unbiased_decorrelated_u_ln', {'mbar'}),
            ('equilibration_data', {'decorrelated_state_indices_ln',
                                    'decorrelated_u_ln', 'decorrelated_N_l'})
        ]
        for cached_property, properties_to_invalidate in test_cases:
            assert_equal(cache_dependency_graph[cached_property], properties_to_invalidate,
                         msg='Property "{}"'.format(cached_property))

    def test_cached_properties_dependencies(self):
        """Test that cached properties are invalidated when their dependencies change."""
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name, unbias_restraint=False)

        def check_cached_properties(is_in):
            for cached_property in cached_properties:
                err_msg = '{} is cached != {}'.format(cached_property, is_in)
                assert (cached_property in analyzer._cache) is is_in, err_msg
            assert (analyzer._computed_observables['free_energy'] is not None) is is_in

        # Test-precondition: make sure the dependencies are as expected.
        assert 'equilibration_data' in self.ANALYZER._decorrelated_state_indices_ln.dependencies
        assert 'max_n_iterations' in self.ANALYZER._equilibration_data.dependencies

        # The cached value and its dependencies are generated lazily when calling the property.
        cached_properties = ['mbar', 'decorrelated_state_indices_ln', 'equilibration_data']
        yield check_cached_properties, False
        self.help_fe_calc(analyzer)
        analyzer._decorrelated_state_indices_ln
        analyzer.get_free_energy()
        yield check_cached_properties, True

        # If we invalidate one of the dependencies, the values that depend on it are invalidated too.
        analyzer.max_n_iterations = analyzer.n_iterations - 1
        yield check_cached_properties, False

        # Dependent values are not invalidated if the dependency doesn't change.
        analyzer._decorrelated_state_indices_ln  # Generate dependent values.
        analyzer.get_free_energy()
        check_cached_properties(is_in=True)  # Test pre-condition.
        analyzer.max_n_iterations = analyzer.n_iterations - 1
        yield check_cached_properties, True  # Cached values are still there.

    def test_max_n_iterations(self):
        """Test that max_n_iterations limits the number of iterations analyzed."""
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name)

        def get_n_analyzed_iterations():
            """Compute the number of equilibrated + decorrelated + truncated iterations."""
            n_iterations = (analyzer.max_n_iterations + 1 - analyzer.n_equilibration_iterations)
            return n_iterations / analyzer.statistical_inefficiency

        # By default, all iterations are analyzed.
        n_analyzed_iterations1 = get_n_analyzed_iterations()
        all_decorrelated_iterations = analyzer._decorrelated_iterations
        all_decorrelated_u_ln = analyzer._decorrelated_u_ln
        n_iterations_ln1 = analyzer.n_replicas * n_analyzed_iterations1
        assert abs(len(all_decorrelated_iterations) - n_analyzed_iterations1) < 1
        assert abs(all_decorrelated_u_ln.shape[1] - n_iterations_ln1) < self.n_replicas

        # Setting max_n_iterations reduces the number of iterations analyzed
        analyzer.max_n_iterations = analyzer.n_iterations - 1
        n_analyzed_iterations2 = get_n_analyzed_iterations()
        n_iterations_ln2 = analyzer.n_replicas * n_analyzed_iterations2
        assert abs(len(analyzer._decorrelated_iterations) - n_analyzed_iterations2) < 1
        assert abs(analyzer._decorrelated_u_ln.shape[1] - n_iterations_ln2) < self.n_replicas

        # Check that the energies of the iterations match.
        def get_matched_iteration(iterations1, iterations2):
            n_analyzed_iterations = len(iterations1)
            n_iterations_ln = len(iterations1) * self.n_replicas
            matched_iterations_ln = set()
            for matched_iteration in sorted(set(iterations1) & set(iterations2)):
                matched_iteration_idx = np.where(iterations1 == matched_iteration)[0][0]
                matched_iterations_ln.update(
                    set(range(matched_iteration_idx, n_iterations_ln, n_analyzed_iterations))
                )
            return sorted(matched_iterations_ln)

        matched_iterations1 = get_matched_iteration(all_decorrelated_iterations, analyzer._decorrelated_iterations)
        matched_iterations2 = get_matched_iteration(analyzer._decorrelated_iterations, all_decorrelated_iterations)
        assert np.array_equal(all_decorrelated_u_ln[:, matched_iterations1],
                              analyzer._decorrelated_u_ln[:, matched_iterations2])

    def test_unbias_restraint(self):
        """Test that restraints are unbiased correctly when unbias_restraint is True."""
        # The energies of the unbiased restraint should be identical to the biased ones.
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name, unbias_restraint=False)
        assert np.array_equal(analyzer._unbiased_decorrelated_u_ln, analyzer._decorrelated_u_ln)
        assert np.array_equal(analyzer._unbiased_decorrelated_N_l, analyzer._decorrelated_N_l)

        # Switch unbias_restraint to True. The old cached value is invalidated and
        # now u_ln and N_l have two extra states.
        analyzer.unbias_restraint = True
        # Set a cutoff that allows us to discard some iterations.
        restraint_data = analyzer._get_radially_symmetric_restraint_data()
        restraint_energies, restraint_distances = analyzer._compute_restraint_energies(*restraint_data)
        analyzer.restraint_distance_cutoff = np.mean(restraint_distances)
        assert analyzer._unbiased_decorrelated_u_ln.shape[0] == analyzer._decorrelated_u_ln.shape[0] + 2
        assert analyzer._unbiased_decorrelated_N_l.shape[0] == analyzer._decorrelated_N_l.shape[0] + 2
        assert analyzer._unbiased_decorrelated_u_ln.shape[1] < analyzer._decorrelated_u_ln.shape[1]
        # The energy without the restraint should always be smaller.
        assert np.all(analyzer._unbiased_decorrelated_u_ln[0] < analyzer._unbiased_decorrelated_u_ln[1])
        assert np.all(analyzer._unbiased_decorrelated_u_ln[-1] < analyzer._unbiased_decorrelated_u_ln[-2])
        assert analyzer._unbiased_decorrelated_N_l[0] == 0
        assert analyzer._unbiased_decorrelated_N_l[-1] == 0

        # With a very big energy cutoff, all the energies besides the extra two states should be identical.
        analyzer.restraint_distance_cutoff = None
        analyzer.restraint_energy_cutoff = 100.0  # kT
        assert np.array_equal(analyzer._unbiased_decorrelated_u_ln[1:-1], analyzer._decorrelated_u_ln)
        assert np.array_equal(analyzer._unbiased_decorrelated_N_l[1:-1], analyzer._decorrelated_N_l)

    def test_extract_trajectory(self):
        """extract_trajectory handles checkpointing and skip frame correctly."""
        n_frames = self.reporter.read_last_iteration(False) + 1  # Include minimization iteration.

        # Make sure the "solute"-only (analysis atoms) trajectory has the correct properties
        solute_trajectory = analyze.extract_trajectory(self.reporter.filepath, replica_index=0, keep_solvent=False)
        assert len(solute_trajectory) == n_frames
        assert solute_trajectory.n_atoms == len(self.analysis_atoms)

        # Check that only the checkpoint trajectory is returned when keep_solvent is True.
        # The -1 is because frame 0 is discarded from trajectory extraction due to equilibration problems.
        # Should this change in analyze, then this logic will need to be changed as well.
        # The int() rounds down from sampling to a state in between the interval.
        full_trajectory = analyze.extract_trajectory(self.reporter.filepath, replica_index=0, keep_solvent=True)
        assert len(full_trajectory) == (n_frames + 1) // self.checkpoint_interval

        # Check that skip_frame reduces the trajectory length correctly.
        skip_frame = 2
        trajectory = analyze.extract_trajectory(self.reporter.filepath, replica_index=0, skip_frame=skip_frame)
        assert len(trajectory) == n_frames / self.checkpoint_interval // skip_frame + 1

        # Extracting the trajectory of a state does not incur into errors.
        analyze.extract_trajectory(self.reporter.filepath, state_index=0, keep_solvent=False)

    def test_online_data_read(self):
        """Test that online data is read from file when available and correctly invalidates when flag is toggled"""
        if not self.ONLINE_ANALYSIS:
            return
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name, use_online_data=True, unbias_restraint=False)
        # Ensure online data was read
        assert analyzer.use_online_data is True
        assert analyzer._online_data is not None
        # Spin up mbar
        _ = analyzer.mbar
        # Ensure the online data was used
        assert analyzer._extra_analysis_kwargs != {}
        # Clear and reset data
        analyzer.use_online_data = False
        assert analyzer._extra_analysis_kwargs == {}
        assert 'mbar' not in analyzer._cache

    def test_online_data_ignored(self):
        """Test that online data is ignored when set from start or when user overrides"""
        if not self.ONLINE_ANALYSIS:
            return
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name, use_online_data=False, unbias_restraint=False)
        # Ensure online data is present, but unread
        assert analyzer.use_online_data is False
        assert analyzer._online_data is not None
        _ = analyzer.mbar
        assert analyzer._extra_analysis_kwargs == {}
        # Start over
        del analyzer
        initial_f_k = [0]*(self.N_STATES + self.n_unsampled_states)
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name,
                                 use_online_data=True, analysis_kwargs={'initial_f_k': initial_f_k},
                                 unbias_restraint=False)
        assert analyzer.use_online_data is True
        assert analyzer._online_data is not None
        _ = analyzer.mbar
        assert np.all(analyzer._extra_analysis_kwargs['initial_f_k'] == initial_f_k)
        # Ensure mbar was not invalidated
        assert 'mbar' in analyzer._cache


class TestRepexAnalyzer(TestMultiPhaseAnalyzer):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 5
    N_STATES = 5
    SAMPLER = ReplicaExchangeSampler


class TestMultiPhaseAnalyzerReverse(TestMultiPhaseAnalyzer):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 5
    N_STATES = 3
    SAMPLER = MultiStateSampler


class TestMultiPhaseAnalyzerOnline(TestMultiPhaseAnalyzerReverse):
    ONLINE_ANALYSIS = True


class TestSAMSAnalyzerSingle(TestMultiPhaseAnalyzer):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 1
    N_STATES = 5
    SAMPLER = SAMSSampler


class TestSAMSAnalyzerSingleOnline(TestSAMSAnalyzerSingle):

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------
    ONLINE_ANALYSIS = True


class TestSAMSAnalyzerMulti(TestSAMSAnalyzerSingle):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 3
