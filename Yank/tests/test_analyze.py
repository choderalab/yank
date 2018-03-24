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

import numpy as np
from nose.tools import assert_raises, assert_equal
import openmmtools as mmtools
import pymbar
from pymbar.utils import ParameterError
from simtk import unit

from yank.yank import Topography
from yank.restraints import RestraintState
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler, SAMSSampler
import yank.analyze as analyze

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

    def get_states_energies(self):
        pass

    def get_timeseries(self, passed_timeseries, replica_state_indices):
        pass

    def get_timeseries_weights(self, *args):
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
        checkpoint_interval = 2
        n_steps = 5

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
        nonalchemical_system = copy.deepcopy(hostguest_test.system)
        nonalchemical_system.addForce(copy.deepcopy(restraint_force))
        nonalchemical_state = mmtools.states.ThermodynamicState(nonalchemical_system, 300*unit.kelvin)
        nonalchemical_compound_state = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=nonalchemical_state,
            composable_states=[RestraintState(lambda_restraints=1.0)]
        )
        hostguest_unsampled_states = [copy.deepcopy(nonalchemical_compound_state) for _ in range(2)]

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
        cls.sampler = cls.SAMPLER(mcmc_moves=move, number_of_iterations=n_steps)
        cls.reporter = MultiStateReporter(storage_path, checkpoint_interval=checkpoint_interval,
                                          analysis_particle_indices=analysis_atoms)
        cls.call_sampler_create(cls.sampler,cls.reporter, thermodynamic_states, sampler_states, unsampled_states,
                                metadata=metadata)
        # run some iterations
        cls.n_replicas = cls.N_SAMPLERS
        cls.n_states = n_states
        cls.n_steps = n_steps
        cls.checkpoint_interval = checkpoint_interval
        cls.analysis_atoms = analysis_atoms
        cls.sampler.run(cls.n_steps-1)  # Initial config
        cls.repex_name = "RepexAnalyzer"  # kind of an unused test

        # Debugging Messages to sent to Nose with --nocapture enabled
        output_descr = "Testing Sampler: {}  -- States: {}  -- Samplers: {}".format(
            cls.SAMPLER.__name__, cls.N_STATES, cls.N_SAMPLERS)
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
        u_n = phase.get_timeseries(u_sampled, sampled_states)
        assert u_n.shape == (self.n_steps,)
        # This may need to be adjusted from time to time as analyze changes
        discard = 1
        # Generate mbar semi-manually, use phases's static methods
        n_eq, g_t, Neff_max = pymbar.timeseries.detectEquilibration(u_n[discard:])
        u_sampled_sub = analyze.multistate.remove_unequilibrated_data(u_sampled, n_eq, -1)
        # Make sure output from subsample is what we expect
        assert u_sampled_sub.shape == (self.n_replicas, self.n_states, Neff_max)
        # Generate MBAR from phase
        phase_mbar = phase.mbar
        # Assert mbar object is formed of nstates + unsampled states, Number of effective samples
        n_effective_samples = Neff_max - discard  # The analysis discards the minimization frame.
        assert phase_mbar.u_kn.shape == (self.n_states + 2, n_effective_samples*self.n_replicas)
        # Check that Free energies are returned correctly
        fe, dfe = self.help_fe_calc(phase)
        assert fe.shape == (self.n_states + 2, self.n_states + 2)
        stored_fe_dict = phase._computed_observables['free_energy']
        stored_fe, stored_dfe = stored_fe_dict['value'], stored_fe_dict['error']
        assert np.all(stored_fe == fe), '{}, {}'.format(stored_fe, fe)
        assert np.all(stored_dfe == dfe), '{}, {}'.format(stored_dfe, dfe)
        # Test reference states and full work up creation
        iinit, jinit = phase.reference_states
        output = phase.analyze_phase()
        fe_out, dfe_out = output['DeltaF'], output['dDeltaF']
        assert fe_out == fe[iinit, jinit]
        assert dfe_out == dfe[iinit, jinit]
        inew, jnew = 1, 2
        phase.reference_states = [inew, jnew]
        new_output = phase.analyze_phase()
        new_fe_out, new_dfe_out = new_output['DeltaF'], new_output['dDeltaF']
        assert new_fe_out == fe[inew, jnew]
        assert new_dfe_out == dfe[inew, jnew]

    def help_fe_calc(self, phase):
        try:
            fe, dfe = phase.get_free_energy()
        except ParameterError as e:
            # Handle case where MBAR does not have a converged free energy yet by attempting to run longer
            # Only run up until we have sampled every state, or we hit some cycle limit
            self.reporter.open(mode='a')
            cycle_limit = 20  # Put some upper limit of cycles
            cycles = 0
            cycle_steps = 20
            throw = True
            phase.clear()
            while (not np.unique(self.sampler._reporter.read_replica_thermodynamic_states()).size == self.N_STATES
                   or cycles == cycle_limit):
                self.sampler.extend(cycle_steps)
                try:
                    fe, dfe = phase.get_free_energy()
                except ParameterError:
                    # If the max error count internally is reached, its a RuntimeError and won't be trapped
                    # So it will be raised correctly
                    pass
                else:
                    # Test is good, let it pass by returning here
                    throw = False
                    break
                cycles += 1
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
        full_phase = self.ANALYZER(self.reporter, name=self.repex_name)
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
        assert dfree_energy_full_fe == np.sqrt(full_dfree_energy**2 + fe_phase.dfev**2)
        # Check by phase (should only yield 1 value, no error)
        combo_standard_state = full_fes.get_standard_state_correction()
        assert combo_standard_state == full_phase.get_standard_state_correction() + fes_phase.ssc
        # Check compound multiphase stuff
        quad_free_energy, quad_dfree_energy = quad.get_free_energy()
        assert quad_free_energy == 0
        assert quad_dfree_energy == np.sqrt(4*full_dfree_energy**2)
        assert triple.get_standard_state_correction() == (2*full_phase.get_standard_state_correction() -
                                                          full_phase.get_standard_state_correction())

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
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name)

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

        # By default, all iterations are analyzed.
        all_decorrelated_iterations = analyzer._decorrelated_iterations
        all_decorrelated_u_ln = analyzer._decorrelated_u_ln
        n_iterations_ln = analyzer.n_replicas * analyzer.n_iterations
        assert len(all_decorrelated_iterations) == analyzer.n_iterations
        assert all_decorrelated_u_ln.shape[1] == n_iterations_ln

        # Setting max_n_iterations reduces the number of iterations analyzed
        new_max_n_iterations = analyzer.n_iterations - 1
        analyzer.max_n_iterations = new_max_n_iterations
        assert len(analyzer._decorrelated_iterations) == new_max_n_iterations
        assert np.array_equal(analyzer._decorrelated_iterations, all_decorrelated_iterations[:-1])
        assert analyzer._decorrelated_u_ln.shape[1] == analyzer.n_replicas * new_max_n_iterations

        expected_iterations_ln_dropped = set(range(analyzer.n_iterations-1, n_iterations_ln, analyzer.n_iterations))
        expected_iterations_ln_kept = sorted(set(range(n_iterations_ln)) - expected_iterations_ln_dropped)
        expected_decorrelated_u_ln = all_decorrelated_u_ln[:, expected_iterations_ln_kept]
        assert np.array_equal(analyzer._decorrelated_u_ln, expected_decorrelated_u_ln)

    def test_unbias_restraint(self):
        """Test that restraints are unbiased correctly when unbias_restraint is True."""
        # The energies of the unbiased restraint should be identical to the biased ones.
        analyzer = self.ANALYZER(self.reporter, name=self.repex_name, unbias_restraint=False)
        assert np.array_equal(analyzer._unbiased_decorrelated_u_ln, analyzer._decorrelated_u_ln)
        assert np.array_equal(analyzer._unbiased_decorrelated_N_l, analyzer._decorrelated_N_l)

        # Switch unbias_restraint to True. The old cached value is invalidated and
        # now u_ln and N_l have two extra states.
        analyzer.unbias_restraint = True
        assert analyzer._unbiased_decorrelated_u_ln.shape[0] == analyzer._decorrelated_u_ln.shape[0] + 2
        assert analyzer._unbiased_decorrelated_N_l.shape[0] == analyzer._decorrelated_N_l.shape[0] + 2
        # The automatic cutoff (default value) should remove some of the iterations.
        assert analyzer._unbiased_decorrelated_u_ln.shape[1] < analyzer._decorrelated_u_ln.shape[1]
        # The energy without the restraint should always be smaller.
        assert np.all(analyzer._unbiased_decorrelated_u_ln[0] < analyzer._unbiased_decorrelated_u_ln[1])
        assert np.all(analyzer._unbiased_decorrelated_u_ln[-1] < analyzer._unbiased_decorrelated_u_ln[-2])
        assert analyzer._unbiased_decorrelated_N_l[0] == 0
        assert analyzer._unbiased_decorrelated_N_l[-1] == 0

        # With a ver big energy cutoff, all the energies besides the extra two states should be identical.
        analyzer.restraint_energy_cutoff = 100.0  # kT
        assert np.array_equal(analyzer._unbiased_decorrelated_u_ln[1:-1], analyzer._decorrelated_u_ln)
        assert np.array_equal(analyzer._unbiased_decorrelated_N_l[1:-1], analyzer._decorrelated_N_l)

    def test_extract_trajectory(self):
        """extract_trajectory handles checkpointing and skip frame correctly."""
        trajectory = analyze.extract_trajectory(self.reporter.filepath, replica_index=0, skip_frame=2)
        assert len(trajectory) == 1
        self.reporter.close()
        full_trajectory = analyze.extract_trajectory(self.reporter.filepath, replica_index=0, keep_solvent=True)
        # This should work since its pure Python integer division
        # Follows the checkpoint interval logic
        # The -1 is because frame 0 is discarded from trajectory extraction due to equilibration problems
        # Should this change in analyze, then this logic will need to be changed as well
        assert len(full_trajectory) == ((self.n_steps + 1) / self.checkpoint_interval) - 1
        self.reporter.close()
        # Make sure the "solute"-only (analysis atoms) trajectory has the correct properties
        solute_trajectory = analyze.extract_trajectory(self.reporter.filepath, replica_index=0, keep_solvent=False)
        assert len(solute_trajectory) == self.n_steps - 1
        assert solute_trajectory.n_atoms == len(self.analysis_atoms)


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


class TestSAMSAnalyzerSingle(TestMultiPhaseAnalyzer):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 1
    N_STATES = 5
    SAMPLER = SAMSSampler


class TestSAMSAnalyzerMulti(TestSAMSAnalyzerSingle):
    """Test suite for YankPhaseAnalyzer class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 3
