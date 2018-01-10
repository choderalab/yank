#!/usr/local/bin/env python

"""
Test multistate.analyzers facility.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import copy
import os
import shutil
import tempfile

import numpy as np
import openmmtools as mmtools
import pymbar
from nose.tools import assert_raises
from openmmtools import testsystems
from simtk import unit

from yank.yank import Topography
from yank.multistate import MultiStateReporter, ReplicaExchangeSampler
import yank.multistate.analyzers as analyze

# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # Boltzmann constant


# ==============================================================================
# TEMPLATE PHASE CLASS WITH NO OBSERVABLES
# ==============================================================================

class BlankPhase(analyze.PhaseAnalyzer):
    """Create a blank phase class with no get_X (observable) methods for testing the MultiPhaseAnalyzer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze_phase(self, *args, **kwargs):
        pass

    def _create_mbar_from_scratch(self):
        pass

    def get_states_energies(self):
        pass

    @staticmethod
    def get_timeseries(passed_timeseries):
        pass

    def _prepare_mbar_input_data(self):
        pass


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

class TestPhaseAnalyzer(object):
    """Test suite for YankPhaseAnalyzer class."""

    @classmethod
    def setup_class(cls):
        """Shared test cases and variables."""
        n_states = 3
        n_steps = 5
        checkpoint_interval = 2

        # Test case with host guest in vacuum at 3 different positions and alchemical parameters.
        # -----------------------------------------------------------------------------------------
        hostguest_test = testsystems.HostGuestVacuum()
        factory = mmtools.alchemy.AbsoluteAlchemicalFactory()
        alchemical_region = mmtools.alchemy.AlchemicalRegion(alchemical_atoms=range(126, 156))
        hostguest_alchemical = factory.create_alchemical_system(hostguest_test.system, alchemical_region)

        # Translate the sampler states to be different one from each other.
        positions = hostguest_test.positions
        box_vectors = hostguest_test.system.getDefaultPeriodicBoxVectors()
        hostguest_sampler_states = [mmtools.states.SamplerState(positions=positions + 10*i*unit.nanometers,
                                                                box_vectors=box_vectors)
                                    for i in range(n_states)]

        # Create the three basic thermodynamic states.
        hostguest_thermodynamic_states = [mmtools.states.ThermodynamicState(hostguest_alchemical, 300*unit.kelvin)
                                          for i in range(n_states)]

        # Create alchemical states at different parameter values.
        alchemical_states = [mmtools.alchemy.AlchemicalState.from_system(hostguest_alchemical)
                             for _ in range(n_states)]
        for i, alchemical_state in enumerate(alchemical_states):
            alchemical_state.set_alchemical_parameters(float(i) / (n_states - 1))

        # Create compound states.
        hostguest_compound_states = list()
        for i in range(n_states):
            hostguest_compound_states.append(
                mmtools.states.CompoundThermodynamicState(thermodynamic_state=hostguest_thermodynamic_states[i],
                                                          composable_states=[alchemical_states[i]])
            )

        # Unsampled states.
        nonalchemical_state = mmtools.states.ThermodynamicState(hostguest_test.system, 300*unit.kelvin)
        hostguest_unsampled_states = [copy.deepcopy(nonalchemical_state), copy.deepcopy(nonalchemical_state)]

        cls.hostguest_test = (hostguest_compound_states, hostguest_sampler_states, hostguest_unsampled_states)

        # Run a quick simulation
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(cls.hostguest_test)
        n_states = len(thermodynamic_states)

        # Remove one sampler state to verify distribution over states.
        sampler_states = sampler_states[:-1]

        # Prepare metadata for analysis.
        reference_state = mmtools.states.ThermodynamicState(hostguest_test.system, 300*unit.kelvin)
        topography = Topography(hostguest_test.topology, ligand_atoms=range(126, 156))
        metadata = {
            'standard_state_correction': 4.0,
            'reference_state': mmtools.utils.serialize(reference_state),
            'topography': mmtools.utils.serialize(topography)
        }
        analysis_atoms = topography.ligand_atoms

        # Create simulation and storage file.
        cls.tmp_dir = tempfile.mkdtemp()
        storage_path = os.path.join(cls.tmp_dir, 'test_analyze.nc')
        move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
        cls.repex = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=n_steps)
        cls.reporter = MultiStateReporter(storage_path, checkpoint_interval=checkpoint_interval,
                                          analysis_particle_indices=analysis_atoms)
        cls.repex.create(thermodynamic_states, sampler_states, storage=cls.reporter,
                         unsampled_thermodynamic_states=unsampled_states, metadata=metadata)
        # run some iterations
        cls.n_states = n_states
        cls.n_steps = n_steps
        cls.checkpoint_interval = checkpoint_interval
        cls.analysis_atoms = analysis_atoms
        cls.repex.run(cls.n_steps-1)  # Initial config
        cls.repex_name = "RepexAnalyzer"

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_repex_phase_initialize(self):
        """Test that the MultiState Phase analyzer initializes correctly"""
        phase = analyze.ReplicaExchangeAnalyzer(self.reporter, name=self.repex_name)
        assert phase.reporter is self.reporter
        assert phase.name == self.repex_name

    def test_repex_mixing_stats(self):
        """Test that the MultiState Phase yields mixing stats that make sense"""
        phase = analyze.ReplicaExchangeAnalyzer(self.reporter, name=self.repex_name)
        t, mu, g = phase.generate_mixing_statistics()
        # Output is the correct number of states
        assert t.shape == (self.n_states, self.n_states)
        # Assert transition matrix values are all 0 <= x <= 1
        assert np.all(np.logical_and(t >= 0, t <= 1))
        # Assert all rows add to 1
        for row in range(self.n_states):
            assert t[row, :].sum() == 1
        # Assert all eigenvalues are all <= 1
        # Floating point error can lead mu[0] to be not exactly 1.0, but like 0.99999998 or something
        assert np.allclose(mu[0], 1)

    def test_mbar_creation_process(self):
        """Test each of the steps of the MBAR creation process

        We do this in one function since the test for each part would be a bunch of repeated recreation of the phase
        """
        phase = analyze.ReplicaExchangeAnalyzer(self.reporter, name=self.repex_name)
        u_sampled, u_unsampled = phase.get_states_energies()
        # Test energy output matches appropriate MBAR shapes
        assert u_sampled.shape == (self.n_states, self.n_states, self.n_steps)
        assert u_unsampled.shape == (self.n_states, 2, self.n_steps)
        u_n = phase.get_timeseries(u_sampled)
        assert u_n.shape == (self.n_steps,)
        # This may need to be adjusted from time to time as analyze changes
        discard = 1
        # Generate mbar semi-manually, use phases's static methods
        n_eq, g_t, Neff_max = pymbar.timeseries.detectEquilibration(u_n[discard:])
        u_sampled_sub = analyze.remove_unequilibrated_data(u_sampled, n_eq, -1)
        # Make sure output from subsample is what we expect
        assert u_sampled_sub.shape == (self.n_states, self.n_states, Neff_max)
        # Generate MBAR from phase
        phase_mbar = phase.mbar
        # Assert mbar object is formed of nstates + unsampled states, Number of effective samples
        assert phase_mbar.u_kn.shape == (self.n_states + 2, Neff_max*self.n_states)
        # Check that Free energies are returned correctly
        fe, dfe = phase.get_free_energy()
        assert fe.shape == (self.n_states + 2, self.n_states + 2)
        stored_fe_dict = phase._computed_observables['free_energy']
        stored_fe, stored_dfe = stored_fe_dict['value'], stored_fe_dict['error']
        assert np.all(stored_fe == fe)
        assert np.all(stored_dfe == dfe)
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

    def test_multi_phase(self):
        """Test MultiPhaseAnalysis"""
        # Create the phases
        full_phase = analyze.ReplicaExchangeAnalyzer(self.reporter, name=self.repex_name)
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
        free_energy_full_fe, dfree_energy_full_fe = full_fe_phase.get_free_energy()
        full_free_energy, full_dfree_energy = full_phase.get_free_energy()
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

    def test_extract_trajectory(self):
        """extract_trajectory handles checkpointing and skip frame correctly."""
        # print(self.reporter.read_last_iteration())
        trajectory = analyze.extract_trajectory(self.reporter.filepath, state_index=0, skip_frame=2)
        assert len(trajectory) == 1
        self.reporter.close()
        full_trajectory = analyze.extract_trajectory(self.reporter.filepath, state_index=0, keep_solvent=True)
        # This should work since its pure Python integer division
        # Follows the checkpoint interval logic
        # The -1 is because frame 0 is discarded from trajectory extraction due to equilibration problems
        # Should this change in analyze, then this logic will need to be changed as well
        assert len(full_trajectory) == ((self.n_steps + 1) / self.checkpoint_interval) - 1
        self.reporter.close()
        # Make sure the "solute"-only (analysis atoms) trajectory has the correct properties
        solute_trajectory = analyze.extract_trajectory(self.reporter.filepath, state_index=0, keep_solvent=False)
        assert len(solute_trajectory) == self.n_steps - 1
        assert solute_trajectory.n_atoms == len(self.analysis_atoms)
