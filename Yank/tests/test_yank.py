#!/usr/local/bin/env python

"""
Test YANK using simple models.

DESCRIPTION

This test suite generates a number of simple models to test the 'Yank' facility.

COPYRIGHT

Written by John D. Chodera <jchodera@gmail.com> while at the University of California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import contextlib
import os

import nose
from openmmtools import testsystems, states

import yank.restraints
from yank.multistate import ReplicaExchangeSampler
from yank.yank import *


# ==============================================================================
# TEST UTILITIES
# ==============================================================================

def prepare_yield(func, test_case_name, *args):
    """Create a description for a test function to yield."""
    f = functools.partial(func, *args)
    f.description = test_case_name + ': ' + func.__doc__
    return f


# ==============================================================================
# TEST TOPOLOGY OBJECT
# ==============================================================================

def test_topography():
    """Test that topology components are isolated correctly by Topography."""
    toluene_vacuum = testsystems.TolueneVacuum()
    topography = Topography(toluene_vacuum.topology)
    assert len(topography.ligand_atoms) == 0
    assert len(topography.receptor_atoms) == 0
    assert topography.solute_atoms == list(range(toluene_vacuum.system.getNumParticles()))
    assert len(topography.solvent_atoms) == 0
    assert len(topography.ions_atoms) == 0

    host_guest_explicit = testsystems.HostGuestExplicit()
    topography = Topography(host_guest_explicit.topology, ligand_atoms='resname B2')
    assert topography.ligand_atoms == list(range(126, 156))
    assert topography.receptor_atoms == list(range(126))
    assert topography.solute_atoms == list(range(156))
    assert topography.solvent_atoms == list(range(156, host_guest_explicit.system.getNumParticles()))
    assert len(topography.ions_atoms) == 0


def test_topography_subset_regions():
    """Test that topography subset region selection works"""
    # This test relies on all other tests for topography passing
    host_guest_explicit = testsystems.HostGuestExplicit()
    topography = Topography(host_guest_explicit.topology, ligand_atoms='resname B2')
    ligand_list = list(range(126, 156))
    receptor_list = list(range(126))
    n_slice = 3  # Number of atoms to slice from front and back
    assert topography.ligand_atoms == ligand_list
    topography.add_region('lig_dsl', 'resname B2')
    topography.add_region('lig_list', ligand_list)
    topography.add_region('rec_list', receptor_list)
    topography.add_region('rec_lig_slice', receptor_list[-n_slice:] + ligand_list[:n_slice])
    # Selection list: selection, subset, sort_by, result
    selections = (
        # DSL select, same DSL subset, small to large
        ('resname B2', 'resname B2', 'index', ligand_list),
        # DSL select, region subset
        ('resname B2', 'lig_list', 'index', ligand_list),
        # region, dsl, small to large
        ('lig_list', 'resname B2', 'index', ligand_list),
        # region, region that is subset, small to large
        ('lig_list', 'rec_lig_slice', 'index', ligand_list[:n_slice]),
        # indices, region thats a subset, index. Remember, if given a subset: selection is RELATIVE
        (receptor_list, 'rec_lig_slice', 'index', topography.get_region('rec_lig_slice')),
        # region, region that is NOT part of selection, index
        ('lig_list', 'resname CB7', 'index', []),
        # compound Region, partial subset in bad order, order of region
        ('lig_list or rec_list', 'rec_lig_slice', 'region_order', ligand_list[:n_slice] + receptor_list[-n_slice:])
                )
    for selection, subset, sort_by, result in selections:
        selected = topography.select(selection, sort_by=sort_by, subset=subset, as_set=False)
        assert result == selected, "Failed to match {}, subset {}, by {} to {},\ngot {}".format(selection, subset,
                                                                                                sort_by, result,
                                                                                                selected)


def test_topography_regions():
    """Test that topography regions are created and fetched"""
    toluene_vacuum = testsystems.TolueneVacuum()
    topography = Topography(toluene_vacuum.topology)
    # Should do nothing and return nothing without error
    assert topography.remove_region('Nothing') is None
    topography.add_region('A hard list', [0, 1, 2])
    assert 'A hard list' in topography
    topography.add_region('A junk list', [5, 5, 5])
    topography.remove_region('A junk list')
    assert 'A junk list' not in topography
    assert topography.get_region('A hard list') == [0, 1, 2]
    # Confirm that string typing is handled
    topography.add_region('carbon', 'element C')
    assert len(topography.get_region('carbon')) > 0
    with nose.tools.assert_raises(ValueError):
        topography.add_region('failure', 'Bad selection string')
    # Ensure region was not added
    assert 'failure' not in topography


def test_topography_region_ordering():
    """Test that the region sorting algorithm is working as intended"""
    toluene_vacuum = testsystems.TolueneVacuum()
    topography = Topography(toluene_vacuum.topology)
    region_A = [2, 3, 1, 0]
    region_B = [3, 2, 4, 5]
    topography.add_region('region_A', region_A)
    topography.add_region('region_B', region_B)
    region_set = topography.select('region_A and region_B', as_set=True)
    assert region_set == set(region_A) & set(region_B)
    # Test ordered sorting
    region_A_and_B_ordered = topography.select('region_A and region_B', sort_by='index')
    region_B_and_A_ordered = topography.select('region_B and region_A', sort_by='index')
    assert region_A_and_B_ordered == [2, 3]
    assert region_A_and_B_ordered == region_B_and_A_ordered
    # Test by region
    region_A_and_B_inferred = topography.select('region_A and region_B', sort_by='region_order')
    region_B_and_A_inferred = topography.select('region_B and region_A', sort_by='region_order')
    assert region_A_and_B_inferred == [2, 3]
    assert region_B_and_A_inferred == [3, 2]
    # Test the or sorting
    region_A_or_B_ordered = topography.select('region_A or region_B', sort_by='index')
    region_B_or_A_ordered = topography.select('region_B or region_A', sort_by='index')
    assert region_A_or_B_ordered == sorted(list(set(region_A) | set(region_B)))
    assert region_A_or_B_ordered == region_B_or_A_ordered
    region_A_or_B_inferred = topography.select('region_A or region_B', sort_by='region_order')
    region_B_or_A_inferred = topography.select('region_B or region_A', sort_by='region_order')
    assert region_A_or_B_inferred == region_A + [i for i in region_B if i not in region_A]
    assert region_B_or_A_inferred == region_B + [i for i in region_A if i not in region_B]


def test_topography_serialization():
    """Correct serialization of Topography objects."""
    test_system = testsystems.AlanineDipeptideImplicit()
    topography = Topography(test_system.topology)
    topography.add_region('atest', [0, 1, 2, 3])
    serialized_topography = mmtools.utils.serialize(topography)
    restored_topography = mmtools.utils.deserialize(serialized_topography)
    assert topography.topology == restored_topography.topology
    assert topography.ligand_atoms == restored_topography.ligand_atoms
    assert topography.solvent_atoms == restored_topography.solvent_atoms
    assert topography.get_region('atest') == restored_topography.get_region('atest')


# ==============================================================================
# TEST ALCHEMICAL PHASE
# ==============================================================================

class TestAlchemicalPhase(object):
    """Test suite for class AlchemicalPhase."""

    @classmethod
    def setup_class(cls):
        """Shared test cases for the suite."""
        temperature = 300 * unit.kelvin

        # Default protocols for tests.
        cls.protocol = dict(lambda_electrostatics=[1.0, 0.5, 0.0, 0.0, 0.0],
                            lambda_sterics=[1.0, 1.0, 1.0, 0.5, 0.0])
        cls.restrained_protocol = dict(lambda_electrostatics=[1.0, 1.0, 0.0, 0.0],
                                       lambda_sterics=[1.0, 1.0, 1.0, 0.0],
                                       lambda_restraints=[0.0, 1.0, 1.0, 1.0])

        # Ligand-receptor in implicit solvent.
        test_system = testsystems.HostGuestImplicit()
        thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                        temperature=temperature)
        sampler_state = states.SamplerState(positions=test_system.positions, box_vectors=test_system.system.getDefaultPeriodicBoxVectors())
        topography = Topography(test_system.topology, ligand_atoms='resname B2')
        cls.host_guest_implicit = ('Host-guest implicit', thermodynamic_state, sampler_state, topography)

        # Ligand-receptor in explicit solvent.
        test_system = testsystems.HostGuestExplicit()
        thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                        temperature=temperature)
        positions = test_system.positions
        box_vectors = test_system.system.getDefaultPeriodicBoxVectors()
        sampler_state = states.SamplerState(positions=positions, box_vectors=box_vectors)
        topography = Topography(test_system.topology, ligand_atoms='resname B2')
        cls.host_guest_explicit = ('Host-guest explicit', thermodynamic_state, sampler_state, topography)

        # Peptide solvated in explicit solvent.
        test_system = testsystems.AlanineDipeptideExplicit()
        thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                        temperature=temperature)
        positions = test_system.positions
        box_vectors = test_system.system.getDefaultPeriodicBoxVectors()
        sampler_state = states.SamplerState(positions=positions, box_vectors=box_vectors)
        topography = Topography(test_system.topology)
        cls.alanine_explicit = ('Alanine dipeptide explicit', thermodynamic_state, sampler_state, topography)

        # All test cases
        cls.all_test_cases = [cls.host_guest_implicit, cls.host_guest_explicit, cls.alanine_explicit]

    @staticmethod
    @contextlib.contextmanager
    def temporary_storage_path():
        """Generate a storage path in a temporary folder and share it.

        It makes it possible to run tests on multiple nodes with MPI.

        """
        mpicomm = mpi.get_mpicomm()
        with mmtools.utils.temporary_directory() as tmp_dir_path:
            storage_file_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            if mpicomm is not None:
                storage_file_path = mpicomm.bcast(storage_file_path, root=0)
            yield storage_file_path

    @staticmethod
    def check_protocol(alchemical_phase, protocol):
        """The compound thermodynamic states created follow the requested protocol."""
        for i, state in enumerate(alchemical_phase._sampler._thermodynamic_states):
            for protocol_key, protocol_values in protocol.items():
                assert getattr(state, protocol_key) == protocol_values[i]

    @staticmethod
    def check_standard_state_correction(alchemical_phase, topography):
        """AlchemicalPhase carries the correct standard state correction."""
        is_complex = len(topography.ligand_atoms) > 0
        metadata = alchemical_phase._sampler.metadata
        standard_state_correction = metadata['standard_state_correction']
        if is_complex:
            assert standard_state_correction != 0
        else:
            assert standard_state_correction == 0

    @classmethod
    def check_expanded_states(cls, alchemical_phase, protocol, expected_cutoff, reference_system):
        """The expanded states have been setup correctly."""
        thermodynamic_state = alchemical_phase._sampler._thermodynamic_states[0]
        unsampled_states = alchemical_phase._sampler._unsampled_states

        is_periodic = thermodynamic_state.is_periodic
        is_restrained = hasattr(thermodynamic_state, 'lambda_restraints')

        # If the expanded cutoff is determined automatically,
        # figure out the expected value.
        if expected_cutoff == 'auto':
            box_vectors = thermodynamic_state._standard_system.getDefaultPeriodicBoxVectors()
            min_box_dimension = min([vector[i] for i, vector in enumerate(box_vectors)])
            if thermodynamic_state.pressure is None:
                min_box_half_size = min_box_dimension * 0.99 / 2.0
            else:
                min_box_half_size = min_box_dimension * 0.8 / 2.0
            expected_cutoff = min(min_box_half_size, 16*unit.angstrom)
        cutoff_unit = expected_cutoff.unit

        # Anisotropic correction is not applied to non-periodic systems.
        if not is_periodic:
            assert len(unsampled_states) == 0
            return
        assert len(unsampled_states) == 2

        # The switch distances in the nonbonded forces should be preserved.
        # Get the original(s) from the reference system. We store 5-decimal
        # strings to avoid having problems with precision in sets.
        expected_switch_widths = set()
        for force in reference_system.getForces():
            try:
                switch_width = force.getCutoffDistance() - force.getSwitchingDistance()
                expected_switch_widths.add('{:.5}'.format(switch_width/unit.nanometers))
            except AttributeError:
                pass

        # Find all cutoff and switch width for the reference and unsampled states.
        for state in unsampled_states:
            switch_widths = set()
            system = state.system
            for force in system.getForces():
                try:
                    cutoff = force.getCutoffDistance()
                except AttributeError:
                    continue
                else:
                    err_msg = 'obtained {}, expected {}'.format(cutoff, expected_cutoff)
                    assert np.isclose(cutoff / cutoff_unit, expected_cutoff / cutoff_unit), err_msg
                    if force.getUseSwitchingFunction():
                        switch_width = cutoff - force.getSwitchingDistance()
                        switch_widths.add('{:.5}'.format(switch_width/unit.nanometers))

            # The unsampled state should preserve all switch widths of the reference system.
            err_msg = 'obtained {}, expected {}'.format(switch_widths, expected_switch_widths)
            assert switch_widths == expected_switch_widths, err_msg

        # If the thermodynamic systems are restrained, so should be the unsampled ones.
        if is_restrained:
            for i, state in zip([0, -1], unsampled_states):
                assert state.lambda_restraints == cls.restrained_protocol['lambda_restraints'][i]

        # The noninteracting state, must be in the same state as the last one.
        noninteracting_state = unsampled_states[1]
        for protocol_key, protocol_values in protocol.items():
            assert getattr(noninteracting_state, protocol_key) == protocol_values[-1]

    def test_create(self):
        """Alchemical state correctly creates the simulation object."""
        available_restraints = list(yank.restraints.available_restraint_classes().values())

        for test_index, test_case in enumerate(self.all_test_cases):
            test_name, thermodynamic_state, sampler_state, topography = test_case

            # Add random restraint if this is ligand-receptor system in implicit solvent.
            if len(topography.ligand_atoms) > 0:
                restraint_cls = available_restraints[np.random.randint(0, len(available_restraints))]
                restraint = restraint_cls()
                protocol = self.restrained_protocol
                test_name += ' with restraint {}'.format(restraint_cls.__name__)
            else:
                restraint = None
                protocol = self.protocol

            # Add either automatic of fixed correction cutoff.
            if test_index % 2 == 0:
                correction_cutoff = 12 * unit.angstroms
            else:
                correction_cutoff = 'auto'

            # Replace the reaction field of the reference system to compare
            # also cutoff and switch width for the electrostatics.
            reference_system = thermodynamic_state.system
            mmtools.forcefactories.replace_reaction_field(reference_system, return_copy=False)

            alchemical_phase = AlchemicalPhase(sampler=ReplicaExchangeSampler())
            with self.temporary_storage_path() as storage_path:
                alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                        protocol, storage_path, restraint=restraint,
                                        anisotropic_dispersion_cutoff=correction_cutoff)

                yield prepare_yield(self.check_protocol, test_name, alchemical_phase, protocol)
                yield prepare_yield(self.check_standard_state_correction, test_name, alchemical_phase,
                                    topography)
                yield prepare_yield(self.check_expanded_states, test_name, alchemical_phase,
                                    protocol, correction_cutoff, reference_system)

                # Free memory.
                del alchemical_phase

    def test_default_alchemical_region(self):
        """The default alchemical region modify the correct system elements."""
        # We test two protocols: with and without alchemically
        # modified bonds, angles and torsions.
        simple_protocol = dict(lambda_electrostatics=[1.0, 0.5, 0.0, 0.0, 0.0],
                               lambda_sterics=[1.0, 1.0, 1.0, 0.5, 0.0])
        full_protocol = dict(lambda_electrostatics=[1.0, 0.5, 0.0, 0.0, 0.0],
                             lambda_sterics=[1.0, 1.0, 1.0, 0.5, 0.0],
                             lambda_bonds=[1.0, 1.0, 1.0, 0.5, 0.0],
                             lambda_angles=[1.0, 1.0, 1.0, 0.5, 0.0],
                             lambda_torsions=[1.0, 1.0, 1.0, 0.5, 0.0])
        protocols = [simple_protocol, full_protocol]

        # Create two test systems with a charged solute and counterions.
        sodium_chloride = testsystems.SodiumChlorideCrystal()
        negative_solute = Topography(sodium_chloride.topology, solvent_atoms=[0])
        positive_solute = Topography(sodium_chloride.topology, solvent_atoms=[1])

        # Test case: (system, topography, topography_region)
        test_cases = [
            (self.host_guest_implicit[1].system, self.host_guest_implicit[3], 'ligand_atoms'),
            (self.host_guest_explicit[1].system, self.host_guest_explicit[3], 'ligand_atoms'),
            (self.alanine_explicit[1].system, self.alanine_explicit[3], 'solute_atoms'),
            (sodium_chloride.system, negative_solute, 'solute_atoms'),
            (sodium_chloride.system, positive_solute, 'solute_atoms')
        ]

        for system, topography, alchemical_region_name in test_cases:
            expected_alchemical_atoms = getattr(topography, alchemical_region_name)

            # Compute net charge of alchemical atoms.
            net_charge = pipeline.compute_net_charge(system, expected_alchemical_atoms)
            if net_charge != 0:
                # Sodium chloride test systems.
                expected_alchemical_atoms = [0, 1]

            for protocol in protocols:
                alchemical_region = AlchemicalPhase._build_default_alchemical_region(
                    system, topography, protocol)
                assert alchemical_region.alchemical_atoms == expected_alchemical_atoms

                # The alchemical region must be neutralized.
                assert pipeline.compute_net_charge(system, expected_alchemical_atoms) == 0

                # We modify elements other than sterics and electrostatics when requested.
                if 'lambda_bonds' in protocol:
                    assert alchemical_region.alchemical_bonds is True
                    assert alchemical_region.alchemical_angles is True
                    assert alchemical_region.alchemical_torsions is True

    def test_non_alchemical_protocol(self):
        """Any property of the ThermodynamicSystem can be specified in the protocol."""
        name, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        protocol = {
            'lambda_sterics': [0.0, 0.5, 1.0],
            'temperature': [300, 320, 300] * unit.kelvin,
            'update_alchemical_charges': [True, True, False]
        }
        alchemical_phase = AlchemicalPhase(sampler=ReplicaExchangeSampler())
        with self.temporary_storage_path() as storage_path:
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    protocol, storage_path, restraint=yank.restraints.Harmonic())
            self.check_protocol(alchemical_phase, protocol)

        # If temperatures of the end states is different, an error is raised.
        protocol['temperature'][-1] = 330 * unit.kelvin
        alchemical_phase = AlchemicalPhase(sampler=ReplicaExchangeSampler())
        with nose.tools.assert_raises(ValueError):
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    protocol, 'not_created.nc', restraint=yank.restraints.Harmonic())

    def test_illegal_protocol(self):
        """An error is raised when the protocol parameters have a different number of states."""
        name, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        protocol = {
            'lambda_sterics': [0.0, 1.0],
            'lambda_electrostatics': [1.0]
        }
        alchemical_phase = AlchemicalPhase(sampler=ReplicaExchangeSampler())
        restraint = yank.restraints.Harmonic()
        with nose.tools.assert_raises(ValueError):
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    protocol, 'not_created.nc', restraint=restraint)

    def test_illegal_restraint(self):
        """Raise an error when restraint is handled incorrectly."""
        # An error is raised with ligand-receptor systems in implicit without restraint.
        test_name, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        alchemical_phase = AlchemicalPhase(sampler=ReplicaExchangeSampler())
        with nose.tools.assert_raises(ValueError):
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    self.protocol, 'not_created.nc')

        # An error is raised when trying to apply restraint to non ligand-receptor systems.
        restraint = yank.restraints.Harmonic()
        test_name, thermodynamic_state, sampler_state, topography = self.alanine_explicit
        with nose.tools.assert_raises(RuntimeError):
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    self.protocol, 'not_created.nc', restraint=restraint)

    def test_from_storage(self):
        """When resuming, the AlchemicalPhase recover the correct sampler."""
        _, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        restraint = yank.restraints.Harmonic()

        with self.temporary_storage_path() as storage_path:
            alchemical_phase = AlchemicalPhase(ReplicaExchangeSampler())
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    self.protocol, storage_path, restraint=restraint)

            # Delete old alchemical phase to close storage file.
            del alchemical_phase

            # Resume, the sampler has the correct class.
            alchemical_phase = AlchemicalPhase.from_storage(storage_path)
            assert isinstance(alchemical_phase._sampler, ReplicaExchangeSampler)

    @staticmethod
    def test_find_similar_sampler_states():
        """Test helper method AlchemicalPhase._find_similar_sampler_states."""
        sampler_state1 = states.SamplerState(np.random.rand(100, 3))
        sampler_state2 = states.SamplerState(np.random.rand(100, 3))
        sampler_state3 = states.SamplerState(np.random.rand(100, 3))

        sampler_states = [sampler_state1, sampler_state1, sampler_state2,
                          sampler_state1, sampler_state3, sampler_state2]
        similar_states = AlchemicalPhase._find_similar_sampler_states(sampler_states)
        assert similar_states == {0: [1, 3], 2: [5], 4: []}

    def test_minimize(self):
        """Test AlchemicalPhase minimization of positions in reference state."""
        # Ligand-receptor in implicit solvent.
        test_system = testsystems.AlanineDipeptideVacuum()
        thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                        temperature=300*unit.kelvin)
        topography = Topography(test_system.topology)

        # We create 3 different sampler states that will be distributed over
        # replicas in a round-robin fashion.
        displacement_vector = np.ones(3) * unit.nanometer
        positions2 = test_system.positions + displacement_vector
        positions3 = positions2 + displacement_vector
        box_vectors = test_system.system.getDefaultPeriodicBoxVectors()
        sampler_state1 = states.SamplerState(positions=test_system.positions, box_vectors=box_vectors)
        sampler_state2 = states.SamplerState(positions=positions2, box_vectors=box_vectors)
        sampler_state3 = states.SamplerState(positions=positions3, box_vectors=box_vectors)
        sampler_states = [sampler_state1, sampler_state2, sampler_state3]

        with self.temporary_storage_path() as storage_path:
            # Create alchemical phase.
            alchemical_phase = AlchemicalPhase(ReplicaExchangeSampler())
            alchemical_phase.create(thermodynamic_state, sampler_states, topography,
                                    self.protocol, storage_path)

            # Measure the average distance between positions. This should be
            # maintained after minimization.
            sampler_states = alchemical_phase._sampler.sampler_states
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                              for i in range(len(sampler_states) - 1)]

            # Minimize.
            alchemical_phase.minimize()

            # The minimized positions should be still more or less
            # one displacement vector from each other.
            sampler_states = alchemical_phase._sampler.sampler_states
            new_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                         for i in range(len(sampler_states) - 1)]
            assert np.allclose(original_diffs, new_diffs)

    def test_randomize_ligand(self):
        """Test method AlchemicalPhase.randomize_ligand."""
        _, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        restraint = yank.restraints.Harmonic()

        ligand_atoms, receptor_atoms = topography.ligand_atoms, topography.receptor_atoms
        ligand_positions = sampler_state.positions[ligand_atoms]
        receptor_positions = sampler_state.positions[receptor_atoms]

        with self.temporary_storage_path() as storage_path:
            alchemical_phase = AlchemicalPhase(ReplicaExchangeSampler())
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    self.protocol, storage_path, restraint=restraint)

            # Randomize ligand positions.
            alchemical_phase.randomize_ligand()

            # The new sampler states have the same receptor positions
            # but different ligand positions.
            for sampler_state in alchemical_phase._sampler.sampler_states:
                assert np.allclose(sampler_state.positions[receptor_atoms], receptor_positions)
                assert not np.allclose(sampler_state.positions[ligand_atoms], ligand_positions)


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

def notest_LennardJonesPair(**kwargs):
    pass

# LNN: Silenced this function since its no longer used, we should resolve it better
# def notest_LennardJonesPair(box_width_nsigma=6.0):
#     """
#     Compute binding free energy of two Lennard-Jones particles and compare to numerical result.
#
#     Parameters
#     ----------
#     box_width_nsigma : float, optional, default=6.0
#         Box width is set to this multiple of Lennard-Jones sigma.
#
#     """
#
#     NSIGMA_MAX = 6.0 # number of standard errors tolerated for success
#
#     # Create Lennard-Jones pair.
#     thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
#     kT = kB * thermodynamic_state.temperature
#     sigma = 3.5 * unit.angstroms
#     epsilon = 6.0 * kT
#     test = testsystems.LennardJonesPair(sigma=sigma, epsilon=epsilon)
#     system, positions = test.system, test.positions
#     binding_free_energy = test.get_binding_free_energy(thermodynamic_state)
#
#     # Create temporary directory for testing.
#     import tempfile
#     store_dir = tempfile.mkdtemp()
#
#     # Initialize YANK object.
#     options = dict()
#     options['number_of_iterations'] = 10
#     options['platform'] = openmm.Platform.getPlatformByName("Reference") # use Reference platform for speed
#     options['mc_rotation'] = False
#     options['mc_displacement'] = True
#     options['mc_displacement_sigma'] = 1.0 * unit.nanometer
#     options['timestep'] = 2 * unit.femtoseconds
#     options['nsteps_per_iteration'] = 50
#
#     # Override receptor mass to keep it stationary.
#     #system.setParticleMass(0, 0)
#
#     # Override box vectors.
#     box_edge = 6*sigma
#     a = unit.Quantity((box_edge, 0 * unit.angstrom, 0 * unit.angstrom))
#     b = unit.Quantity((0 * unit.angstrom, box_edge, 0 * unit.angstrom))
#     c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
#     system.setDefaultPeriodicBoxVectors(a, b, c)
#
#     # Override positions
#     positions[0,:] = box_edge/2
#     positions[1,:] = box_edge/4
#
#     phase = 'complex-explicit'
#
#     # Alchemical protocol.
#     from yank.alchemy import AlchemicalState
#     alchemical_states = list()
#     lambda_values = [0.0, 0.25, 0.50, 0.75, 1.0]
#     for lambda_value in lambda_values:
#         alchemical_state = AlchemicalState()
#         alchemical_state['lambda_electrostatics'] = lambda_value
#         alchemical_state['lambda_sterics'] = lambda_value
#         alchemical_states.append(alchemical_state)
#     protocols = dict()
#     protocols[phase] = alchemical_states
#
#     # Create phases.
#     alchemical_phase = AlchemicalPhase(phase, system, test.topology, positions,
#                                        {'complex-explicit': {'ligand': [1]}},
#                                        alchemical_states)
#
#     # Create new simulation.
#     yank = Yank(store_dir, **options)
#     yank.create(thermodynamic_state, alchemical_phase)
#
#     # Run the simulation.
#     yank.run()
#
#     # Analyze the data.
#     results = yank.analyze()
#     standard_state_correction = results[phase]['standard_state_correction']
#     Delta_f = results[phase]['Delta_f_ij'][0,1] - standard_state_correction
#     dDelta_f = results[phase]['dDelta_f_ij'][0,1]
#     nsigma = abs(binding_free_energy/kT - Delta_f) / dDelta_f
#
#     # Check results against analytical results.
#     # TODO: Incorporate standard state correction
#     output = "\n"
#     output += "Analytical binding free energy                                  : %10.5f +- %10.5f kT\n" % (binding_free_energy / kT, 0)
#     output += "Computed binding free energy (with standard state correction)   : %10.5f +- %10.5f kT (nsigma = %3.1f)\n" % (Delta_f, dDelta_f, nsigma)
#     output += "Computed binding free energy (without standard state correction): %10.5f +- %10.5f kT (nsigma = %3.1f)\n" % (Delta_f + standard_state_correction, dDelta_f, nsigma)
#     output += "Standard state correction alone                                 : %10.5f           kT\n" % (standard_state_correction)
#     print(output)
#
#     #if (nsigma > NSIGMA_MAX):
#     #    output += "\n"
#     #    output += "Computed binding free energy differs from true binding free energy.\n"
#     #    raise Exception(output)
#
#     return [Delta_f, dDelta_f]

if __name__ == '__main__':
    from yank import utils
    utils.config_root_logger(True, log_file_path='test_LennardJones_pair.log')

    box_width_nsigma_values = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    Delta_f_n = list()
    dDelta_f_n = list()
    for (n, box_width_nsigma) in enumerate(box_width_nsigma_values):
        [Delta_f, dDelta_f] = notest_LennardJonesPair(box_width_nsigma=box_width_nsigma)
        Delta_f_n.append(Delta_f)
        dDelta_f_n.append(dDelta_f)
    Delta_f_n = np.array(Delta_f_n)
    dDelta_f_n = np.array(dDelta_f_n)

    for (box_width_nsigma, Delta_f, dDelta_f) in zip(box_width_nsigma_values, Delta_f_n, dDelta_f_n):
        print("%8.3f %12.6f %12.6f" % (box_width_nsigma, Delta_f, dDelta_f))
