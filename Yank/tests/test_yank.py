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

import functools
import contextlib

from openmmtools.constants import kB
from openmmtools import testsystems, states
from mdtraj.utils import enter_temp_directory

import nose
import netCDF4 as netcdf

import yank.restraints
from yank.repex import ReplicaExchange

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


def test_topography_serialization():
    """Correct serialization of Topography objects."""
    test_system = testsystems.AlanineDipeptideImplicit()
    topography = Topography(test_system.topology)
    serialized_topography = mmtools.utils.serialize(topography)
    restored_topography = mmtools.utils.deserialize(serialized_topography)
    assert topography.topology == restored_topography.topology
    assert topography.ligand_atoms == restored_topography.ligand_atoms
    assert topography.solvent_atoms == restored_topography.solvent_atoms


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
        sampler_state = states.SamplerState(test_system.positions)
        topography = Topography(test_system.topology, ligand_atoms='resname B2')
        cls.host_guest_implicit = ('Host-guest implicit', thermodynamic_state, sampler_state, topography)

        # Ligand-receptor in explicit solvent.
        test_system = testsystems.HostGuestExplicit()
        thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                        temperature=temperature)
        sampler_state = states.SamplerState(test_system.positions)
        topography = Topography(test_system.topology, ligand_atoms='resname B2')
        cls.host_guest_explicit = ('Host-guest explicit', thermodynamic_state, sampler_state, topography)

        # Peptide solvated in explicit solvent.
        test_system = testsystems.AlanineDipeptideExplicit()
        thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                        temperature=temperature)
        sampler_state = states.SamplerState(test_system.positions)
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

    @staticmethod
    def check_expanded_states(alchemical_phase, protocol, expected_cutoff):
        """The expanded states have been setup correctly."""
        cutoff_unit = expected_cutoff.unit
        thermodynamic_state = alchemical_phase._sampler._thermodynamic_states[0]
        unsampled_states = alchemical_phase._sampler._unsampled_states

        is_periodic = thermodynamic_state.is_periodic

        # Anisotropic correction is not applied to non-periodic systems.
        if not is_periodic:
            assert len(unsampled_states) == 0
            return
        assert len(unsampled_states) == 2

        # Find all nonbonded forces and verify their cutoff.
        for state in unsampled_states:
            for force in state.system.getForces():
                try:
                    cutoff = force.getCutoffDistance()
                except AttributeError:
                    continue
                else:
                    err_msg = 'obtained {}, expected {}'.format(cutoff, expected_cutoff)
                    assert np.isclose(cutoff / cutoff_unit, expected_cutoff / cutoff_unit), err_msg

        # The noninteracting state, must be in the same state as the last one.
        noninteracting_state = unsampled_states[1]
        for protocol_key, protocol_values in protocol.items():
            assert getattr(noninteracting_state, protocol_key) == protocol_values[-1]

    def test_create(self):
        """Alchemical state correctly creates the simulation object."""
        available_restraints = list(yank.restraints.available_restraint_classes().values())
        correction_cutoff = 12 * unit.angstroms

        for test_case in self.all_test_cases:
            test_name, thermodynamic_state, sampler_state, topography = test_case

            # Add random restraint if this is ligand-receptor system in implicit solvent.
            if len(topography.ligand_atoms) > 0 and not thermodynamic_state.is_periodic:
                restraint_cls = available_restraints[np.random.randint(0, len(available_restraints))]
                restraint = restraint_cls()
                protocol = self.restrained_protocol
            else:
                restraint = None
                protocol = self.protocol

            alchemical_phase = AlchemicalPhase(sampler=ReplicaExchange())
            with self.temporary_storage_path() as storage_path:
                alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                        protocol, storage_path, restraint=restraint,
                                        anisotropic_dispersion_cutoff=correction_cutoff)

                yield prepare_yield(self.check_protocol, test_name, alchemical_phase, protocol)
                yield prepare_yield(self.check_standard_state_correction, test_name,
                                    alchemical_phase, topography)
                yield prepare_yield(self.check_expanded_states, test_name, alchemical_phase,
                                    protocol, correction_cutoff)

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

    def test_illegal_restraint(self):
        """Raise an error when restraint is handled incorrectly."""
        # An error is raised with ligand-receptor systems in implicit without restraint.
        test_name, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        alchemical_phase = AlchemicalPhase(sampler=ReplicaExchange())
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
            alchemical_phase = AlchemicalPhase(ReplicaExchange())
            alchemical_phase.create(thermodynamic_state, sampler_state, topography,
                                    self.protocol, storage_path, restraint=restraint)

            # Delete old alchemical phase to close storage file.
            del alchemical_phase

            # Resume, the sampler has the correct class.
            alchemical_phase = AlchemicalPhase.from_storage(storage_path)
            assert isinstance(alchemical_phase._sampler, ReplicaExchange)

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
        sampler_state1 = states.SamplerState(test_system.positions)
        sampler_state2 = states.SamplerState(positions2)
        sampler_state3 = states.SamplerState(positions3)
        sampler_states = [sampler_state1, sampler_state2, sampler_state3]

        with self.temporary_storage_path() as storage_path:
            # Create alchemical phase.
            alchemical_phase = AlchemicalPhase(ReplicaExchange())
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
            print(original_diffs, new_diffs)
            assert np.allclose(original_diffs, new_diffs)

    def test_randomize_ligand(self):
        """Test method AlchemicalPhase.randomize_ligand."""
        _, thermodynamic_state, sampler_state, topography = self.host_guest_implicit
        restraint = yank.restraints.Harmonic()

        ligand_atoms, receptor_atoms = topography.ligand_atoms, topography.receptor_atoms
        ligand_positions = sampler_state.positions[ligand_atoms]
        receptor_positions = sampler_state.positions[receptor_atoms]

        with self.temporary_storage_path() as storage_path:
            alchemical_phase = AlchemicalPhase(ReplicaExchange())
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

def test_parameters():
    """Test Yank parameters initialization."""

    # Check that both Yank and Repex parameters are accepted
    Yank(store_directory='test', randomize_ligand=True, nsteps_per_iteration=1)

@nose.tools.raises(TypeError)
def test_unknown_parameters():
    """Test whether Yank raises exception on wrong initialization."""
    Yank(store_directory='test', wrong_parameter=False)


@nose.tools.raises(ValueError)
def test_no_alchemical_atoms():
    """Test whether Yank raises exception when no alchemical atoms are specified."""
    toluene = testsystems.TolueneImplicit()

    # Create parameters. With the exception of atom_indices, all other
    # parameters must be legal, we don't want to catch an exception
    # different than the one we are testing.
    phase = AlchemicalPhase(name='solvent-implicit', reference_system=toluene.system,
                            reference_topology=toluene.topology,
                            positions=toluene.positions, atom_indices={'ligand': []},
                            protocol=AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit())
    thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)

    # Create new simulation.
    with enter_temp_directory():
        yank = Yank(store_directory='output')
        yank.create(thermodynamic_state, phase)


def test_phase_creation():
    """Phases are initialized correctly by Yank.create()."""
    phase_name = 'my-solvent-phase'
    toluene = testsystems.TolueneImplicit()
    protocol = AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()
    atom_indices = find_components(toluene.system, toluene.topology, 'resname TOL')

    phase = AlchemicalPhase(phase_name, toluene.system, toluene.topology,
                            toluene.positions, atom_indices, protocol)
    thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)

    # Create new simulation.
    with enter_temp_directory():
        output_dir = 'output'
        utils.config_root_logger(verbose=False)
        yank = Yank(store_directory=output_dir)
        yank.create(thermodynamic_state, phase)

        # Netcdf dataset has been created
        nc_path = os.path.join(output_dir, phase_name + '.nc')
        assert os.path.isfile(nc_path)

        # Read data
        try:
            nc_file = netcdf.Dataset(nc_path, mode='r')
            metadata_group = nc_file.groups['metadata']
            serialized_system = metadata_group.variables['reference_system'][0]
            serialized_topology = metadata_group.variables['topology'][0]
        finally:
            nc_file.close()

        # Yank doesn't add a barostat to implicit systems
        serialized_system = str(serialized_system)  # convert unicode
        deserialized_system = openmm.XmlSerializer.deserialize(serialized_system)
        for force in deserialized_system.getForces():
            assert 'Barostat' not in force.__class__.__name__

        # Topology has been stored correctly
        deserialized_topology = utils.deserialize_topology(serialized_topology)
        assert deserialized_topology == mdtraj.Topology.from_openmm(toluene.topology)


def test_expanded_cutoff_creation():
    """Test Anisotropic Dispersion Correction expanded cutoff states are created"""
    phase_name = 'my-explicit-phase'
    alanine = testsystems.AlanineDipeptideExplicit()
    protocol = AbsoluteAlchemicalFactory.defaultSolventProtocolExplicit()
    atom_indices = find_components(alanine.system, alanine.topology, 'not water')
    phase = AlchemicalPhase(phase_name, alanine.system, alanine.topology,
                            alanine.positions, atom_indices, protocol)
    thermodynamic_state = ThermodynamicState(temperature=300.0 * unit.kelvin)
    # Calculate the max cutoff
    box_vectors = alanine.system.getDefaultPeriodicBoxVectors()
    min_box_dimension = min([max(vector) for vector in box_vectors])
    # Shrink cutoff to just below maximum allowed
    max_expanded_cutoff = (min_box_dimension / 2.0) * 0.99

    # Create new simulation.
    with enter_temp_directory():
        output_dir = 'output'
        utils.config_root_logger(verbose=False)
        yank = Yank(store_directory=output_dir,
                    anisotropic_dispersion_correction=True,
                    anisotropic_dispersion_cutoff=max_expanded_cutoff)
        yank.create(thermodynamic_state, phase)

        # Test that the expanded cutoff systems were created
        nc_path = os.path.join(output_dir, phase_name + '.nc')

        # Read data
        nc_file = netcdf.Dataset(nc_path, mode='r')
        expanded_cutoff_group = nc_file.groups['expanded_cutoff_states']
        fully_interacting_serial_state = expanded_cutoff_group.variables['fully_interacting_expanded_system'][0]
        noninteracting_serial_state = expanded_cutoff_group.variables['noninteracting_expanded_system'][0]
        nc_file.close()

        for serial_system in [fully_interacting_serial_state, noninteracting_serial_state]:
            system = openmm.XmlSerializer.deserialize(str(serial_system))
            forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
                      range(system.getNumForces())}
            assert forces['NonbondedForce'].getCutoffDistance() == max_expanded_cutoff


def notest_LennardJonesPair(box_width_nsigma=6.0):
    """
    Compute binding free energy of two Lennard-Jones particles and compare to numerical result.

    Parameters
    ----------
    box_width_nsigma : float, optional, default=6.0
        Box width is set to this multiple of Lennard-Jones sigma.

    """

    NSIGMA_MAX = 6.0 # number of standard errors tolerated for success

    # Create Lennard-Jones pair.
    thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    kT = kB * thermodynamic_state.temperature
    sigma = 3.5 * unit.angstroms
    epsilon = 6.0 * kT
    test = testsystems.LennardJonesPair(sigma=sigma, epsilon=epsilon)
    system, positions = test.system, test.positions
    binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    # Create temporary directory for testing.
    import tempfile
    store_dir = tempfile.mkdtemp()

    # Initialize YANK object.
    options = dict()
    options['number_of_iterations'] = 10
    options['platform'] = openmm.Platform.getPlatformByName("Reference") # use Reference platform for speed
    options['mc_rotation'] = False
    options['mc_displacement'] = True
    options['mc_displacement_sigma'] = 1.0 * unit.nanometer
    options['timestep'] = 2 * unit.femtoseconds
    options['nsteps_per_iteration'] = 50

    # Override receptor mass to keep it stationary.
    #system.setParticleMass(0, 0)

    # Override box vectors.
    box_edge = 6*sigma
    a = unit.Quantity((box_edge, 0 * unit.angstrom, 0 * unit.angstrom))
    b = unit.Quantity((0 * unit.angstrom, box_edge, 0 * unit.angstrom))
    c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Override positions
    positions[0,:] = box_edge/2
    positions[1,:] = box_edge/4

    phase = 'complex-explicit'

    # Alchemical protocol.
    from yank.alchemy import AlchemicalState
    alchemical_states = list()
    lambda_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    for lambda_value in lambda_values:
        alchemical_state = AlchemicalState()
        alchemical_state['lambda_electrostatics'] = lambda_value
        alchemical_state['lambda_sterics'] = lambda_value
        alchemical_states.append(alchemical_state)
    protocols = dict()
    protocols[phase] = alchemical_states

    # Create phases.
    alchemical_phase = AlchemicalPhase(phase, system, test.topology, positions,
                                       {'complex-explicit': {'ligand': [1]}},
                                       alchemical_states)

    # Create new simulation.
    yank = Yank(store_dir, **options)
    yank.create(thermodynamic_state, alchemical_phase)

    # Run the simulation.
    yank.run()

    # Analyze the data.
    results = yank.analyze()
    standard_state_correction = results[phase]['standard_state_correction']
    Delta_f = results[phase]['Delta_f_ij'][0,1] - standard_state_correction
    dDelta_f = results[phase]['dDelta_f_ij'][0,1]
    nsigma = abs(binding_free_energy/kT - Delta_f) / dDelta_f

    # Check results against analytical results.
    # TODO: Incorporate standard state correction
    output = "\n"
    output += "Analytical binding free energy                                  : %10.5f +- %10.5f kT\n" % (binding_free_energy / kT, 0)
    output += "Computed binding free energy (with standard state correction)   : %10.5f +- %10.5f kT (nsigma = %3.1f)\n" % (Delta_f, dDelta_f, nsigma)
    output += "Computed binding free energy (without standard state correction): %10.5f +- %10.5f kT (nsigma = %3.1f)\n" % (Delta_f + standard_state_correction, dDelta_f, nsigma)
    output += "Standard state correction alone                                 : %10.5f           kT\n" % (standard_state_correction)
    print(output)

    #if (nsigma > NSIGMA_MAX):
    #    output += "\n"
    #    output += "Computed binding free energy differs from true binding free energy.\n"
    #    raise Exception(output)

    return [Delta_f, dDelta_f]

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
