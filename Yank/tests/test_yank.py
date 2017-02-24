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

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================


from openmmtools import testsystems
import mdtraj
from mdtraj.utils import enter_temp_directory

from nose import tools
import netCDF4 as netcdf

from yank.repex import ThermodynamicState
from yank.pipeline import find_components


from yank.yank import *

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

from simtk import unit
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

def test_parameters():
    """Test Yank parameters initialization."""

    # Check that both Yank and Repex parameters are accepted
    Yank(store_directory='test', randomize_ligand=True, nsteps_per_iteration=1)

@tools.raises(TypeError)
def test_unknown_parameters():
    """Test whether Yank raises exception on wrong initialization."""
    Yank(store_directory='test', wrong_parameter=False)


@tools.raises(ValueError)
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
