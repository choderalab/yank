#!/usr/local/bin/env python

"""
Test repex.py facility.

TODO

* Create a few simulation objects on simple systems (e.g. harmonic oscillators?) and run multiple tests on each object?

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import sys
import contextlib

import numpy
import scipy.integrate

from nose import tools

import openmoltools as moltools
from openmmtools import testsystems

from yank.repex import *

# =============================================================================================
# MODULE CONSTANTS
# =============================================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA # Boltzmann constant

# =============================================================================================
# SUBROUTINES
# =============================================================================================


def computeHarmonicOscillatorExpectations(K, mass, temperature):
    """
    Compute mean and variance of potential and kinetic energies for a 3D harmonic oscillator.

    NOTES

    Numerical quadrature is used to compute the mean and standard deviation of the potential energy.
    Mean and standard deviation of the kinetic energy, as well as the absolute free energy, is computed analytically.

    ARGUMENTS

    K (simtk.unit.Quantity) - spring constant
    mass (simtk.unit.Quantity) - mass of particle
    temperature (simtk.unit.Quantity) - temperature

    RETURNS

    values (dict)

    TODO

    Replace this with built-in analytical expectations for new repex.testsystems classes.

    """

    values = dict()

    # Compute thermal energy and inverse temperature from specified temperature.
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature  # thermal energy
    beta = 1.0 / kT  # inverse temperature

    # Compute standard deviation along one dimension.
    sigma = 1.0 / unit.sqrt(beta * K)

    # Define limits of integration along r.
    r_min = 0.0 * unit.nanometers  # initial value for integration
    r_max = 10.0 * sigma      # maximum radius to integrate to

    # Compute mean and std dev of potential energy.
    V = lambda r : (K/2.0) * (r*unit.nanometers)**2 / unit.kilojoules_per_mole # potential in kJ/mol, where r in nm
    q = lambda r : 4.0 * math.pi * r**2 * math.exp(-beta * (K/2.0) * (r*unit.nanometers)**2) # q(r), where r in nm
    (IqV2, dIqV2) = scipy.integrate.quad(lambda r : q(r) * V(r)**2, r_min / unit.nanometers, r_max / unit.nanometers)
    (IqV, dIqV)   = scipy.integrate.quad(lambda r : q(r) * V(r), r_min / unit.nanometers, r_max / unit.nanometers)
    (Iq, dIq)     = scipy.integrate.quad(lambda r : q(r), r_min / unit.nanometers, r_max / unit.nanometers)
    values['potential'] = dict()
    values['potential']['mean'] = (IqV / Iq) * unit.kilojoules_per_mole
    values['potential']['stddev'] = (IqV2 / Iq) * unit.kilojoules_per_mole

    # Compute mean and std dev of kinetic energy.
    values['kinetic'] = dict()
    values['kinetic']['mean'] = (3./2.) * kT
    values['kinetic']['stddev'] = math.sqrt(3./2.) * kT

    # Compute dimensionless free energy.
    # f = - \ln \int_{-\infty}^{+\infty} \exp[-\beta K x^2 / 2]
    #   = - \ln \int_{-\infty}^{+\infty} \exp[-x^2 / 2 \sigma^2]
    #   = - \ln [\sqrt{2 \pi} \sigma]
    values['f'] = - numpy.log(2 * numpy.pi * (sigma / unit.angstroms)**2) * (3.0/2.0)

    return values


def test_replica_exchange(mpicomm=None, verbose=True, verbose_simulation=False):
    """
    Test that free energies and average potential energies of a 3D harmonic oscillator are correctly computed by parallel tempering.

    TODO

    * Test ParallelTempering and HamiltonianExchange subclasses as well.
    * Test with different combinations of input parameters.

    """

    if verbose and ((not mpicomm) or (mpicomm.rank==0)): sys.stdout.write("Testing replica exchange facility with harmonic oscillators: ")

    # Define mass of carbon atom.
    mass = 12.0 * unit.amu

    # Define thermodynamic states.
    states = list() # thermodynamic states
    Ks = [500.00, 400.0, 300.0] * unit.kilocalories_per_mole / unit.angstroms**2 # spring constants
    temperatures = [300.0, 350.0, 400.0] * unit.kelvin # temperatures
    seed_positions = list()
    analytical_results = list()
    f_i_analytical = list() # dimensionless free energies
    u_i_analytical = list() # reduced potential
    for (K, temperature) in zip(Ks, temperatures):
        # Create harmonic oscillator system.
        testsystem = testsystems.HarmonicOscillator(K=K, mass=mass, mm=openmm)
        [system, positions] = [testsystem.system, testsystem.positions]
        # Create thermodynamic state.
        state = ThermodynamicState(system=system, temperature=temperature)
        # Append thermodynamic state and positions.
        states.append(state)
        seed_positions.append(positions)
        # Store analytical results.
        results = computeHarmonicOscillatorExpectations(K, mass, temperature)
        analytical_results.append(results)
        f_i_analytical.append(results['f'])
        kT = kB * temperature # thermal energy
        reduced_potential = results['potential']['mean'] / kT
        u_i_analytical.append(reduced_potential)

    # Compute analytical Delta_f_ij
    nstates = len(f_i_analytical)
    f_i_analytical = numpy.array(f_i_analytical)
    u_i_analytical = numpy.array(u_i_analytical)
    s_i_analytical = u_i_analytical - f_i_analytical
    Delta_f_ij_analytical = numpy.zeros([nstates, nstates], numpy.float64)
    Delta_u_ij_analytical = numpy.zeros([nstates, nstates], numpy.float64)
    Delta_s_ij_analytical = numpy.zeros([nstates, nstates], numpy.float64)
    for i in range(nstates):
        for j in range(nstates):
            Delta_f_ij_analytical[i, j] = f_i_analytical[j] - f_i_analytical[i]
            Delta_u_ij_analytical[i, j] = u_i_analytical[j] - u_i_analytical[i]
            Delta_s_ij_analytical[i, j] = s_i_analytical[j] - s_i_analytical[i]

    # Define file for temporary storage.
    import tempfile # use a temporary file
    file = tempfile.NamedTemporaryFile(delete=False)
    store_filename = file.name
    # print("node %d : Storing data in temporary file: %s" % (mpicomm.rank, str(store_filename))) # DEBUG

    # Create and configure simulation object.
    simulation = ReplicaExchange(store_filename, mpicomm=mpicomm)
    simulation.create(states, seed_positions)
    simulation.platform = openmm.Platform.getPlatformByName('Reference')
    simulation.minimize = False
    simulation.number_of_iterations = 200
    simulation.nsteps_per_iteration = 500
    simulation.timestep = 2.0 * unit.femtoseconds
    simulation.collision_rate = 20.0 / unit.picosecond
    simulation.verbose = verbose_simulation
    simulation.show_mixing_statistics = False
    simulation.online_analysis = True

    # Run simulation.
    utils.config_root_logger(False)
    simulation.run()  # run the simulation

    # Run an extension simulation
    simulation.extend_simulation = True
    simulation.number_of_iterations = 1
    simulation.run()
    utils.config_root_logger(True)

    # Stop here if not root node.
    if mpicomm and (mpicomm.rank != 0): return

    # Retrieve extant analysis object.
    online_analysis = simulation.analysis

    # Analyze simulation to compute free energies.
    analysis = simulation.analyze()

    # Check if online analysis is close to final analysis.
    error = numpy.abs(online_analysis['Delta_f_ij'] - analysis['Delta_f_ij'])
    derror = (online_analysis['dDelta_f_ij']**2 + analysis['dDelta_f_ij']**2)
    indices = numpy.where(derror > 0.0)
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / derror[indices]
    MAX_SIGMA = 6.0 # maximum allowed number of standard errors
    if numpy.any(nsigma > MAX_SIGMA):
        print("Delta_f_ij from online analysis")
        print(online_analysis['Delta_f_ij'])
        print("Delta_f_ij from final analysis")
        print(analysis['Delta_f_ij'])
        print("error")
        print(error)
        print("derror")
        print(derror)
        print("nsigma")
        print(nsigma)
        raise Exception("Dimensionless free energy differences between online and final analysis exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    # TODO: Check if deviations exceed tolerance.
    Delta_f_ij = analysis['Delta_f_ij']
    dDelta_f_ij = analysis['dDelta_f_ij']
    error = numpy.abs(Delta_f_ij - Delta_f_ij_analytical)
    indices = numpy.where(dDelta_f_ij > 0.0)
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / dDelta_f_ij[indices]
    MAX_SIGMA = 6.0 # maximum allowed number of standard errors
    if numpy.any(nsigma > MAX_SIGMA):
        print("Delta_f_ij")
        print(Delta_f_ij)
        print("Delta_f_ij_analytical")
        print(Delta_f_ij_analytical)
        print("error")
        print(error)
        print("stderr")
        print(dDelta_f_ij)
        print("nsigma")
        print(nsigma)
        raise Exception("Dimensionless free energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    error = analysis['Delta_u_ij'] - Delta_u_ij_analytical
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / dDelta_f_ij[indices]
    if numpy.any(nsigma > MAX_SIGMA):
        print("Delta_u_ij")
        print(analysis['Delta_u_ij'])
        print("Delta_u_ij_analytical")
        print(Delta_u_ij_analytical)
        print("error")
        print(error)
        print("nsigma")
        print(nsigma)
        raise Exception("Dimensionless potential energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    # Clean up.
    del simulation

    if verbose:
        print("PASSED.")
    return


def notest_hamiltonian_exchange(mpicomm=None, verbose=True):
    """
    Test that free energies and average potential energies of a 3D harmonic oscillator are correctly computed when running HamiltonianExchange.

    TODO

    * Integrate with test_replica_exchange.
    * Test with different combinations of input parameters.

    """

    if verbose and ((not mpicomm) or (mpicomm.rank==0)): sys.stdout.write("Testing Hamiltonian exchange facility with harmonic oscillators: ")

    # Create test system of harmonic oscillators
    testsystem = testsystems.HarmonicOscillatorArray()
    [system, coordinates] = [testsystem.system, testsystem.positions]

    # Define mass of carbon atom.
    mass = 12.0 * unit.amu

    # Define thermodynamic states.
    sigmas = [0.2, 0.3, 0.4] * unit.angstroms # standard deviations: beta K = 1/sigma^2 so K = 1/(beta sigma^2)
    temperature = 300.0 * unit.kelvin # temperatures
    seed_positions = list()
    analytical_results = list()
    f_i_analytical = list() # dimensionless free energies
    u_i_analytical = list() # reduced potential
    systems = list() # Systems list for HamiltonianExchange
    for sigma in sigmas:
        # Compute corresponding spring constant.
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature # thermal energy
        beta = 1.0 / kT # inverse temperature
        K = 1.0 / (beta * sigma**2)
        # Create harmonic oscillator system.
        testsystem = testsystems.HarmonicOscillator(K=K, mass=mass, mm=openmm)
        [system, positions] = [testsystem.system, testsystem.positions]
        # Append to systems list.
        systems.append(system)
        # Append positions.
        seed_positions.append(positions)
        # Store analytical results.
        results = computeHarmonicOscillatorExpectations(K, mass, temperature)
        analytical_results.append(results)
        f_i_analytical.append(results['f'])
        reduced_potential = results['potential']['mean'] / kT
        u_i_analytical.append(reduced_potential)

    # DEBUG
    print("")
    print(seed_positions)
    print(analytical_results)
    print(u_i_analytical)
    print(f_i_analytical)
    print("")

    # Compute analytical Delta_f_ij
    nstates = len(f_i_analytical)
    f_i_analytical = numpy.array(f_i_analytical)
    u_i_analytical = numpy.array(u_i_analytical)
    s_i_analytical = u_i_analytical - f_i_analytical
    Delta_f_ij_analytical = numpy.zeros([nstates,nstates], numpy.float64)
    Delta_u_ij_analytical = numpy.zeros([nstates,nstates], numpy.float64)
    Delta_s_ij_analytical = numpy.zeros([nstates,nstates], numpy.float64)
    for i in range(nstates):
        for j in range(nstates):
            Delta_f_ij_analytical[i,j] = f_i_analytical[j] - f_i_analytical[i]
            Delta_u_ij_analytical[i,j] = u_i_analytical[j] - u_i_analytical[i]
            Delta_s_ij_analytical[i,j] = s_i_analytical[j] - s_i_analytical[i]

    # Define file for temporary storage.
    import tempfile # use a temporary file
    file = tempfile.NamedTemporaryFile(delete=False)
    store_filename = file.name
    #print("Storing data in temporary file: %s" % str(store_filename))

    # Create reference thermodynamic state.
    reference_state = ThermodynamicState(systems[0], temperature=temperature)

    # Create and configure simulation object.
    simulation = HamiltonianExchange(store_filename, mpicomm=mpicomm)
    simulation.create(reference_state, systems, seed_positions)
    simulation.platform = openmm.Platform.getPlatformByName('Reference')
    simulation.number_of_iterations = 200
    simulation.timestep = 2.0 * unit.femtoseconds
    simulation.nsteps_per_iteration = 500
    simulation.collision_rate = 9.2 / unit.picosecond
    simulation.verbose = False
    simulation.show_mixing_statistics = False

    # Run simulation.
    utils.config_root_logger(True)
    simulation.run() # run the simulation
    utils.config_root_logger(False)

    # Stop here if not root node.
    if mpicomm and (mpicomm.rank != 0): return

    # Analyze simulation to compute free energies.
    analysis = simulation.analyze()

    # TODO: Check if deviations exceed tolerance.
    Delta_f_ij = analysis['Delta_f_ij']
    dDelta_f_ij = analysis['dDelta_f_ij']
    error = Delta_f_ij - Delta_f_ij_analytical
    indices = numpy.where(dDelta_f_ij > 0.0)
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / dDelta_f_ij[indices]
    MAX_SIGMA = 6.0 # maximum allowed number of standard errors
    if numpy.any(nsigma > MAX_SIGMA):
        print("Delta_f_ij")
        print(Delta_f_ij)
        print("Delta_f_ij_analytical")
        print(Delta_f_ij_analytical)
        print("error")
        print(error)
        print("stderr")
        print(dDelta_f_ij)
        print("nsigma")
        print(nsigma)
        raise Exception("Dimensionless free energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    error = analysis['Delta_u_ij'] - Delta_u_ij_analytical
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / dDelta_f_ij[indices]
    if numpy.any(nsigma > MAX_SIGMA):
        print("Delta_u_ij")
        print(analysis['Delta_u_ij'])
        print("Delta_u_ij_analytical")
        print(Delta_u_ij_analytical)
        print("error")
        print(error)
        print("nsigma")
        print(nsigma)
        raise Exception("Dimensionless potential energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    if verbose: print("PASSED.")
    return


# ==============================================================================
# TEST REPORTER
# ==============================================================================

@contextlib.contextmanager
def temporary_reporter(file_name='test_storage.nc'):
    """Create and initialize a reporter in a temporary directory."""
    with moltools.utils.temporary_directory() as tmp_dir_path:
        storage_file_path = os.path.join(tmp_dir_path, file_name)
        reporter = Reporter(storage=storage_file_path)
        reporter.initialize_storage()
        yield reporter


def test_storage_initialization():
    """Check that storage file is initialized correctly."""
    with moltools.utils.temporary_directory() as tmp_dir_path:
        storage_file_path = os.path.join(tmp_dir_path, 'test_storage.nc')
        reporter = Reporter(storage=storage_file_path)
        storage_file_path = reporter._storage_file_path
        assert not os.path.isfile(storage_file_path)
        reporter.initialize_storage()
        assert os.path.isfile(storage_file_path)


def test_store_thermodynamic_states():
    """Check correct storage of thermodynamic states."""
    with temporary_reporter() as reporter:
        # Create thermodynamic states.
        temperature = 300*unit.kelvin
        alanine_system = testsystems.AlanineDipeptideExplicit().system
        thermodynamic_state_nvt = mmtools.states.ThermodynamicState(alanine_system, temperature)
        thermodynamic_state_npt = mmtools.states.ThermodynamicState(alanine_system, temperature,
                                                                    1.0*unit.atmosphere)
        thermodynamic_states = [thermodynamic_state_nvt, thermodynamic_state_npt]

        # Check that after writing and reading, states are identical.
        reporter.write_thermodynamic_states(thermodynamic_states)
        restored_thermodynamic_states = reporter.read_thermodynamic_states()
        for state, restored_state in zip(thermodynamic_states, restored_thermodynamic_states):
            assert state.system.__getstate__() == restored_state.system.__getstate__()
            assert state.temperature == restored_state.temperature
            assert state.pressure == restored_state.pressure


def test_store_sampler_states():
    """Check correct storage of thermodynamic states."""
    with temporary_reporter() as reporter:
        # Create sampler states.
        alanine_test = testsystems.AlanineDipeptideVacuum()
        positions = alanine_test.positions
        box_vectors = alanine_test.system.getDefaultPeriodicBoxVectors()
        sampler_states = [mmtools.states.SamplerState(positions=positions, box_vectors=box_vectors)
                          for _ in range(2)]

        # Check that after writing and reading, states are identical.
        reporter.write_sampler_states(sampler_states, iteration=0)
        restored_sampler_states = reporter.read_sampler_states(iteration=0)
        for state, restored_state in zip(sampler_states, restored_sampler_states):
            assert np.allclose(state.positions, restored_state.positions)
            assert np.allclose(state.box_vectors / unit.nanometer, restored_state.box_vectors / unit.nanometer)


def test_store_replica_thermodynamic_states():
    """Check storage of replica thermodynamic states indices."""
    with temporary_reporter() as reporter:
        for i, replica_states in enumerate([[2, 1, 0, 3], np.array([3, 1, 0, 2])]):
            reporter.write_replica_thermodynamic_states(replica_states, iteration=i)
            restored_replica_states = reporter.read_replica_thermodynamic_states(iteration=i)
            assert np.all(replica_states == restored_replica_states)


def test_store_energies():
    """Check storage of energies."""
    with temporary_reporter() as reporter:
        energy_matrix = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        reporter.write_energies(energy_matrix, iteration=0)
        restored_energy_matrix = reporter.read_energies(iteration=0)
        assert np.all(energy_matrix == restored_energy_matrix)


def test_store_dict():
    """Check correct storage and restore of dictionaries."""
    data = {
        'mystring': 'test',
        'myinteger': 3, 'myfloat': 4.0,
        'mylist': [0, 1, 2, 3], 'mynumpyarray': np.array([2.0, 3, 4]),
        'mynestednumpyarray': np.array([[1, 2, 3], [4.0, 5, 6]]),
        'myquantity': 5.0 / unit.femtosecond,
        'myquantityarray': unit.Quantity(np.array([[1, 2, 3], [4.0, 5, 6]]), unit=unit.angstrom)
    }
    with temporary_reporter() as reporter:
        reporter.write_dict('metadata', data)
        restored_data = reporter.read_dict('metadata')
        for key, value in data.items():
            restored_value = restored_data[key]
            try:
                assert value == restored_value, '{}, {}'.format(value, restored_value)
            except ValueError:  # array
                assert np.all(value == restored_value)


def test_parameters():
    """Test ReplicaExchange parameters initialization."""
    repex = ReplicaExchange(store_filename='test', nsteps_per_iteration=1e6)
    assert repex.nsteps_per_iteration == 1000000
    assert repex.collision_rate == repex.default_parameters['collision_rate']


@tools.raises(TypeError)
def test_unknown_parameters():
    """Test ReplicaExchange raises exception on wrong initialization."""
    ReplicaExchange(store_filename='test', wrong_parameter=False)


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    # Configure logger.
    utils.config_root_logger(False)

    # Try MPI, if possible.
    try:
        mpicomm = utils.initialize_mpi()
        if mpicomm.rank == 0:
            print("MPI initialized successfully.")
    except Exception as e:
        print(e)
        print("Could not start MPI. Using serial code instead.")
        mpicomm = None

    # Test simple system of harmonic oscillators.
    # Disabled until we fix the test
    # test_hamiltonian_exchange(mpicomm)
    test_replica_exchange(mpicomm)
