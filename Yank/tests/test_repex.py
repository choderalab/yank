#!/usr/local/bin/env python

"""
Test repex.py facility.

TODO

* Create a few simulation objects on simple systems (e.g. harmonic oscillators?) and run multiple tests on each object?

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import sys
import pickle
import contextlib

import numpy
import scipy.integrate

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

def write_thermodynamic_states_0_1(reporter, thermodynamic_states):
    """Simulate ThermodynamicStates stored with Conventions 0.1.

    This is for testing backwards compatibility.

    """
    setattr(reporter._storage, 'ConventionVersion', '0.1')

    n_states = len(thermodynamic_states)
    is_barostated = thermodynamic_states[0].pressure is not None

    ncgrp_states = reporter._storage.createGroup('thermodynamic_states')
    reporter._storage.createDimension('replica', n_states)
    ncvar_serialized_systems = ncgrp_states.createVariable('systems', str, ('replica',), zlib=True)
    ncvar_temperatures = ncgrp_states.createVariable('temperatures', 'f', ('replica',))
    if is_barostated:
        ncvar_pressures = ncgrp_states.createVariable('pressures', 'f', ('replica',))

    for state_id, thermodynamic_state in enumerate(thermodynamic_states):
        serialized_system = thermodynamic_state.system.__getstate__()
        ncvar_serialized_systems[state_id] = serialized_system
        ncvar_temperatures[state_id] = thermodynamic_state.temperature / unit.kelvin
        if is_barostated:
            ncvar_pressures[state_id] = thermodynamic_state.pressure / unit.atmospheres


class TestReporter(object):
    """Test suite for Reporter class."""

    @staticmethod
    @contextlib.contextmanager
    def temporary_reporter():
        """Create and initialize a reporter in a temporary directory."""
        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_file_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            assert not os.path.isfile(storage_file_path)
            reporter = Reporter(storage=storage_file_path, open_mode='w')
            assert os.path.isfile(storage_file_path)
            yield reporter

    def test_store_thermodynamic_states(self):
        """Check correct storage of thermodynamic states."""
        # Create thermodynamic states.
        temperature = 300*unit.kelvin
        alanine_system = testsystems.AlanineDipeptideImplicit().system
        alanine_explicit_system = testsystems.AlanineDipeptideExplicit().system
        thermodynamic_state_nvt = mmtools.states.ThermodynamicState(alanine_system, temperature)
        thermodynamic_state_nvt_compatible = mmtools.states.ThermodynamicState(alanine_system,
                                                                               temperature + 20*unit.kelvin)
        thermodynamic_state_npt = mmtools.states.ThermodynamicState(alanine_explicit_system,
                                                                    temperature, 1.0*unit.atmosphere)

        # Create compound states.
        factory = mmtools.alchemy.AlchemicalFactory()
        alchemical_region = mmtools.alchemy.AlchemicalRegion(alchemical_atoms=range(22))
        alanine_alchemical = factory.create_alchemical_system(alanine_system, alchemical_region)
        alchemical_state_interacting = mmtools.alchemy.AlchemicalState.from_system(alanine_alchemical)
        alchemical_state_noninteracting = copy.deepcopy(alchemical_state_interacting)
        alchemical_state_noninteracting.set_alchemical_parameters(0.0)
        compound_state_interacting = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=mmtools.states.ThermodynamicState(alanine_alchemical, temperature),
            composable_states=[alchemical_state_interacting]
        )
        compound_state_noninteracting = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=mmtools.states.ThermodynamicState(alanine_alchemical, temperature),
            composable_states=[alchemical_state_noninteracting]
        )

        # Check all conventions.
        thermodynamic_states = [thermodynamic_state_nvt, thermodynamic_state_nvt_compatible,
                                thermodynamic_state_npt, compound_state_interacting,
                                compound_state_noninteracting]
        writers = [
            Reporter.write_thermodynamic_states,  # latest
            write_thermodynamic_states_0_1
        ]
        for writer in writers:
            with self.temporary_reporter() as reporter:
                thermodynamic_states = copy.deepcopy(thermodynamic_states)

                # Check that after writing and reading, states are identical.
                writer(reporter, thermodynamic_states)
                restored_thermodynamic_states = reporter.read_thermodynamic_states()
                for state, restored_state in zip(thermodynamic_states, restored_thermodynamic_states):
                    assert state.system.__getstate__() == restored_state.system.__getstate__()
                    assert state.temperature == restored_state.temperature
                    assert state.pressure == restored_state.pressure

                # The latest writer only stores one full serialization per compatible state.
                if writer != write_thermodynamic_states_0_1:
                    ncgrp_states = reporter._storage.groups['thermodynamic_states']
                    assert isinstance(ncgrp_states.groups['state0'].variables['standard_system'][0], str)
                    assert 'standard_system' not in ncgrp_states.groups['state1'].variables
                    assert ncgrp_states.groups['state1'].variables['_Reporter__standard_system_id'].getValue() == 0
                    assert 'standard_system' in ncgrp_states.groups['state2'].variables
                    assert 'standard_system' in ncgrp_states.groups['state3'].groups['thermodynamic_state'].variables
                    ncgrp_state4 = ncgrp_states.groups['state4'].groups['thermodynamic_state']
                    assert ncgrp_state4.variables['_Reporter__standard_system_id'].getValue() == 3

    def test_store_sampler_states(self):
        """Check correct storage of thermodynamic states."""
        with self.temporary_reporter() as reporter:
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

    def test_store_replica_thermodynamic_states(self):
        """Check storage of replica thermodynamic states indices."""
        with self.temporary_reporter() as reporter:
            for i, replica_states in enumerate([[2, 1, 0, 3], np.array([3, 1, 0, 2])]):
                reporter.write_replica_thermodynamic_states(replica_states, iteration=i)
                restored_replica_states = reporter.read_replica_thermodynamic_states(iteration=i)
                assert np.all(replica_states == restored_replica_states)

    def test_store_mcmc_moves(self):
        """Check storage of MCMC moves."""
        sequence_move = mmtools.mcmc.SequenceMove(move_list=[mmtools.mcmc.LangevinDynamicsMove(),
                                                             mmtools.mcmc.GHMCMove()],
                                                  context_cache=mmtools.cache.ContextCache(capacity=1))
        integrator_move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond),
                                                      n_steps=100)
        mcmc_moves = [sequence_move, integrator_move]
        with self.temporary_reporter() as reporter:
            reporter.write_mcmc_moves(mcmc_moves)
            restored_mcmc_moves = reporter.read_mcmc_moves()

            # Check that restored MCMCMoves are exactly the same.
            original_pickle = pickle.dumps(mcmc_moves)
            restored_pickle = pickle.dumps(restored_mcmc_moves)
            assert original_pickle == restored_pickle

    def test_store_energies(self):
        """Check storage of energies."""
        with self.temporary_reporter() as reporter:
            energy_matrix = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
            reporter.write_energies(energy_matrix, iteration=0)
            restored_energy_matrix = reporter.read_energies(iteration=0)
            assert np.all(energy_matrix == restored_energy_matrix)

    def test_store_dict(self):
        """Check correct storage and restore of dictionaries."""
        data = {
            'mybool': False,
            'mystring': 'test',
            'myinteger': 3, 'myfloat': 4.0,
            'mylist': [0, 1, 2, 3], 'mynumpyarray': np.array([2.0, 3, 4]),
            'mynestednumpyarray': np.array([[1, 2, 3], [4.0, 5, 6]]),
            'myquantity': 5.0 / unit.femtosecond,
            'myquantityarray': unit.Quantity(np.array([[1, 2, 3], [4.0, 5, 6]]), unit=unit.angstrom),
            'mynesteddict': {'field1': 'string', 'field2': {'field21': 3.0, 'field22': True}}
        }
        with self.temporary_reporter() as reporter:
            reporter.write_dict('testdict', data)
            restored_data = reporter.read_dict('testdict')
            for key, value in data.items():
                restored_value = restored_data[key]
                err_msg = '{}, {}'.format(value, restored_value)
                try:
                    assert value == restored_value, err_msg
                except ValueError:  # array-like
                    assert np.all(value == restored_value)
                else:
                    assert type(value) == type(restored_value), err_msg

            # write_dict supports updates.
            data['mybool'] = True
            data['mystring'] = 'substituted'
            reporter.write_dict('testdict', data)
            restored_data = reporter.read_dict('testdict')
            assert restored_data['mybool'] is True
            assert restored_data['mystring'] == 'substituted'

    def test_store_mixing_statistics(self):
        """Check mixing statistics are correctly stored."""
        n_accepted_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        n_proposed_matrix = np.array([[3, 3, 3], [6, 6, 6], [9, 9, 9]])
        with self.temporary_reporter() as reporter:
            reporter.write_mixing_statistics(n_accepted_matrix, n_proposed_matrix, iteration=0)
            restored_n_accepted, restored_n_proposed = reporter.read_mixing_statistics(iteration=0)
            assert np.all(n_accepted_matrix == restored_n_accepted)
            assert np.all(n_proposed_matrix == restored_n_proposed)


# ==============================================================================
# TEST REPLICA EXCHANGE
# ==============================================================================

class TestReplicaExchange(object):
    """Test suite for ReplicaExchange class."""

    @classmethod
    def setup_class(cls):
        """Shared test cases and variables."""
        # Test case with alanine in vacuum at 3 different positions and temperatures.
        n_states = 3
        alanine_test = testsystems.AlanineDipeptideVacuum()
        # Translate the sampler states to be different one from each other.
        alanine_sampler_states = [mmtools.states.SamplerState(alanine_test.positions + 10*i*unit.nanometers)
                                  for i in range(n_states)]
        # Set increasing temperature.
        temperatures = [(300 + 10*i) * unit.kelvin for i in range(n_states)]
        alanine_thermodynamic_states = [mmtools.states.ThermodynamicState(alanine_test.system, temperatures[i])
                                        for i in range(n_states)]
        cls.alanine_test = (alanine_thermodynamic_states, alanine_sampler_states)

    def test_repex_create(self):
        """Test creation of a new ReplicaExchange simulation.

        Checks that the storage file is correctly initialized with all the
        information needed. With MPI, this checks that only node 0 has an
        open Reporter for writing.

        """
        # TODO test with CompoundState
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        # Remove one sampler state to verify distribution over states.
        sampler_states = sampler_states[:-1]

        with moltools.utils.temporary_directory() as tmp_dir_path:
            repex = ReplicaExchange()
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)

            # Check that reporter has reporter only if rank 0.
            mpicomm = mpi.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                assert repex._reporter.is_open()
            else:
                assert not repex._reporter.is_open()

            # Open reporter to read stored data.
            reporter = Reporter(storage_path, open_mode='r')

            # The n_states-1 sampler states have been distributed to n_states replica.
            restored_sampler_states = reporter.read_sampler_states(iteration=0)
            assert len(repex._sampler_states) == n_states
            assert len(restored_sampler_states) == n_states
            assert np.allclose(restored_sampler_states[0].positions, repex._sampler_states[0].positions)

            # MCMCMove was stored correctly.
            restored_mcmc_moves = reporter.read_mcmc_moves()
            assert len(repex._mcmc_moves) == n_states
            assert len(restored_mcmc_moves) == n_states
            for repex_move, restored_move in zip(repex._mcmc_moves, restored_mcmc_moves):
                assert isinstance(repex_move, mmtools.mcmc.LangevinDynamicsMove)
                assert isinstance(restored_move, mmtools.mcmc.LangevinDynamicsMove)

            # Options have been stored.
            option_names, _, _, defaults = inspect.getargspec(repex.__init__)
            option_names = option_names[2:]  # Discard 'self' and 'mcmc_moves' arguments.
            defaults = defaults[1:]  # Discard 'mcmc_moves' default.
            options = reporter.read_dict('options')
            assert len(options) == len(defaults)
            for key, value in zip(option_names, defaults):
                assert options[key] == value
                assert getattr(repex, '_' + key) == value

            # A default title has been added to the stored metadata.
            metadata = reporter.read_dict('metadata')
            assert len(metadata) == 1
            assert 'title' in metadata

    def test_from_storage(self):
        """Test that from_storage completely restore ReplicaExchange.

        Checks that the static constructor ReplicaExchange.from_storage()
        restores the simulation object in the exact same state as the last
        iteration.

        """
        # TODO test after running few iterations
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)

        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')

            # Create first repex and store its state (its __dict__). We leave the
            # reporter out because when the NetCDF file is copied, it runs into issues.
            repex = ReplicaExchange()
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)
            original_dict = copy.deepcopy({k: v for k, v in repex.__dict__.items() if not k == '_reporter'})

            # Create a new repex from the storage file.
            repex = ReplicaExchange.from_storage(storage_path)
            restored_dict = copy.deepcopy({k: v for k, v in repex.__dict__.items() if not k == '_reporter'})

            # Check thermodynamic states.
            original_ts = original_dict.pop('_thermodynamic_states')
            restored_ts = restored_dict.pop('_thermodynamic_states')
            for original, restored in zip(original_ts, restored_ts):
                assert original.system.__getstate__() == restored.system.__getstate__()

            # Check sampler states.
            original_ss = original_dict.pop('_sampler_states')
            restored_ss = restored_dict.pop('_sampler_states')
            for original, restored in zip(original_ss, restored_ss):
                assert np.allclose(original.positions, restored.positions)
                assert np.all(original.box_vectors == restored.box_vectors)

            # The reporter of the restored simulation must be open only in node 0.
            mpicomm = mpi.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                assert repex._reporter.is_open()
            else:
                assert not repex._reporter.is_open()

            # Check all arrays. Instantiate list so that we can pop from original_dict.
            for attr, original_value in list(original_dict.items()):
                if isinstance(original_value, np.ndarray):
                    original_value = original_dict.pop(attr)
                    restored_value = restored_dict.pop(attr)
                    assert np.all(original_value == restored_value)

            # Check everything else as pickle.
            original_pickle = pickle.dumps(original_dict)
            restored_pickle = pickle.dumps(restored_dict)
            assert original_pickle == restored_pickle

    def test_stored_properties(self):
        """Test that storage is kept in sync with options."""
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)

        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            repex = ReplicaExchange()
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)

            # Get original options.
            reporter = Reporter(storage_path, open_mode='r')
            options = reporter.read_dict('options')

            # Update options and check the storage is synchronized.
            repex.number_of_iterations = 123
            repex.replica_mixing_scheme = 'none'
            repex.online_analysis = not repex.online_analysis
            changed_options = reporter.read_dict('options')
            assert changed_options['number_of_iterations'] == 123
            assert changed_options['replica_mixing_scheme'] == 'none'
            assert changed_options['online_analysis'] == (not options['online_analysis'])

    def test_propagate_replicas(self):
        """Test method _propagate_replicas from ReplicaExchange.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the propagated StateSamplers when it communicates
        the new positions and box vectors.

        """
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')

            # For this test to work, positions should be the same but
            # translated, so that minimized positions should satisfy
            # the same condition.
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                              for i in range(n_states - 1)]
            assert not np.allclose(original_diffs, [0 for _ in range(n_states - 1)])

            # Create a replica exchange that propagates only 1 femtosecond
            # per iteration so that positions won't change much.
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=1)
            repex = ReplicaExchange(mcmc_moves=move)
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)

            # Propagate.
            repex._propagate_replicas()

            # The relative positions between the new sampler states should
            # be still translated the same way (i.e. we are not assigning
            # the minimized positions to the incorrect sampler states).
            new_sampler_states = repex._sampler_states
            new_diffs = [np.average(new_sampler_states[i].positions - new_sampler_states[i+1].positions)
                         for i in range(n_states - 1)]
            assert np.allclose(original_diffs, new_diffs)

    def test_minimize(self):
        """Test ReplicaExchange minimize method.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the minimized StateSamplers when it communicates
        the new positions. It also checks that the energies are effectively
        decreased.

        """
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            repex = ReplicaExchange()
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)

            # For this test to work, positions should be the same but
            # translated, so that minimized positions should satisfy
            # the same condition.
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                              for i in range(n_states - 1)]
            assert not np.allclose(original_diffs, [0 for _ in range(n_states - 1)])

            # Compute initial energies.
            repex._compute_energies()
            original_energies = [repex._u_kl[i, i] for i in range(n_states)]

            # Minimize.
            repex.minimize()

            # The relative positions between the new sampler states should
            # be still translated the same way (i.e. we are not assigning
            # the minimized positions to the incorrect sampler states).
            new_sampler_states = repex._sampler_states
            new_diffs = [np.average(new_sampler_states[i].positions - new_sampler_states[i+1].positions)
                         for i in range(n_states - 1)]
            assert np.allclose(original_diffs, new_diffs)

            # The energies have been minimized.
            repex._compute_energies()
            new_energies = [repex._u_kl[i, i] for i in range(n_states)]
            for i in range(n_states):
                assert new_energies[i] < original_energies[i]

            # The storage has been updated.
            reporter = Reporter(storage_path, open_mode='r')
            stored_sampler_states = reporter.read_sampler_states(iteration=0)
            for new_state, stored_state in zip(new_sampler_states, stored_sampler_states):
                assert np.allclose(new_state.positions, stored_state.positions)

    def test_equilibrate(self):
        """Test equilibration of ReplicaExchange simulation.

        During equilibration, we set temporarily different MCMCMoves. This checks
        that they are restored correctly. It also checks that the storage has the
        updated positions.

        """
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)

        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')

            # We create a ReplicaExchange with a GHMC move but use Langevin for equilibration.
            repex = ReplicaExchange(mcmc_moves=mmtools.mcmc.GHMCMove())
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)

            # Minimize.
            equilibration_move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
            repex.equilibrate(n_iterations=10, mcmc_moves=equilibration_move)
            assert isinstance(repex._mcmc_moves[0], mmtools.mcmc.GHMCMove)

            # The storage has been updated.
            reporter = Reporter(storage_path, open_mode='r')
            stored_sampler_states = reporter.read_sampler_states(iteration=0)
            for new_state, stored_state in zip(repex._sampler_states, stored_sampler_states):
                assert np.allclose(new_state.positions, stored_state.positions)

            # We are still at iteration 0.
            assert repex._iteration == 0

    def test_run_extend(self):
        """Test methods run and extend of ReplicaExchange."""
        thermodynamic_states, sampler_states = copy.deepcopy(self.alanine_test)

        with moltools.utils.temporary_directory() as tmp_dir_path:
            storage_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            repex = ReplicaExchange(mcmc_moves=mmtools.mcmc.GHMCMove(n_steps=1),
                                    show_energies=True, show_mixing_statistics=True,
                                    number_of_iterations=2)
            repex.create(thermodynamic_states, sampler_states, storage=storage_path)

            # ReplicaExchange.run doesn't go past number_of_iterations.
            repex.run(n_iterations=3)
            assert repex.iteration == 2

            # ReplicaExchange.extend does.
            repex.extend(n_iterations=2)
            assert repex.iteration == 4

            # The MCMCMoves statistics in the storage are updated.
            reporter = Reporter(storage_path, open_mode='r')
            restored_mcmc_moves = reporter.read_mcmc_moves()
            assert restored_mcmc_moves[0].n_attempted != 0



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
