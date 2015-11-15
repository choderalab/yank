#!/usr/local/bin/env python

"""
Test repex.py facility.

TODO

* Create a few simulation objects on simple systems (e.g. harmonic oscillators?) and run multiple tests on each object?

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import sys
import math
import copy
import time
import datetime

import numpy
import scipy.integrate

import simtk.openmm as openmm
import simtk.unit as units

from nose import tools

from openmmtools import testsystems

from yank.repex import ThermodynamicState, ReplicaExchange, HamiltonianExchange, ParallelTempering

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# SUBROUTINES
#=============================================================================================

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
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
    kT = kB * temperature # thermal energy
    beta = 1.0 / kT # inverse temperature

    # Compute standard deviation along one dimension.
    sigma = 1.0 / units.sqrt(beta * K)

    # Define limits of integration along r.
    r_min = 0.0 * units.nanometers # initial value for integration
    r_max = 10.0 * sigma      # maximum radius to integrate to

    # Compute mean and std dev of potential energy.
    V = lambda r : (K/2.0) * (r*units.nanometers)**2 / units.kilojoules_per_mole # potential in kJ/mol, where r in nm
    q = lambda r : 4.0 * math.pi * r**2 * math.exp(-beta * (K/2.0) * (r*units.nanometers)**2) # q(r), where r in nm
    (IqV2, dIqV2) = scipy.integrate.quad(lambda r : q(r) * V(r)**2, r_min / units.nanometers, r_max / units.nanometers)
    (IqV, dIqV)   = scipy.integrate.quad(lambda r : q(r) * V(r), r_min / units.nanometers, r_max / units.nanometers)
    (Iq, dIq)     = scipy.integrate.quad(lambda r : q(r), r_min / units.nanometers, r_max / units.nanometers)
    values['potential'] = dict()
    values['potential']['mean'] = (IqV / Iq) * units.kilojoules_per_mole
    values['potential']['stddev'] = (IqV2 / Iq) * units.kilojoules_per_mole

    # Compute mean and std dev of kinetic energy.
    values['kinetic'] = dict()
    values['kinetic']['mean'] = (3./2.) * kT
    values['kinetic']['stddev'] = math.sqrt(3./2.) * kT

    # Compute dimensionless free energy.
    # f = - \ln \int_{-\infty}^{+\infty} \exp[-\beta K x^2 / 2]
    #   = - \ln \int_{-\infty}^{+\infty} \exp[-x^2 / 2 \sigma^2]
    #   = - \ln [\sqrt{2 \pi} \sigma]
    values['f'] = - numpy.log(2 * numpy.pi * (sigma / units.angstroms)**2) * (3.0/2.0)

    return values

def test_replica_exchange(mpicomm=None, verbose=True):
    """
    Test that free energies and average potential energies of a 3D harmonic oscillator are correctly computed.

    TODO

    * Test ParallelTempering and HamiltonianExchange subclasses as well.
    * Test with different combinations of input parameters.

    """

    if verbose and ((not mpicomm) or (mpicomm.rank==0)): print "Testing replica exchange facility with harmonic oscillators: ",

    # Define mass of carbon atom.
    mass = 12.0 * units.amu

    # Define thermodynamic states.
    states = list() # thermodynamic states
    Ks = [500.00, 400.0, 300.0] * units.kilocalories_per_mole / units.angstroms**2 # spring constants
    temperatures = [300.0, 350.0, 400.0] * units.kelvin # temperatures
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
    #print "node %d : Storing data in temporary file: %s" % (mpicomm.rank, str(store_filename)) # DEBUG

    # Create and configure simulation object.
    simulation = ReplicaExchange(store_filename, mpicomm=mpicomm)
    simulation.create(states, seed_positions)
    simulation.platform_name = 'Reference'
    simulation.minimize = False
    simulation.number_of_iterations = 200
    simulation.nsteps_per_iteration = 500
    simulation.timestep = 2.0 * units.femtoseconds
    simulation.collision_rate = 20.0 / units.picosecond
    simulation.verbose = False
    simulation.show_mixing_statistics = False
    simulation.online_analysis = True

    # Run simulation.
    from yank import utils
    utils.config_root_logger(False)
    simulation.run() # run the simulation
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
        print "Delta_f_ij from online analysis"
        print online_analysis['Delta_f_ij']
        print "Delta_f_ij from final analysis"
        print analysis['Delta_f_ij']
        print "error"
        print error
        print "derror"
        print derror
        print "nsigma"
        print nsigma
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
        print "Delta_f_ij"
        print Delta_f_ij
        print "Delta_f_ij_analytical"
        print Delta_f_ij_analytical
        print "error"
        print error
        print "stderr"
        print dDelta_f_ij
        print "nsigma"
        print nsigma
        raise Exception("Dimensionless free energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    error = analysis['Delta_u_ij'] - Delta_u_ij_analytical
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / dDelta_f_ij[indices]
    if numpy.any(nsigma > MAX_SIGMA):
        print "Delta_u_ij"
        print analysis['Delta_u_ij']
        print "Delta_u_ij_analytical"
        print Delta_u_ij_analytical
        print "error"
        print error
        print "nsigma"
        print nsigma
        raise Exception("Dimensionless potential energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    # Clean up.
    del simulation

    if verbose: print "PASSED."
    return

def test_hamiltonian_exchange(mpicomm=None, verbose=True):
    """
    Test that free energies and average potential energies of a 3D harmonic oscillator are correctly computed
    when running HamiltonianExchange.

    TODO

    * Integrate with test_replica_exchange.
    * Test with different combinations of input parameters.

    """

    if verbose and ((not mpicomm) or (mpicomm.rank==0)): print "Testing Hamiltonian exchange facility with harmonic oscillators: ",

    # Create test system of harmonic oscillators
    testsystem = testsystems.HarmonicOscillatorArray()
    [system, coordinates] = [testsystem.system, testsystem.positions]

    # Define mass of carbon atom.
    mass = 12.0 * units.amu

    # Define thermodynamic states.
    sigmas = [0.2, 0.3, 0.4] * units.angstroms # standard deviations: beta K = 1/sigma^2 so K = 1/(beta sigma^2)
    temperature = 300.0 * units.kelvin # temperatures
    seed_positions = list()
    analytical_results = list()
    f_i_analytical = list() # dimensionless free energies
    u_i_analytical = list() # reduced potential
    systems = list() # Systems list for HamiltonianExchange
    for sigma in sigmas:
        # Compute corresponding spring constant.
        kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
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
    print ""
    print seed_positions
    print analytical_results
    print u_i_analytical
    print f_i_analytical    
    print ""

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
    #print "Storing data in temporary file: %s" % str(store_filename)

    # Create reference thermodynamic state.
    reference_state = ThermodynamicState(systems[0], temperature=temperature)

    # Create and configure simulation object.
    simulation = HamiltonianExchange(store_filename, mpicomm=mpicomm)
    simulation.create(reference_state, systems, seed_positions)
    simulation.platform_name = 'Reference'
    simulation.number_of_iterations = 200
    simulation.timestep = 2.0 * units.femtoseconds
    simulation.nsteps_per_iteration = 500
    simulation.collision_rate = 9.2 / units.picosecond
    simulation.verbose = False
    simulation.show_mixing_statistics = False

    # Run simulation.
    from yank import utils
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
        print "Delta_f_ij"
        print Delta_f_ij
        print "Delta_f_ij_analytical"
        print Delta_f_ij_analytical
        print "error"
        print error
        print "stderr"
        print dDelta_f_ij
        print "nsigma"
        print nsigma
        raise Exception("Dimensionless free energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    error = analysis['Delta_u_ij'] - Delta_u_ij_analytical
    nsigma = numpy.zeros([nstates,nstates], numpy.float32)
    nsigma[indices] = error[indices] / dDelta_f_ij[indices]
    if numpy.any(nsigma > MAX_SIGMA):
        print "Delta_u_ij"
        print analysis['Delta_u_ij']
        print "Delta_u_ij_analytical"
        print Delta_u_ij_analytical
        print "error"
        print error
        print "nsigma"
        print nsigma
        raise Exception("Dimensionless potential energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

    if verbose: print "PASSED."
    return

def test_parameters():
    """Test ReplicaExchange parameters initialization."""
    repex = ReplicaExchange(store_filename='test', nsteps_per_iteration=1e6)
    assert repex.nsteps_per_iteration == 1000000
    assert repex.collision_rate == repex.default_parameters['collision_rate']

@tools.raises(TypeError)
def test_unknown_parameters():
    """Test ReplicaExchange raises exception on wrong initialization."""
    ReplicaExchange(store_filename='test', wrong_parameter=False)

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # Configure logger.
    from yank import utils
    utils.config_root_logger(False)

    # Try MPI, if possible.
    try:
        from mpi4py import MPI # MPI wrapper
        hostname = os.uname()[1]
        mpicomm = MPI.COMM_WORLD
        if mpicomm.rank == 0:
            print "MPI initialized successfully."
    except Exception as e:
        print e
        print "Could not start MPI. Using serial code instead."
        mpicomm = None

    # Test simple system of harmonic oscillators.
    test_hamiltonian_exchange(mpicomm)
    test_replica_exchange(mpicomm)

