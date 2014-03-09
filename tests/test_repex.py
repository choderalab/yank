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
import numpy.linalg
import scipy.integrate

import simtk.openmm as openmm
import simtk.unit as units

#import scipy.io.netcdf as netcdf # scipy pure Python netCDF interface - GIVES US TROUBLE FOR NOW
import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now
#import tables as hdf5 # HDF5 will be supported in the future

from oldrepex import ThermodynamicState, ReplicaExchange, HamiltonianExchange, ParallelTempering

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def test_velocity_assignment(mpicomm=None, verbose=True):
    """
    Test Maxwell-Boltzmann velocity assignment subtroutine produces correct distribution, raising an exception if this test fails.

    """

    # Stop here if not root node.
    if mpicomm and (mpicomm.rank != 0): return

    if verbose: print "Testing Maxwell-Boltzmann velocity assignment: ",

    # Make a list of all test system constructors.
    import testsystems

    # Test parameters
    temperature = 298.0 * units.kelvin # test temperature
    kT = kB * temperature # thermal energy
    ntrials = 1000 # number of test trials
    systems_to_test = ['HarmonicOscillator', 'HarmonicOscillatorArray', 'AlanineDipeptideImplicit'] # systems to test
    
    for system_name in systems_to_test:
        #print '*' * 80
        #print system_name
   
        # Create system.
        constructor = getattr(testsystems, system_name)
        [system, coordinates] = constructor()

        # Create temporary filename.
        import tempfile # use a temporary file for testing
        file = tempfile.NamedTemporaryFile() 
        store_filename = file.name

        # Create repex instance.
        states = [ ThermodynamicState(system, temperature=temperature) ]
        simulation = ReplicaExchange(states=states, coordinates=coordinates, store_filename=store_filename)

        # Create integrator and context.
        natoms = system.getNumParticles()

        velocity_trials = numpy.zeros([ntrials, natoms, 3])
        kinetic_energy_trials = numpy.zeros([ntrials])

        for trial in range(ntrials):
            velocities = simulation._assign_Maxwell_Boltzmann_velocities(system, temperature)  
            kinetic_energy = 0.5 * units.sum(units.sum(system.masses * velocities**2)) 
            velocity_trials[trial,:,:] = velocities / (units.nanometers / units.picosecond)
            kinetic_energy_trials[trial] = kinetic_energy / units.kilocalories_per_mole
            
        velocity_mean = velocity_trials.mean(0)
        velocity_stderr = velocity_trials.std(0) / numpy.sqrt(ntrials)

        kinetic_analytical = (3.0/2.0) * natoms * kT / units.kilocalories_per_mole
        kinetic_mean = kinetic_energy_trials.mean()
        kinetic_error = kinetic_mean - kinetic_analytical
        kinetic_stderr = kinetic_energy_trials.std() / numpy.sqrt(ntrials)

        # Test if violations exceed tolerance.
        MAX_SIGMA = 6.0 # maximum number of standard errors allowed
        if numpy.any(numpy.abs(kinetic_error / kinetic_stderr) > MAX_SIGMA):
            print "analytical kinetic energy"
            print kinetic_analytical
            print "mean kinetic energy (kcal/mol)"
            print kinetic_mean
            print "difference (kcal/mol)"
            print kinetic_mean - kinetic_analytical
            print "stderr (kcal/mol)"
            print kinetic_stderr
            print "nsigma"
            print (kinetic_mean - kinetic_analytical) / kinetic_stderr
            raise Exception("Mean kinetic energy exceeds error tolerance of %.1f standard errors." % MAX_SIGMA)
        if numpy.any(numpy.abs(velocity_mean / velocity_stderr) > MAX_SIGMA):
            print "mean velocity (nm/ps)"
            print velocity_mean
            print "stderr (nm/ps)"
            print velocity_stderr
            print "nsigma"
            print velocity_mean / velocity_stderr
            raise Exception("Mean velocity exceeds error tolerance of %.1f standard errors." % MAX_SIGMA)
        
    if verbose: print "PASSED"
    return 

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
    Test that free energies and avergae potential energies of a 3D harmonic oscillator are correctly computed.

    TODO

    * Test ParallelTempering and HamiltonianExchange subclasses as well.
    * Test with different combinations of input parameters.
    
    """    

    if verbose and ((not mpicomm) or (mpicomm.rank==0)): print "Testing replica exchange facility with harmonic oscillators: ",

    # Create test system of harmonic oscillators
    import testsystems
    [system, coordinates] = testsystems.HarmonicOscillatorArray()

    # Define mass of carbon atom.
    mass = 12.0 * units.amu

    # Define thermodynamic states.
    states = list() # thermodynamic states
    sigmas = [0.2, 0.3, 0.4] * units.angstroms # standard deviations: beta K = 1/sigma^2 so K = 1/(beta sigma^2)
    temperatures = [300.0, 350.0, 400.0] * units.kelvin # temperatures
    seed_positions = list()
    analytical_results = list()
    f_i_analytical = list() # dimensionless free energies
    u_i_analytical = list() # reduced potential
    for (sigma, temperature) in zip(sigmas, temperatures):
        # Compute corresponding spring constant.
        kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA    
        kT = kB * temperature # thermal energy
        beta = 1.0 / kT # inverse temperature
        K = 1.0 / (beta * sigma**2)
        # Create harmonic oscillator system.
        [system, positions] = testsystems.HarmonicOscillator(K=K, mass=mass, mm=openmm)
        # Create thermodynamic state.
        state = ThermodynamicState(system=system, temperature=temperature)
        # Append thermodynamic state and positions.
        states.append(state)
        seed_positions.append(positions) 
        # Store analytical results.
        results = computeHarmonicOscillatorExpectations(K, mass, temperature) 
        analytical_results.append(results)
        f_i_analytical.append(results['f'])
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
    simulation = ReplicaExchange(states, seed_positions, store_filename, mpicomm=mpicomm) # initialize the replica-exchange simulation
    simulation.number_of_iterations = 1000 # set the simulation to only run 2 iterations
    simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    simulation.collision_rate = 0.001 / units.picosecond # DEBUG: Use a low collision rate
    simulation.platform = openmm.Platform.getPlatformByName('Reference') # use reference platform
    simulation.verbose = True # DEBUG

    # Run simulation.
    simulation.run() # run the simulation
    
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

def test_hamiltonian_exchange(mpi=None, verbose=True):
    """
    Test that free energies and avergae potential energies of a 3D harmonic oscillator are correctly computed
    when running HamiltonianExchange.

    TODO

    * Integrate with test_replica_exchange.
    * Test with different combinations of input parameters.
    
    """    

    if verbose and ((not mpicomm) or (mpicomm.rank==0)): print "Testing Hamiltonian exchange facility with harmonic oscillators: ",

    # Create test system of harmonic oscillators
    import testsystems
    [system, coordinates] = testsystems.HarmonicOscillatorArray()

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
        [system, positions] = testsystems.HarmonicOscillator(K=K, mass=mass, mm=openmm)
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
    simulation = HamiltonianExchange(reference_state, systems, seed_positions, store_filename, mpicomm=mpicomm) # initialize the replica-exchange simulation
    simulation.number_of_iterations = 1000 # set the simulation to only run 2 iterations
    simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    simulation.collision_rate = 9.2 / units.picosecond 
    simulation.platform = openmm.Platform.getPlatformByName('Reference') # use reference platform

    # Run simulation.
    simulation.run() # run the simulation

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

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
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

    # Test Maxwell-Boltzmann velocity assignment.
    test_velocity_assignment(mpicomm)
    
    # Test simple system of harmonic oscillators.
    test_hamiltonian_exchange(mpicomm)
    test_replica_exchange(mpicomm)

    
