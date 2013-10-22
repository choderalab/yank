#!/usr/local/bin/env python

"""
Test replica-exchange facility repex.py using analytically soluble models.

DESCRIPTION

This test suite generates a number of analytically soluble harmonic oscillator models to test
the simulation algorithms defined in the 'repex' replica exchange facility.

COPYRIGHT

Written by John D. Chodera <jchodera@gmail.com> while at the University of California Berkeley.

LICENSE

This code is licensed under the latest available version of the GNU General Public License.

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

import simtk.openmm 
import simtk.unit as units

import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now

from thermodynamics import ThermodynamicState
import repex

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def computeHarmonicOscillatorExpectations(K, mass, temperature):
    """
    Compute moments of potential and kinetic energy distributions and free energies for harmonic oscillator.
        
    ARGUMENTS
    
    K - spring constant
    mass - mass of particle
    temperature - temperature
    
    RETURNS
    
    values (dict) - values['potential'] is a dict with 'mean' and 'stddev' of potential energy distribution;
                    values['kinetic'] is a dict with 'mean' and 'stddev' of kinetic energy distribution;
                    values['free energies'] is the free energy due to the 'potential', 'kinetic', or 'total' part of the partition function
    
    NOTES

    Numerical quadrature is used to evaluate the moments of the potential energy distribution.

    EXAMPLES
    
    >>> import simtk.unit as units
    >>> temperature = 298.0 * units.kelvin
    >>> sigma = 0.5 * units.angstroms # define standard deviation for harmonic oscillator
    >>> mass = 12.0 * units.amu # mass of harmonic oscillator
    >>> kT = kB * temperature # thermal energy
    >>> beta = 1.0 / kT # inverse temperature
    >>> K = kT / sigma**2 # spring constant consistent with variance sigma**2
    >>> values = computeHarmonicOscillatorExpectations(K, mass, temperature)

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
    import scipy.integrate
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

    # Compute free energies.
    V0 = (units.liter / units.AVOGADRO_CONSTANT_NA / units.mole).in_units_of(units.angstroms**3) # standard state reference volume (1M)
    values['free energies'] = dict()
    values['free energies']['potential'] = - numpy.log((numpy.sqrt(2.0 * math.pi) * sigma)**3 / V0)
    values['free energies']['kinetic'] = (3./2.) 
    values['free energies']['total'] = values['free energies']['potential'] + values['free energies']['kinetic']
    
    return values

def detect_equilibration(A_t):
    """
    Automatically detect equilibrated region.

    ARGUMENTS

    A_t (numpy.array) - timeseries

    RETURNS

    t (int) - start of equilibrated data
    g (float) - statistical inefficiency of equilibrated data
    Neff_max (float) - number of uncorrelated samples   
    
    """
    T = A_t.size

    g_t = numpy.ones([T-1], numpy.float32)
    Neff_t = numpy.ones([T-1], numpy.float32)
    for t in range(T-1):
        g_t[t] = timeseries.statisticalInefficiency(A_t[t:T])
        Neff_t[t] = (T-t+1) / g_t[t]
    
    Neff_max = Neff_t.max()
    t = Neff_t.argmax()
    g = g_t[t]
    
    return (t, g, Neff_max)

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # Test subroutines.
    import doctest
    doctest.testmod()

    import repex
    import simtk.pyopenmm.extras.testsystems as testsystems
    import timeseries
    import pymbar

    verbose = True

    # Use reference platform.
    platform = simtk.openmm.Platform.getPlatformByName("Reference")

    #
    # Test parallel tempering.
    #

    # Simulation parameters.
    Tmin = 270.0 * units.kelvin # minimum temperature
    Tmax = 600.0 * units.kelvin # maximum temperature
    nstates = 3 # number of temperature replicas
    niterations = 100
    sigma = 0.5 * units.angstroms # define standard deviation for harmonic oscillator
    mass = 12.0 * units.amu # mass of harmonic oscillator
    K = kB * Tmin / sigma**2 # spring constant consistent with variance sigma**2
    [system, coordinates] = testsystems.HarmonicOscillator(K=K, mass=mass)
    # Create temporary file for testing.
    import tempfile
    file = tempfile.NamedTemporaryFile() 
    store_filename = file.name
    # Create simulation.
    simulation = repex.ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=nstates)
    simulation.verbose = False
    simulation.platform = platform
    simulation.number_of_equilibration_iterations = 5 # set number of equilibration iterations
    simulation.number_of_iterations = niterations # set the simulation to only run 10 iterations
    simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    simulation.nsteps_per_iteration = 500 
    simulation.minimize = False # don't minimize first
    simulation.show_mixing_statistics = False 
    # Run the simulation.
    simulation.run()
    
    #
    # Analyze the data.
    #

    # Deconvolute replicas
    ncfile = netcdf.Dataset(store_filename, 'r')
    u_kln = numpy.zeros([nstates, nstates, niterations], numpy.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = ncfile.variables['energies'][iteration,:,:]
    ncfile.close()
    # Extract log probability history.
    u_n = numpy.zeros([niterations], numpy.float64)
    for iteration in range(niterations):
        u_n[iteration] = 0.0        
        for state in range(nstates):
            u_n[iteration] += u_kln[state,state,iteration]
    # Detect equilibration.
    [nequil, g, Neff] = detect_equilibration(u_n)
    u_n = u_n[nequil:]
    u_kln = u_kln[:,:,nequil:]
    # Subsample data.
    indices = timeseries.subsampleCorrelatedData(u_n, g=g)
    u_n = u_n[indices]
    u_kln = u_kln[:,:,indices]
    N_k = len(indices) * numpy.ones([nstates], numpy.int32)
    # Analyze with MBAR.
    mbar = pymbar.MBAR(u_kln, N_k)
    [Delta_f_ij, dDelta_f_ij] = mbar.getFreeEnergyDifferences()
    # Compare with analytical.
    f_i_analytical = numpy.zeros([nstates], numpy.float64)    
    for (state_index, state) in enumerate(simulation.states):
        values = computeHarmonicOscillatorExpectations(K, mass, state.temperature)
        f_i_analytical[state_index] = values['free energies']['potential'] 
    Delta_f_ij_analytical = numpy.zeros([nstates, nstates], numpy.float64)
    for i in range(nstates):
        for j in range(nstates):
            Delta_f_ij_analytical[i,j] = f_i_analytical[j] - f_i_analytical[i]        

    # Show difference.
    #print "estimated"
    #print Delta_f_ij
    #print dDelta_f_ij
    #print "analytical"
    #print Delta_f_ij_analytical

    # Check agreement.
    SIGNIFICANCE_CUTOFF = 6.0
    nerrors = 0
    for i in range(nstates):
        for j in range(i+1,nstates):
            error = Delta_f_ij[i,j] - Delta_f_ij_analytical[i,j]
            nsigma = abs(error) / dDelta_f_ij[i,j]
            if (nsigma > SIGNIFICANCE_CUTOFF):
                print "WARNING: states (%d,%d) estimated %f +- %f | analytical %f | error %f +- %f | nsigma %.1f" % (i, j, Delta_f_ij[i,j], dDelta_f_ij[i,j], Delta_f_ij_analytical[i,j], error, dDelta_f_ij[i,j], nsigma)
                nerrors += 1
    
    if (nerrors > 0):
        print ""
        print "WARNING: There were %d warnings." % nerrors
        sys.exit(1)
    else:
        print "There were no warnings."
        print ""
        sys.exit(0)
