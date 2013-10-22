#!/usr/local/bin/env python

"""
Test YANK using simple models.

DESCRIPTION

This test suite generates a number of simple models to test the 'Yank' facility.

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
    # Test Lennard-Jones particle association.
    #

    # Default parameters for argon.
    mass          = 39.9 * units.amu
    charge        = 0.05 * units.elementary_charge
    sigma         = 3.350 * units.angstrom
    epsilon       = 100 * 0.001603 * units.kilojoule_per_mole

    # Simulation parameters.
    temperature   = 300.0 * units.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    
    import simtk.openmm as openmm

    # Create receptor.
    receptor_system = openmm.System()
    receptor_system.addParticle(10 * mass)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    charge = +charge * units.elementary_charge # DEBUG
    force.addParticle(charge, sigma, epsilon)
    receptor_system.addForce(force)
    
    # Create ligand.
    ligand_system = openmm.System()
    ligand_system.addParticle(mass)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    charge = -charge * units.elementary_charge # DEBUG
    force.addParticle(charge, sigma, epsilon)
    ligand_system.addForce(force)

    # Prevent receptor from diffusing away by imposing a spring.
    #sigma = 0.5 * units.angstroms
    #force = openmm.CustomExternalForce('(Kext/2) * (x^2 + y^2 + z^2)')
    #force.addGlobalParameter('Kext', kT / sigma**2)
    #force.addParticle(0, [])
    #receptor_system.addForce(force)

    # Create complex coordinates.
    import simtk.unit as units
    complex_coordinates = units.Quantity(numpy.zeros([2,3], numpy.float64), units.angstroms)
    complex_coordinates[1,0] = 10.0 * units.angstroms

    # Create temporary directory for testing.
    import tempfile
    output_directory = tempfile.mkdtemp()

    # Initialize YANK object.
    from yank import Yank
    yank = Yank(receptor=receptor_system, ligand=ligand_system, complex_coordinates=[complex_coordinates], output_directory=output_directory, verbose=True)
    yank.solvent_protocol = yank.vacuum_protocol
    yank.complex_protocol = yank.vacuum_protocol
    yank.restraint_type = 'flat-bottom'
    yank.temperature = temperature
    yank.niterations = 100
    yank.platform = openmm.Platform.getPlatformByName("Reference")
    yank.verbose = False

    # Run the simulation.
    yank.run()
    
    #
    # Analyze the data.
    #
    
    results = yank.analyze(verbose=True)

    # TODO: Check results against analytical results.

    
