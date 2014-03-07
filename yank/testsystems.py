#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Module to generate Systems and coordinates for simple reference molecular systems for testing.

DESCRIPTION

This module provides functions for building a number of test systems of varying complexity,
useful for testing both OpenMM and various codes based on pyopenmm.

Note that the PYOPENMM_SOURCE_DIR must be set to point to where the PyOpenMM package is unpacked.

EXAMPLES

Create a 3D harmonic oscillator.

>>> import testsystems
>>> [system, coordinates] = testsystems.HarmonicOscillator()

See list of methods for a complete list of provided test systems.

COPYRIGHT

@author Randall J. Radmer <radmer@stanford.edu>
@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

* Add units checking code to check arguments.
* Change default arguments to Quantity objects, rather than None?

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import numpy
import math

import simtk
import simtk.openmm
import simtk.unit as units

#=============================================================================================
# 3D Harmonic Oscillator
#=============================================================================================

def HarmonicOscillator(K=None, mass=None, mm=None):
    """
    Create a 3D harmonic oscillator, with a single particle confined in an isotropic harmonic well.

    OPTIONAL ARGUMENTS
    
    K (simtk.unit.Quantity of energy/particle/distance^2) - harmonic restraining potential (default: 100.0 * units.kilocalories_per_mole/units.angstrom**2)    
    mass (simtk.unit.Quantity of mass) - particle mass (default: 39.948 * units.amu)
    mm (simtk.openmm or compatible) - openmm implementation (default: simtk.openmm)

    RETURNS

    system (simtk.chem.System) - the system object
    coordinates (simtk.unit.Quantity of distance, Nx3 numpy array) - coordinates of all particles in system

    NOTES

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    EXAMPLES

    Create a 3D harmonic oscillator with default parameters.

    >>> [system, coordinates] = HarmonicOscillator()

    Create a harmonic oscillator with specified mass and spring constant.

    >>> mass = 12.0 * units.amu
    >>> K = 1.0 * units.kilocalories_per_mole / units.angstroms**2
    >>> [system, coordinates] = HarmonicOscillator(K=K, mass=mass)

    """

    # Use pyOpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if K       is None:  K           = 100.0 * units.kilocalories_per_mole / units.angstroms**2
    if mass    is None:  mass        = 39.948 * units.amu # arbitrary reference mass            

    # Create an empty system object.
    system = mm.System()

    # Add the particle to the system.
    system.addParticle(mass)

    # Set the coordinates.
    coordinates = units.Quantity(numpy.zeros([1,3], numpy.float32), units.angstroms)

    # Add a restrining potential centered at the origin.
    force = mm.CustomExternalForce('(K/2.0) * (x^2 + y^2 + z^2)')
    force.addGlobalParameter('K', K)
    force.addParticle(0, [])
    system.addForce(force)

    return (system, coordinates)

#=============================================================================================
# Free diatomic harmonic oscillator
#=============================================================================================

def Diatom(K=None, r0=None, m1=None, m2=None, constraint=False, use_central_potential=False, mm=None):
    """
    Create a diatomic molecule with a single harmonic bond between the two atoms.

    OPTIONAL ARGUMENTS
    
    K (simtk.unit.Quantity of energy/particle/distance^2) - harmonic bond potential (default: 290.1 * units.kilocalories_per_mole / units.angstrom**2, Amber GAFF c-c bond)
    r0 (simtk.unit.Quantity of distance) - bond length (default: 1.550 * units.angstrom, Amber GAFF c-c bond)
    constraint (boolean) - if True, the bond length will be constrained (default: False)
    m1 (simtk.unit.Quantity of mass) - particle 1 mass (default: 12.01 * units.amu, Amber GAFF c atom)
    m2 (simtk.unit.Quantity of mass) - particle 2 mass (default: 12.01 * units.amu, Amber GAFF c atom)    
    mm (simtk.openmm or compatible) - openmm implementation (default: simtk.openmm)
    use_central_potential (boolean) - if True, a soft central potential will also be added to keep the system from drifting away

    RETURNS

    system (simtk.chem.System) - the system object
    coordinates (simtk.unit.Quantity of distance, Nx3 numpy array) - coordinates of all particles in system

    NOTES

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    EXAMPLES

    Create a diatom with default parameters.

    >>> [system, coordinates] = Diatom()

    """

    # Use pyOpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if K       is None:  K           = 290.1 * units.kilocalories_per_mole / units.angstrom**2
    if r0      is None:  r0          = 1.550 * units.angstroms
    if m1      is None:  m1          = 39.948 * units.amu 
    if m2      is None:  m2          = 39.948 * units.amu

    # Create an empty system object.
    system = mm.System()

    # Add two particles to the system.
    system.addParticle(m1)
    system.addParticle(m2)

    # Add a harmonic bond.
    force = mm.HarmonicBondForce()
    force.addBond(0, 1, r0, K)
    system.addForce(force)

    if constraint:
        # Add constraint between particles.
        system.addConstraint(0, 1, r0)
    
    # Set the coordinates.
    coordinates = units.Quantity(numpy.zeros([2,3], numpy.float32), units.angstroms)
    coordinates[1,0] = r0

    if use_central_potential:
        # Add a central restraining potential.
        Kcentral = 1.0 * units.kilocalories_per_mole / units.nanometer**2
        force = mm.CustomExternalForce('(Kcentral/2.0) * (x^2 + y^2 + z^2)')
        force.addGlobalParameter('K', Kcentral)
        force.addParticle(0, [])
        force.addParticle(1, [])    
        system.addForce(force)

    return (system, coordinates)

#=============================================================================================
# Constraint-coupled Pair of Harmonic Oscillators
#=============================================================================================

def ConstraintCoupledHarmonicOscillator(K=None, d=None, mass=None, mm=None):
    """
    Create a pair of particles in 3D harmonic oscillator wells, coupled by a constraint.

    @keyword K: harmonic restraining potential (default: 1.0 * units.kilojoules_per_mole/units.nanometer**2)    
    @keyword d: distance between harmonic oscillators (default: 1.0 * units.nanometer)
    @keyword mass: particle mass (default: 39.948 * units.amu)

    @return system: the system
    @type system: a System object

    @return coordinates: initial coordinates for the system
    @type coordinates: a Coordinates object

    EXAMPLES

    Create a constraint-coupled 3D harmonic oscillator with default parameters.

    >>> [system, coordinates] = ConstraintCoupledHarmonicOscillator()

    Create a constraint-coupled harmonic oscillator with specified mass, distance, and spring constant.

    >>> mass = 12.0 * units.amu
    >>> d = 5.0 * units.angstroms
    >>> K = 1.0 * units.kilocalories_per_mole / units.angstroms**2
    >>> [system, coordinates] = ConstraintCoupledHarmonicOscillator(K=K, d=d, mass=mass)

    """

    # Use pyOpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if K       is None:  K           = 1.0 * units.kilojoules_per_mole / units.nanometer**2
    if d       is None:  d           = 1.0 * units.nanometer
    if mass    is None:  mass        = 39.948 * units.amu # arbitrary reference mass            

    # Create an empty system object.
    system = mm.System()

    # Add particles to the system.
    system.addParticle(mass)
    system.addParticle(mass)    

    # Set the coordinates.
    coordinates = units.Quantity(numpy.zeros([2,3], numpy.float32), units.angstroms)
    coordinates[1,0] = d

    # Add a restrining potential centered at the origin.
    force = mm.CustomExternalForce('(K/2.0) * ((x-d)^2 + y^2 + z^2)')
    force.addGlobalParameter('K', K)
    force.addPerParticleParameter('d')    
    force.addParticle(0, [0.0])
    force.addParticle(1, [d / units.nanometers])    
    system.addForce(force)

    # Add constraint between particles.
    system.addConstraint(0, 1, d)   

    # Add a harmonic bond force as well so minimization will roughly satisfy constraints.
    force = mm.HarmonicBondForce()
    K = 10.0 * units.kilocalories_per_mole / units.angstrom**2 # force constant
    force.addBond(0, 1, d, K)
    system.addForce(force)

    return (system, coordinates)

#=============================================================================================
# Array of harmonic oscillators
#=============================================================================================

def HarmonicOscillatorArray(K=None, d=None, mass=None, N=None, mm=None):
    """
    Create a 1D array of noninteracting particles in 3D harmonic oscillator wells.

    @keyword K: harmonic restraining potential (default: 90.0 * units.kilocalories_per_mole/units.angstroms**2)    
    @keyword d: distance between harmonic oscillators (default: 1.0 * units.nanometer)
    @keyword mass: particle mass (default: 39.948 * units.amu)

    @return system: the system
    @type system: a System object

    @return coordinates: initial coordinates for the system
    @type coordinates: a Coordinates object

    EXAMPLES

    Create a constraint-coupled 3D harmonic oscillator with default parameters.

    >>> [system, coordinates] = HarmonicOscillatorArray()

    Create a constraint-coupled harmonic oscillator with specified mass, distance, and spring constant.

    >>> mass = 12.0 * units.amu
    >>> d = 5.0 * units.angstroms
    >>> K = 1.0 * units.kilocalories_per_mole / units.angstroms**2
    >>> N = 10 # number of oscillators
    >>> [system, coordinates] = HarmonicOscillatorArray(K=K, d=d, mass=mass, N=N)

    """

    # Use pyOpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if K       is None:  K           = 90.0 * units.kilocalories_per_mole/units.angstroms**2
    if d       is None:  d           = 1.0 * units.nanometer
    if mass    is None:  mass        = 39.948 * units.amu 
    if N       is None:  N           = 5 

    # Create an empty system object.
    system = mm.System()

    # Add particles to the system.
    for n in range(N):
        system.addParticle(mass)

    # Set the coordinates for a 1D array of particles spaced d apart along the x-axis.
    coordinates = units.Quantity(numpy.zeros([N,3], numpy.float32), units.angstroms)
    for n in range(N):
        coordinates[n,0] = n*d

    # Add a restrining potential for each oscillator.
    force = mm.CustomExternalForce('(K/2.0) * ((x-x0)^2 + y^2 + z^2)')
    force.addGlobalParameter('K', K)
    force.addPerParticleParameter('x0')
    for n in range(N):
        parameters = (d*n / units.nanometers, )
        force.addParticle(n, parameters)
    system.addForce(force)

    return (system, coordinates)

#=============================================================================================
# Salt crystal
#=============================================================================================

def SodiumChlorideCrystal(mm=None):
    """
    Create an FCC crystal of sodium chloride.

    Each atom is represented by a charged Lennard-Jones sphere in an Ewald lattice.

    @return system: the system
    @type system: a System object

    @return coordinates: initial coordinates for the system
    @type coordinates: a Coordinates object

    EXAMPLES

    Create sodium chloride crystal unit.
    
    >>> [system, coordinates] = SodiumChlorideCrystal()
    
    TODO

    * Lennard-Jones interactions aren't correctly being included now, due to LJ cutoff.  Fix this by hard-coding LJ interactions?
    * Add nx, ny, nz arguments to allow user to specify replication of crystal unit in x,y,z.
    * Choose more appropriate lattice parameters and lattice spacing.

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Set default parameters (from Tinker).
    mass_Na     = 22.990 * units.amu
    mass_Cl     = 35.453 * units.amu
    q_Na        = 1.0 * units.elementary_charge 
    q_Cl        =-1.0 * units.elementary_charge 
    sigma_Na    = 3.330445 * units.angstrom
    sigma_Cl    = 4.41724 * units.angstrom
    epsilon_Na  = 0.002772 * units.kilocalorie_per_mole 
    epsilon_Cl  = 0.118 * units.kilocalorie_per_mole 

    # Create system
    system = mm.System()

    # Set box vectors.
    box_size = 5.628 * units.angstroms # box width
    a = units.Quantity(numpy.zeros([3]), units.nanometers); a[0] = box_size
    b = units.Quantity(numpy.zeros([3]), units.nanometers); b[1] = box_size
    c = units.Quantity(numpy.zeros([3]), units.nanometers); c[2] = box_size
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Create nonbonded force term.
    force = mm.NonbondedForce()

    # Set interactions to be periodic Ewald.
    force.setNonbondedMethod(mm.NonbondedForce.Ewald)

    # Set cutoff to be less than one half the box length.
    cutoff = box_size / 2.0 * 0.99
    force.setCutoffDistance(cutoff)
    
    # Allocate storage for coordinates.
    natoms = 2
    coordinates = units.Quantity(numpy.zeros([natoms,3], numpy.float32), units.angstroms)

    # Add sodium ion.
    system.addParticle(mass_Na)
    force.addParticle(q_Na, sigma_Na, epsilon_Na)
    coordinates[0,0] = 0.0 * units.angstrom
    coordinates[0,1] = 0.0 * units.angstrom
    coordinates[0,2] = 0.0 * units.angstrom
    
    # Add chloride atom.
    system.addParticle(mass_Cl)
    force.addParticle(q_Cl, sigma_Cl, epsilon_Cl)
    coordinates[1,0] = 2.814 * units.angstrom
    coordinates[1,1] = 2.814 * units.angstrom
    coordinates[1,2] = 2.814 * units.angstrom

    # Add nonbonded force term to the system.
    system.addForce(force)
       
    # Return system and coordinates.
    return (system, coordinates)

#=============================================================================================
# Cluster of Lennard-Jones particles
#=============================================================================================

def LennardJonesCluster(nx=3, ny=3, nz=3, K=None, mm=None):
    """
    Create a non-periodic rectilinear grid of Lennard-Jones particles in a harmonic restraining potential.

    @keyword nx: number of particles in x-dimension (default: 3)
    @keyword ny: number of particles in y-dimension (default: 3)
    @keyword nz: number of particles in z-dimension (default: 3)
    @keyword K: harmonic restraining potential (default: 1.0 * units.kilojoules_per_mole/units.nanometer**2)

    @return system: the system
    @type system: a System object

    @return coordinates: initial coordinates for the system
    @type coordinates: a Coordinates object

    EXAMPLES

    Create default 3x3x3 Lennard-Jones cluster in a harmonic restraining potential.

    >>> [system, coordinates] = LennardJonesCluster()

    Create a larger 10x10x10 grid of Lennard-Jones particles.

    >>> [system, coordinates] = LennardJonesCluster(nx=10, ny=10, nz=10)

    """

    # Use pyOpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    mass_Ar     = 39.9 * units.amu
    q_Ar        = 0.0 * units.elementary_charge
    sigma_Ar    = 3.350 * units.angstrom
    epsilon_Ar  = 0.001603 * units.kilojoule_per_mole

    # Set spring constant.
    if K is None:
        K = 1.0 * units.kilojoules_per_mole/units.nanometer**2

    scaleStepSizeX = 1.0
    scaleStepSizeY = 1.0
    scaleStepSizeZ = 1.0

    # Determine total number of atoms.
    natoms = nx * ny * nz

    # Create an empty system object.
    system = mm.System()

    # Create a NonbondedForce object with no cutoff.
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)

    coordinates = units.Quantity(numpy.zeros([natoms,3],numpy.float32), units.angstrom)

    atom_index = 0
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nz):
                system.addParticle(mass_Ar)
                nb.addParticle(q_Ar, sigma_Ar, epsilon_Ar)
                x = sigma_Ar*scaleStepSizeX*(ii - nx/2.0)
                y = sigma_Ar*scaleStepSizeY*(jj - ny/2.0)
                z = sigma_Ar*scaleStepSizeZ*(kk - nz/2.0)

                coordinates[atom_index,0] = x
                coordinates[atom_index,1] = y
                coordinates[atom_index,2] = z
                atom_index += 1

    # Add the nonbonded force.
    system.addForce(nb)

    # Add a restrining potential centered at the origin.
    force = mm.CustomExternalForce('(K/2.0) * (x^2 + y^2 + z^2)')
    force.addGlobalParameter('K', K)
    for particle_index in range(natoms):
        force.addParticle(particle_index, [])
    system.addForce(force)

    return (system, coordinates)

#=============================================================================================
# Periodic box of Lennard-Jones particles
#=============================================================================================

def LennardJonesFluid(mm=None, nx=6, ny=6, nz=6, mass=None, sigma=None, epsilon=None, cutoff=None, switch=False, dispersion_correction=True):
    """
    Create a periodic rectilinear grid of Lennard-Jones particles.

    DESCRIPTION

    Parameters for argon are used by default.
    Cutoff is set to 3 sigma by default.
    
    OPTIONAL ARGUMENTS

    mm (implements simtk.openmm) - mm implementation to use (default: simtk.openmm)
    nx, ny, nz (int) - number of atoms to initially place on grid in each dimension (default: 6)
    mass (simtk.unit.Quantity with units of mass) - particle masses (default: 39.9 amu)
    sigma (simtk.unit.Quantity with units of length) - Lennard-Jones sigma parameter (default: 3.4 A)
    epsilon (simtk.unit.Quantity with units of energy) - Lennard-Jones well depth (default: 0.234 kcal/mol)
    cutoff (simtk.unit.Quantity with units of length) - cutoff for nonbonded interactions (default: 2.5 * sigma)
    switch (simtk.unit.Quantity with units of length) - if specified, the switching function will be turned on at this distance (default: None)
    dispersion_correction (boolean) - if True, will use analytical dispersion correction (if not using switching function) (default: True)

    EXAMPLES

    Create default-size Lennard-Jones fluid.

    >>> [system, coordinates] = LennardJonesFluid()

    Create a larger 10x8x5 box of Lennard-Jones particles.

    >>> [system, coordinates] = LennardJonesFluid(nx=10, ny=8, nz=5)

    Create Lennard-Jones fluid using switched particle interactions (switched off betwee 7 and 9 A) and more particles.

    >>> [system, coordinates] = LennardJonesFluid(nx=10, ny=10, nz=10, switch=7.0*units.angstroms, cutoff=9.0*units.angstroms)


    """

    # Use pyOpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if mass is None: mass = 39.9 * units.amu # argon
    if sigma is None: sigma = 3.4 * units.angstrom # argon
    if epsilon is None: epsilon = 0.238 * units.kilocalories_per_mole # argon
    charge        = 0.0 * units.elementary_charge
    if cutoff is None: cutoff = 2.5 * sigma

    scaleStepSizeX = 1.0
    scaleStepSizeY = 1.0
    scaleStepSizeZ = 1.0

    # Determine total number of atoms.
    natoms = nx * ny * nz

    # Create an empty system object.
    system = mm.System()

    # Set up periodic nonbonded interactions with a cutoff.
    if switch:
        energy_expression = "LJ * S;"
        energy_expression += "LJ = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
        #energy_expression += "sigma = 0.5 * (sigma1 + sigma2);"
        #energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
        energy_expression += "S = (cutoff^2 - r^2)^2 * (cutoff^2 + 2*r^2 - 3*switch^2) / (cutoff^2 - switch^2)^3;"
        nb = mm.CustomNonbondedForce(energy_expression)
        nb.addGlobalParameter('switch', switch)
        nb.addGlobalParameter('cutoff', cutoff)
        nb.addGlobalParameter('sigma', sigma)
        nb.addGlobalParameter('epsilon', epsilon)
        nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(cutoff)        
    else:
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)    
        nb.setCutoffDistance(cutoff)
        nb.setUseDispersionCorrection(dispersion_correction)
        
    coordinates = units.Quantity(numpy.zeros([natoms,3],numpy.float32), units.angstrom)

    maxX = 0.0 * units.angstrom
    maxY = 0.0 * units.angstrom
    maxZ = 0.0 * units.angstrom

    atom_index = 0
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nz):
                system.addParticle(mass)
                if switch:
                    nb.addParticle([])                    
                else:
                    nb.addParticle(charge, sigma, epsilon)
                x = sigma*scaleStepSizeX*ii
                y = sigma*scaleStepSizeY*jj
                z = sigma*scaleStepSizeZ*kk

                coordinates[atom_index,0] = x
                coordinates[atom_index,1] = y
                coordinates[atom_index,2] = z
                atom_index += 1
                
                # Wrap coordinates as needed.
                if x>maxX: maxX = x
                if y>maxY: maxY = y
                if z>maxZ: maxZ = z
                
    # Set periodic box vectors.
    x = maxX+2*sigma*scaleStepSizeX
    y = maxY+2*sigma*scaleStepSizeY
    z = maxZ+2*sigma*scaleStepSizeZ

    a = units.Quantity((x,                0*units.angstrom, 0*units.angstrom))
    b = units.Quantity((0*units.angstrom,                y, 0*units.angstrom))
    c = units.Quantity((0*units.angstrom, 0*units.angstrom, z))
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Add the nonbonded force.
    system.addForce(nb)

    return (system, coordinates)

#=============================================================================================
# Periodic box of Lennard-Jones particles implemented via CustomNonbondedForce instead of NonbondedForce.
#=============================================================================================

def CustomLennardJonesFluid(mm=None, nx=6, ny=6, nz=6, mass=None, sigma=None, epsilon=None, cutoff=None, switch=False, dispersion_correction=True):
    """
    Create a periodic rectilinear grid of Lennard-Jones particles.

    DESCRIPTION

    Parameters for argon are used by default.
    Cutoff is set to 3 sigma by default.
    
    OPTIONAL ARGUMENTS

    mm (implements simtk.openmm) - mm implementation to use (default: simtk.openmm)
    nx, ny, nz (int) - number of atoms to initially place on grid in each dimension (default: 6)
    mass (simtk.unit.Quantity with units of mass) - particle masses (default: 39.9 amu)
    sigma (simtk.unit.Quantity with units of length) - Lennard-Jones sigma parameter (default: 3.4 A)
    epsilon (simtk.unit.Quantity with units of energy) - Lennard-Jones well depth (default: 0.234 kcal/mol)
    cutoff (simtk.unit.Quantity with units of length) - cutoff for nonbonded interactions (default: 2.5 * sigma)
    switch (simtk.unit.Quantity with units of length) - if specified, the switching function will be turned on at this distance (default: None)
    dispersion_correction (boolean) - if True, will use analytical dispersion correction (if not using switching function) (default: True)

    NOTE

    No analytical dispersion correction is included here.

    EXAMPLES

    Create default-size Lennard-Jones fluid.

    >>> [system, coordinates] = LennardJonesFluid()

    Create a larger 10x8x5 box of Lennard-Jones particles.

    >>> [system, coordinates] = CustomLennardJonesFluid(nx=10, ny=8, nz=5)

    Create Lennard-Jones fluid using switched particle interactions (switched off betwee 7 and 9 A) and more particles.

    >>> [system, coordinates] = CustomLennardJonesFluid(nx=10, ny=10, nz=10, switch=7.0*units.angstroms, cutoff=9.0*units.angstroms)


    """

    # Use OpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if mass is None: mass = 39.9 * units.amu # argon
    if sigma is None: sigma = 3.4 * units.angstrom # argon
    if epsilon is None: epsilon = 0.238 * units.kilocalories_per_mole # argon
    charge        = 0.0 * units.elementary_charge
    if cutoff is None: cutoff = 2.5 * sigma

    scaleStepSizeX = 1.0
    scaleStepSizeY = 1.0
    scaleStepSizeZ = 1.0

    # Determine total number of atoms.
    natoms = nx * ny * nz

    # Create an empty system object.
    system = mm.System()

    # Set up periodic nonbonded interactions with a cutoff.
    if switch:
        energy_expression = "LJ * S;"
        energy_expression += "LJ = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
        energy_expression += "S = (cutoff^2 - r^2)^2 * (cutoff^2 + 2*r^2 - 3*switch^2) / (cutoff^2 - switch^2)^3;"
        nb = mm.CustomNonbondedForce(energy_expression)
        nb.addGlobalParameter('switch', switch)
        nb.addGlobalParameter('cutoff', cutoff)
        nb.addGlobalParameter('sigma', sigma)
        nb.addGlobalParameter('epsilon', epsilon)
        nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(cutoff)        
    else:
        energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
        nb = mm.CustomNonbondedForce(energy_expression)
        nb.addGlobalParameter('sigma', sigma)
        nb.addGlobalParameter('epsilon', epsilon)
        nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(cutoff)        
        
    coordinates = units.Quantity(numpy.zeros([natoms,3],numpy.float32), units.angstrom)

    maxX = 0.0 * units.angstrom
    maxY = 0.0 * units.angstrom
    maxZ = 0.0 * units.angstrom

    atom_index = 0
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nz):
                system.addParticle(mass)
                nb.addParticle([])                    
                x = sigma*scaleStepSizeX*ii
                y = sigma*scaleStepSizeY*jj
                z = sigma*scaleStepSizeZ*kk

                coordinates[atom_index,0] = x
                coordinates[atom_index,1] = y
                coordinates[atom_index,2] = z
                atom_index += 1
                
                # Wrap coordinates as needed.
                if x>maxX: maxX = x
                if y>maxY: maxY = y
                if z>maxZ: maxZ = z
                
    # Set periodic box vectors.
    x = maxX+2*sigma*scaleStepSizeX
    y = maxY+2*sigma*scaleStepSizeY
    z = maxZ+2*sigma*scaleStepSizeZ

    a = units.Quantity((x,                0*units.angstrom, 0*units.angstrom))
    b = units.Quantity((0*units.angstrom,                y, 0*units.angstrom))
    c = units.Quantity((0*units.angstrom, 0*units.angstrom, z))
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Add the nonbonded force.
    system.addForce(nb)

    # Add long-range correction.
    if switch:
        # TODO
        pass
    else:
        volume = x*y*z
        density = natoms / volume
        per_particle_dispersion_energy = -(8./3.)*math.pi*epsilon*(sigma**6)/(cutoff**3)*density  # attraction
        per_particle_dispersion_energy += (8./9.)*math.pi*epsilon*(sigma**12)/(cutoff**9)*density  # repulsion
        energy_expression = "%f" % (per_particle_dispersion_energy / units.kilojoules_per_mole)
        force = mm.CustomExternalForce(energy_expression)
        for i in range(natoms):
            force.addParticle(i, [])
        system.addForce(force)
    
    return (system, coordinates)

#=============================================================================================
# Ideal gas (noninteracting particles in a periodic box)
#=============================================================================================

def IdealGas(nparticles=216, mm=None, mass=None, temperature=None, pressure=None, volume=None):
    """
    Create an 'ideal gas' of noninteracting particles in a periodic box.

    @keyword nparticles: number of particles (default: 216)
    @type nparticles: int
    
    @return system: the system
    @type system: a System object

    @return coordinates: initial coordinates for the system
    @type coordinates: a Coordinates object

    EXAMPLES

    Create an ideal gas system.

    >>> [system, coordinates] = IdealGas()

    Create a smaller ideal gas system containing 64 particles.

    >>> [system, coordinates] = IdealGas(nparticles=64)

    """

    # Use OpenMM by default.
    if mm is None:
        mm = simtk.openmm

    # Default parameters
    if mass is None: mass = 39.9 * units.amu # argon mass
    if temperature is None: temperature = 298.0 * units.kelvin
    if pressure is None: pressure = 1.0 * units.atmosphere
    if volume is None: volume = (nparticles * temperature * units.BOLTZMANN_CONSTANT_kB / pressure).in_units_of(units.nanometers**3)

    charge   = 0.0 * units.elementary_charge
    sigma    = 3.350 * units.angstrom # argon LJ 
    epsilon  = 0.0 * units.kilojoule_per_mole # zero interaction

    # Create an empty system object.
    system = mm.System()
    
    # Compute box size.
    length = volume**(1.0/3.0)
    a = units.Quantity((length,           0*units.nanometer, 0*units.nanometer))
    b = units.Quantity((0*units.nanometer,           length, 0*units.nanometer))
    c = units.Quantity((0*units.nanometer, 0*units.nanometer, length))
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Add particles.
    for index in range(nparticles):
        system.addParticle(mass)
    
    # Place particles at random coordinates within the box.
    # TODO: Use reproducible seed.
    # NOTE: This may not be thread-safe.
    import numpy.random
    state = numpy.random.get_state()
    numpy.random.seed(0)
    coordinates = units.Quantity((length/units.nanometer) * numpy.random.rand(nparticles,3), units.nanometer)
    numpy.random.set_state(state)

    return (system, coordinates)

#=============================================================================================
# Periodic water box
#=============================================================================================

def WaterBox(constrain=True, mm=None, nonbonded_method=None, filename=None, charges=True, box_edge=2.3*units.nanometers, cutoff=0.9*units.nanometers):
    """
    Create a test system containing a periodic box of TIP3P water.

    Flexible bonds and angles are always added, and constraints are optional (but on by default).
    Addition of flexible bond and angle terms doesn't affect constrained dynamics, but allows for minimization to work properly.

    OPTIONAL ARGUMENTS

    filename (string) - name of file containing water coordinates (default: 'watbox216.pdb')
    mm (OpenMM implementation) - name of simtk.openmm implementation to use (default: simtk.openmm)
    flexible (boolean) - if True, will add harmonic OH bonds and HOH angle between
    constrain (boolean) - if True, will also constrain OH and HH bonds in water (default: True)
    nonbonded_method
    box_edge (simtk.unit.Quantity with units compatible with nanometers) - edge length for cubic box [should be greater than 2*cutoff] (default: 2.3 nm)
    cutoff  (simtk.unit.Quantity with units compatible with nanometers) - nonbonded cutoff (default: 0.9 * units.nanometers)

    RETURNS

    system (System)
    coordinates (numpy array)

    EXAMPLES

    Create a 216-water system.
    
    >>> [system, coordinates] = WaterBox()

    TODO

    * Allow size of box (either dimensions or number of waters) to be specified, replicating equilibrated waterbox to fill these dimensions.

    """
    import simtk.openmm.app as app

    # Load forcefield for solvent model.
    ff =  app.ForceField('tip3p.xml')

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = units.Quantity((), units.angstroms)
    
    # Create new Modeller instance.
    m = app.Modeller(top, pos)

    # Add solvent to specified box dimensions.
    boxSize = units.Quantity(numpy.ones([3]) * box_edge/box_edge.unit, box_edge.unit)
    m.addSolvent(ff, boxSize=boxSize)
   
    # Get new topology and coordinates.
    newtop = m.getTopology()
    newpos = m.getPositions()
   
    # Convert positions to numpy.
    positions = units.Quantity(numpy.array(newpos / newpos.unit), newpos.unit)
   
    # Create OpenMM System.
    if nonbonded_method:
        nonbondedMethod  = nonbonded_method
    else:
        # Use periodic system.
        nonbondedMethod = app.CutoffPeriodic
        
    if constrain:
        constraints = app.HBonds
    else:
        constraints = None
    
    system = ff.createSystem(newtop, nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff, constraints=constraints, rigidWater=True, removeCMMotion=False)

    return [system, positions]

#=============================================================================================
# Alanine dipeptide in implicit solvent
#=============================================================================================

def AlanineDipeptideImplicit(mm=None, flexibleConstraints=True, shake='h-bonds'):
    """
    Alanine dipeptide ff96 in OBC GBSA implicit solvent.
    
    EXAMPLES
    
    >>> [system, coordinates] = AlanineDipeptideImplicit()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    prmtop_filename = os.path.join(os.path.dirname(__file__), 'data', 'alanine-dipeptide-gbsa', 'alanine-dipeptide.prmtop')
    crd_filename = os.path.join(os.path.dirname(__file__), 'data', 'alanine-dipeptide-gbsa', 'alanine-dipeptide.crd')

    # Initialize system.
    import simtk.openmm.app as app
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(implicitSolvent=app.OBC1, constraints=app.HBonds, nonbondedCutoff=None)

    # Read coordinates.
    inpcrd = app.AmberInpcrdFile(crd_filename)
    coordinates = inpcrd.getPositions(asNumpy=True)

    return (system, coordinates)

#=============================================================================================
# Alanine dipeptide in explicit solvent
#=============================================================================================

def AlanineDipeptideExplicit(mm=None, flexibleConstraints=True, shake='h-bonds', nonbondedCutoff=None):
    """
    Alanine dipeptide ff96 in TIP3P explicit solvent with PME electrostatics.

    OPTIONAL ARGUMENTS

    mm (simtk.openmm or compatible) - openmm implementation (default: simtk.openmm)
    nonbondedCutof(simtk.unit.Quantity of units length) - nonbonded cutoff (default: 9 A)
    
    EXAMPLES
    
    >>> [system, coordinates] = AlanineDipeptideExplicit()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Set defaults.
    if nonbondedCutoff is None:
        nonbondedCutoff = 9.0 * units.angstroms

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    prmtop_filename = os.path.join(os.path.dirname(__file__), 'data', 'alanine-dipeptide-explicit', 'alanine-dipeptide.prmtop')
    crd_filename = os.path.join(os.path.dirname(__file__), 'data', 'alanine-dipeptide-explicit', 'alanine-dipeptide.crd')

    # Initialize system.
    import simtk.openmm.app as app
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(constraints=app.HBonds, nonbondedMethod=app.PME, rigidWater=True, nonbondedCutoff=0.9*units.nanometer)

    # Read coordinates.
    inpcrd = app.AmberInpcrdFile(crd_filename, loadBoxVectors=True)
    coordinates = inpcrd.getPositions(asNumpy=True)

    # Set box vectors.
    box_vectors = inpcrd.getBoxVectors(asNumpy=True)
    system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])
    
    return (system, coordinates)

#=============================================================================================
# T4 lysozyme L99A with p-xylene ligand in implicit OBC GBSA solvent
#=============================================================================================

def LysozymeImplicit(mm=None, flexibleConstraints=True, shake='h-bonds'):
    """
    T4 lysozyme L99A (AMBER ff96) with p-xylene ligand (GAFF + AM1-BCC) in implicit OBC GBSA solvent.

    OPTIONS

    flexibleConstraints (boolean) - if True, harmonic bonds will be added to constrained bonds to allow minimization 
    
    EXAMPLES
    
    >>> [system, coordinates] = LysozymeImplicit()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    prmtop_filename = os.path.join(os.path.dirname(__file__), 'data',  'T4-lysozyme-L99A-implicit', 'complex.prmtop')
    crd_filename = os.path.join(os.path.dirname(__file__), 'data',  'T4-lysozyme-L99A-implicit', 'complex.crd')

    # Initialize system.
    import simtk.openmm.app as app
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(implicitSolvent=app.OBC1, constraints=app.HBonds, nonbondedCutoff=None)

    # Read coordinates.
    inpcrd = app.AmberInpcrdFile(crd_filename)
    coordinates = inpcrd.getPositions(asNumpy=True)
    
    return (system, coordinates)

#=============================================================================================
# Src kinase in implicit OBC GBSA solvent
#=============================================================================================

def SrcImplicit(mm=None):
    """
    Src kinase in implicit AMBER 99sb-ildn with OBC GBSA solvent.

    OPTIONS

    EXAMPLES
    
    >>> [system, coordinates] = SrcImplicit()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    pdb_filename = os.path.join(os.path.dirname(__file__), 'data',  'src-implicit', 'implicit-refined.pdb')

    # Read PDB.
    import simtk.openmm.app as app
    pdbfile = app.PDBFile(pdb_filename)

    # Construct system.
    forcefields_to_use = ['amber99sbildn.xml', 'amber99_obc.xml'] # list of forcefields to use in parameterization
    forcefield = app.ForceField(*forcefields_to_use)
    system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    # Get coordinates.
    coordinates = pdbfile.getPositions()
    
    return (system, coordinates)

#=============================================================================================
# Src kinase in explicit TIP3P solvent
#=============================================================================================

def SrcExplicit(mm=None):
    """
    Src kinase (AMBER 99sb-ildn) in explicit TIP3P solvent.

    OPTIONS

    EXAMPLES
    
    >>> [system, coordinates] = SrcExplicit()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    system_xml_filename = os.path.join(os.path.dirname(__file__), 'data',  'src-explicit', 'system.xml')
    state_xml_filename = os.path.join(os.path.dirname(__file__), 'data',  'src-explicit', 'state.xml')

    # Read system.
    infile = open(system_xml_filename, 'r')
    system = mm.XmlSerializer.deserialize(infile.read())
    infile.close()

    # Read state.
    infile = open(state_xml_filename, 'r')
    serialized_state = mm.XmlSerializer.deserialize(infile.read())
    infile.close()

    coordinates = serialized_state.getPositions()
    box_vectors = serialized_state.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    
    return (system, coordinates)

#=============================================================================================
# Methanol box.
#=============================================================================================

def MethanolBox(mm=None, flexibleConstraints=True, shake='h-bonds', nonbondedCutoff=None, nonbondedMethod='CutoffPeriodic'):
    """
    Methanol box.

    OPTIONAL ARGUMENTS

    mm (simtk.openmm or compatible) - openmm implementation (default: simtk.openmm)
    nonbondedCutoff (simtk.unit.Quantity of units length) - nonbonded cutoff (default: 7 A)
    monbondedMethod (string) - nonbonded method (default: 'CutoffPeriodic')
    
    EXAMPLES
    
    >>> [system, coordinates] = MethanolBox()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Set defaults.
    if nonbondedCutoff is None:
        nonbondedCutoff = 7.0 * units.angstroms

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    system_name = 'methanol-box'
    prmtop_filename = os.path.join(os.path.dirname(__file__), 'data',  system_name, system_name + '.prmtop')
    crd_filename = os.path.join(os.path.dirname(__file__), 'data',  system_name, system_name + '.crd')

    # Initialize system.
    import simtk.openmm.app as app
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(constraints=app.HBonds, nonbondedMethod=app.PME, rigidWater=True, nonbondedCutoff=0.9*units.nanometer)

    # Read coordinates.
    inpcrd = app.AmberInpcrdFile(crd_filename, loadBoxVectors=True)
    coordinates = inpcrd.getPositions(asNumpy=True)

    # Set box vectors.
    box_vectors = inpcrd.getBoxVectors(asNumpy=True)
    system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])
    
    return (system, coordinates)

#=============================================================================================
# Molecular ideal gas (methanol box).
#=============================================================================================

def MolecularIdealGas(mm=None, flexibleConstraints=True, shake=None, nonbondedCutoff=None, nonbondedMethod='CutoffPeriodic'):
    """
    Molecular ideal gas (methanol box).

    OPTIONAL ARGUMENTS

    mm (simtk.openmm or compatible) - openmm implementation (default: simtk.openmm)
    nonbondedCutoff (simtk.unit.Quantity of units length) - nonbonded cutoff (default: 7 A)
    shake (String) - if 'h-bonds', will SHAKE all bonds to hydrogen and water; if 'all-bonds', will SHAKE all bonds and water (default: None)    
    monbondedMethod (string) - nonbonded method (default: 'CutoffPeriodic')
    
    EXAMPLES
    
    >>> [system, coordinates] = MolecularIdealGas()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    # Set defaults.
    if nonbondedCutoff is None:
        nonbondedCutoff = 7.0 * units.angstroms

    # Determine prmtop and crd filenames in test directory.
    # TODO: This will need to be revised in order to be able to find the test systems.
    system_name = 'methanol-box'
    prmtop_filename = os.path.join(os.path.dirname(__file__), 'data', system_name, system_name + '.prmtop')
    crd_filename = os.path.join(os.path.dirname(__file__), 'data', system_name, system_name + '.crd')

    # Initialize system.
    import simtk.openmm.app as app
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    reference_system = prmtop.createSystem(constraints=app.HBonds, nonbondedMethod=app.PME, rigidWater=True, nonbondedCutoff=0.9*units.nanometer)

    # Make a new system that contains no intermolecular interactions.
    system = mm.System()
        
    # Add atoms.
    for atom_index in range(reference_system.getNumParticles()):
        mass = reference_system.getParticleMass(atom_index)
        system.addParticle(mass)

    # Add constraints
    for constraint_index in range(reference_system.getNumConstraints()):
        [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
        system.addConstraint(iatom, jatom, r0)

    # Copy only intramolecular forces.
    nforces = reference_system.getNumForces()
    for force_index in range(nforces):
        reference_force = reference_system.getForce(force_index)
        if isinstance(reference_force, mm.HarmonicBondForce):
            # HarmonicBondForce
            force = mm.HarmonicBondForce()
            for bond_index in range(reference_force.getNumBonds()):
                [iatom, jatom, r0, K] = reference_force.getBondParameters(bond_index)
                force.addBond(iatom, jatom, r0, K)
            system.addForce(force)
        elif isinstance(reference_force, mm.HarmonicAngleForce):
            # HarmonicAngleForce
            force = mm.HarmonicAngleForce()
            for angle_index in range(reference_force.getNumAngles()):
                [iatom, jatom, katom, theta0, Ktheta] = reference_force.getAngleParameters(angle_index)
                force.addAngle(iatom, jatom, katom, theta0, Ktheta)
            system.addForce(force)
        elif isinstance(reference_force, mm.PeriodicTorsionForce):
            # PeriodicTorsionForce
            force = mm.PeriodicTorsionForce()
            for torsion_index in range(reference_force.getNumTorsions()):
                [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
            system.addForce(force)
        else:
            # Don't add any other forces.
            pass

    # Read coordinates.
    inpcrd = app.AmberInpcrdFile(crd_filename, loadBoxVectors=True)
    coordinates = inpcrd.getPositions(asNumpy=True)

    # Set box vectors.
    box_vectors = inpcrd.getBoxVectors(asNumpy=True)
    system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])
    
    return (system, coordinates)

#=============================================================================================
# CustomGBForce system
#=============================================================================================

def CustomGBForceSystem(mm=None):
    """
    A system of particles with a CustomGBForce.

    NOTES

    This example comes from TestReferenceCustomGBForce.cpp from the OpenMM distribution.
    
    EXAMPLES
    
    >>> [system, coordinates] = CustomGBForceSystem()

    """

    # Choose OpenMM package if not specified.
    if mm is None:
        mm = simtk.openmm

    numMolecules = 70
    numParticles = numMolecules*2
    boxSize = 10.0 * units.nanometers

    # Default parameters
    mass     = 39.9 * units.amu
    sigma    = 3.350 * units.angstrom
    epsilon  = 0.001603 * units.kilojoule_per_mole
    cutoff   = 2.0 * units.nanometers
    
    system = mm.System()
    for i in range(numParticles):
        system.addParticle(mass)

    system.setDefaultPeriodicBoxVectors(mm.Vec3(boxSize, 0.0, 0.0), mm.Vec3(0.0, boxSize, 0.0), mm.Vec3(0.0, 0.0, boxSize))

    # Create NonbondedForce.
    nonbonded = mm.NonbondedForce()    
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(cutoff)

    # Create CustomGBForce.
    custom = mm.CustomGBForce()
    custom.setNonbondedMethod(mm.CustomGBForce.CutoffPeriodic)
    custom.setCutoffDistance(cutoff)
    
    custom.addPerParticleParameter("q")
    custom.addPerParticleParameter("radius")
    custom.addPerParticleParameter("scale")

    custom.addGlobalParameter("solventDielectric", 80.0)
    custom.addGlobalParameter("soluteDielectric", 1.0)
    custom.addComputedValue("I", "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
                                  "U=r+sr2;"
                                  "C=2*(1/or1-1/L)*step(sr2-r-or1);"
                                  "L=max(or1, D);"
                                  "D=abs(r-sr2);"
                                  "sr2 = scale2*or2;"
                                  "or1 = radius1-0.009; or2 = radius2-0.009", mm.CustomGBForce.ParticlePairNoExclusions);
    custom.addComputedValue("B", "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-0.009", mm.CustomGBForce.SingleParticle);
    custom.addEnergyTerm("28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", mm.CustomGBForce.SingleParticle);
    custom.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                          "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", mm.CustomGBForce.ParticlePairNoExclusions);

    # Add particles.
    for i in range(numMolecules):
        if (i < numMolecules/2):
            charge = 1.0 * units.elementary_charge
            radius = 0.2 * units.nanometers
            scale = 0.5
            nonbonded.addParticle(charge, sigma, epsilon)
            custom.addParticle([charge, radius, scale])

            charge = -1.0 * units.elementary_charge
            radius = 0.1 * units.nanometers
            scale = 0.5
            nonbonded.addParticle(charge, sigma, epsilon)            
            custom.addParticle([charge, radius, scale]);
        else:
            charge = 1.0 * units.elementary_charge
            radius = 0.2 * units.nanometers
            scale = 0.8
            nonbonded.addParticle(charge, sigma, epsilon)
            custom.addParticle([charge, radius, scale])

            charge = -1.0 * units.elementary_charge
            radius = 0.1 * units.nanometers
            scale = 0.8
            nonbonded.addParticle(charge, sigma, epsilon)            
            custom.addParticle([charge, radius, scale]);

    system.addForce(nonbonded)
    system.addForce(custom)    

    # Place particles at random coordinates within the box.
    # TODO: Use reproducible random number seed.
    # NOTE: This may not be thread-safe.
    import numpy.random
    state = numpy.random.get_state()
    numpy.random.seed(0)
    coordinates = units.Quantity((boxSize/units.nanometer) * numpy.random.rand(numParticles,3), units.nanometer)
    numpy.random.set_state(state)

    return [system, coordinates]

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest

    # Test all systems on Reference platform.
    platform = simtk.openmm.Platform.getPlatformByName("Reference")
    doctest.testmod()    

