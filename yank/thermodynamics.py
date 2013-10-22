#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Thermodynamic ensemble utilities.

DESCRIPTION

This module provides a utilities for simulating physical thermodynamic ensembles.

Provided classes include:

* ThermodynamicState - Data specifying a thermodynamic state obeying Boltzmann statistics (System/temperature/pressure/pH)

DEPENDENCIES

TODO

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import copy
import math
import numpy

import simtk.openmm 
import simtk.unit as units

#=============================================================================================
# REVISION CONTROL
#=============================================================================================

__version__ = "$Revision: $"

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# Thermodynamic state description
#=============================================================================================

class ThermodynamicState(object):
    """
    Data specifying a thermodynamic state obeying Boltzmann statistics.

    EXAMPLES

    Specify an NVT state for a water box at 298 K.

    >>> import simtk.unit as units
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.WaterBox()    
    >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin)

    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin, pressure=1.0*units.atmospheres)
    
    Note that the pressure is only relevant for periodic systems.

    NOTES

    This state object cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    TODO

    * Implement a more fundamental ProbabilityState as a base class?
    * Implement pH.

    """
    
    def __init__(self, system=None, temperature=None, pressure=None, mm=None):
        """
        Initialize the thermodynamic state.

        OPTIONAL ARGUMENTS

        system (simtk.openmm.System) - a System object describing the potential energy function for the system (default: None)
        temperature (simtk.unit.Quantity compatible with 'kelvin') - the temperature for a system with constant temperature (default: None)
        pressure (simtk.unit.Quantity compatible with 'atmospheres') - the pressure for constant-pressure systems (default: None)

        mm (simtk.openmm API) - OpenMM API implementation to use
        cache_context (boolean) - if True, will try to cache Context objects

        """

        # Initialize.
        self.system = None          # the System object governing the potential energy computation
        self.temperature = None     # the temperature
        self.pressure = None        # the pressure, or None if not isobaric

        self._mm = None             # Cached OpenMM implementation

        self._cache_context = True  # if True, try to cache Context object
        self._context = None        # cached Context
        self._integrator = None     # cached Integrator

        # Store OpenMM implementation.
        if mm:
            self._mm = mm
        else:
            self._mm = simtk.openmm

        # Store provided values.
        if system is not None:
            # TODO: Check to make sure system object implements OpenMM System API.
            self.system = copy.deepcopy(system) # TODO: Do this when deep copy works.
            # self.system = system # we make a shallow copy for now, which can cause trouble later
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

        # If temperature and pressure are specified, make sure MonteCarloBarostat is attached.
        if temperature and pressure:
            # Try to find barostat.
            barostat = False
            for force_index in range(self.system.getNumForces()):
                force = self.system.getForce(force_index)
                # Dispatch forces
                if isinstance(force, self._mm.MonteCarloBarostat):
                    barostat = force
                    break
            if barostat:
                # Set temperature.
                # TODO: Set pressure too, once that option is available.
                barostat.setTemperature(temperature)                
            else:
                # Create barostat.
                barostat = self._mm.MonteCarloBarostat(pressure, temperature)
                self.system.addForce(barostat)                    

        return

    def _create_context(self, platform=None):
        """
        Create Integrator and Context objects if they do not already exist.
        
        """

        # Check if we already have a Context defined.
        if self._context:
            #if platform and (platform != self._context.getPlatform()): # TODO: Figure out why requested and cached platforms differed in tests.
            if platform and (platform.getName() != self._context.getPlatform().getName()): # DEBUG: Only compare Platform names for now; change this later to incorporate GPU IDs.
                # Platform differs from the one requested; destroy it.
                print (platform.getName(), self._context.getPlatform().getName())
                print "Platform differs from the one requested; destroying and recreating..." # DEBUG
                del self._context, self._integrator
            else:
                # Cached context is what we expect; do nothing.
                return

        # Create an integrator.
        timestep = 1.0 * units.femtosecond
        self._integrator = self._mm.VerletIntegrator(timestep)        
            
        # Create a new OpenMM context.
        if platform:                
            self._context = self._mm.Context(self.system, self._integrator, platform)
        else:
            self._context = self._mm.Context(self.system, self._integrator)
            
        print "_create_context created a new integrator and context" # DEBUG

        return

    def _cleanup_context(self):
        del self._context, self._integrator
        self._context = None
        self._integrator = None

    def _compute_potential(self, coordinates, box_vectors):
        # Set coordinates and periodic box vectors.
        self._context.setPositions(coordinates)
        if box_vectors is not None: self._context.setPeriodicBoxVectors(*box_vectors)                
        
        # Retrieve potential energy.
        openmm_state = self._context.getState(getEnergy=True)
        potential_energy = openmm_state.getPotentialEnergy()

        return potential_energy
    
    def reduced_potential(self, coordinates, box_vectors=None, mm=None, platform=None):
        """
        Compute the reduced potential for the given coordinates in this thermodynamic state.

        ARGUMENTS

        coordinates (simtk.unit.Quantity of Nx3 numpy.array) - coordinates[n,k] is kth coordinate of particle n

        OPTIONAL ARGUMENTS
        
        box_vectors - periodic box vectors

        RETURNS

        u (float) - the unitless reduced potential (which can be considered to have units of kT)
        
        EXAMPLES

        Compute the reduced potential of a Lennard-Jones cluster at 100 K.
        
        >>> import simtk.unit as units
        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system, coordinates] = testsystems.LennardJonesCluster()
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)
        >>> potential = state.reduced_potential(coordinates)
    
        Compute the reduced potential of a Lennard-Jones fluid at 100 K and 1 atm.

        >>> [system, coordinates] = testsystems.LennardJonesFluid()
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin, pressure=1.0*units.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(coordinates, box_vectors)

        Compute the reduced potential of a water box at 298 K and 1 atm.

        >>> [system, coordinates] = testsystems.WaterBox()
        >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin, pressure=1.0*units.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(coordinates, box_vectors)
    
        NOTES

        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is he inverse temperature
        U(x) is the potential energy
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of titratible groups)
        
        REFERENCES

        [1] Shirts MR and Chodera JD. Statistically optimal analysis of equilibrium states. J Chem Phys 129:124105, 2008.

        TODO

        * Instead of requiring configuration and box_vectors be passed separately, develop a Configuration or Snapshot class.
        
        """

        # Select OpenMM implementation if not specified.
        if mm is None: mm = simtk.openmm

        # If pressure is specified, ensure box vectors have been provided.
        if (self.pressure is not None) and (box_vectors is None):
            raise ParameterException("box_vectors must be specified if constant-pressure ensemble.")

        # Make sure we have Context and Integrator objects.
        self._create_context(platform)

        # Compute energy.
        try:
            potential_energy = self._compute_potential(coordinates, box_vectors)
        except Exception as e:
            print e # DEBUG

            # Our cached context failed, so try deleting it and creating it anew.
            self._cleanup_context()
            self._create_context()

            # Compute energy
            potential_energy = self._compute_potential(coordinates, box_vectors)            
        
        # Compute inverse temperature.
        beta = 1.0 / (kB * self.temperature)

        # Compute reduced potential.
        reduced_potential = beta * potential_energy
        if self.pressure is not None:
            reduced_potential += beta * self.pressure * self._volume(box_vectors) * units.AVOGADRO_CONSTANT_NA

        # Clean up context if requested, or if we're using Cuda (which can only have one active Context at a time).
        if (not self._cache_context) or (self._context.getPlatform().getName() == 'Cuda'):
            self._cleanup_context()

        return reduced_potential

    def reduced_potential_multiple(self, coordinates_list, box_vectors_list=None, mm=None, platform=None):
        """
        Compute the reduced potential for the given sets of coordinates in this thermodynamic state.
        This can pontentially be more efficient than repeated calls to reduced_potential.

        ARGUMENTS

        coordinates_list (list of simtk.unit.Quantity of Nx3 numpy.array) - coordinates[n,k] is kth coordinate of particle n

        OPTIONAL ARGUMENTS
        
        box_vectors - periodic box vectors

        RETURNS

        u_k (K numpy array of float) - the unitless reduced potentials (which can be considered to have units of kT)
        
        EXAMPLES

        Compute the reduced potential of a Lennard-Jones cluster at multiple configurations at 100 K.
        
        >>> import simtk.unit as units
        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system, coordinates] = testsystems.LennardJonesCluster()
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)
        >>> # create example list of coordinates
        >>> import copy
        >>> coordinates_list = [ copy.deepcopy(coordinates) for i in range(10) ]
        >>> # compute potential for all sets of coordinates
        >>> potentials = state.reduced_potential_multiple(coordinates_list)

        NOTES

        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is he inverse temperature
        U(x) is the potential energy
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of titratible groups)
        
        REFERENCES

        [1] Shirts MR and Chodera JD. Statistically optimal analysis of equilibrium states. J Chem Phys 129:124105, 2008.

        TODO

        * Instead of requiring configuration and box_vectors be passed separately, develop a Configuration or Snapshot class.
        
        """

        # Select OpenMM implementation if not specified.
        if mm is None: mm = simtk.openmm

        # If pressure is specified, ensure box vectors have been provided.
        if (self.pressure is not None) and (box_vectors_list is None):
            raise ParameterException("box_vectors must be specified if constant-pressure ensemble.")

        # Make sure we have Context and Integrator objects.
        self._create_context(platform)

        # Allocate storage.
        K = len(coordinates_list)
        u_k = numpy.zeros([K], numpy.float64)

        # Compute energies.
        for k in range(K):
            # Compute energy
            if box_vectors_list:
                potential_energy = self._compute_potential(coordinates_list[k], box_vectors_list[k])
            else:
                potential_energy = self._compute_potential(coordinates_list[k], None)
        
            # Compute inverse temperature.
            beta = 1.0 / (kB * self.temperature)

            # Compute reduced potential.
            u_k[k] = beta * potential_energy
            if self.pressure is not None:
                u_k[k] += beta * self.pressure * self._volume(box_vectors_list[k]) * units.AVOGADRO_CONSTANT_NA

        # Clean up context if requested, or if we're using Cuda (which can only have one active Context at a time).
        if (not self._cache_context) or (self._context.getPlatform().getName() == 'Cuda'):
            self._cleanup_context()

        return u_k

    def is_compatible_with(self, state):
        """
        Determine whether another state is in the same thermodynamic ensemble (e.g. NVT, NPT).

        ARGUMENTS

        state (ThermodynamicState) - thermodynamic state whose compatibility is to be determined

        RETURNS
        
        is_compatible (boolean) - True if 'state' is of the same ensemble (e.g. both NVT, both NPT), False otherwise

        EXAMPLES

        Create NVT and NPT states.
        
        >>> import simtk.unit as units
        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system, coordinates] = testsystems.LennardJonesCluster()
        >>> nvt_state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)
        >>> npt_state = ThermodynamicState(system=system, temperature=100.0*units.kelvin, pressure=1.0*units.atmospheres)

        Test compatibility.

        >>> test1 = nvt_state.is_compatible_with(nvt_state)
        >>> test2 = nvt_state.is_compatible_with(npt_state)
        >>> test3 = npt_state.is_compatible_with(nvt_state)        
        >>> test4 = npt_state.is_compatible_with(npt_state)        

        """

        is_compatible = True

        # Make sure systems have the same number of atoms.
        if (self.system.getNumParticles() != state.system.getNumParticles()):
            is_compatible = False

        # Make sure other terms are defined for both states.
        # TODO: Use introspection to get list of parameters?
        for parameter in ['temperature', 'pressure']:
            if (parameter in dir(self)) is not (parameter in dir(state)):
                # parameter is not shared by both states
                is_compatible = False

        return is_compatible

    def __repr__(self):
        """
        Returns a string representation of a state.

        EXAMPLES

        Create an NVT state.
        
        >>> import simtk.unit as units
        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system, coordinates] = testsystems.LennardJonesCluster()
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)

        Return a representation of the state.
        
        >>> state_string = repr(state)

        """

        r = "<ThermodynamicState object"
        if self.temperature is not None:
            r += ", temperature = %s" % str(self.temperature)
        if self.pressure is not None:
            r += ", pressure = %s" % str(self.pressure)
        r += ">"

        return r

    def __str__(self):
        # TODO: Write a human-readable representation.
        
        return repr(r)

    def _volume(self, box_vectors):
        """
        Return the volume of the current configuration.

        RETURNS

        volume (simtk.unit.Quantity) - the volume of the system (in units of length^3), or None if no box coordinates are defined

        EXAMPLES
        
        Compute the volume of a Lennard-Jones fluid at 100 K and 1 atm.

        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system, coordinates] = testsystems.LennardJonesFluid()
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin, pressure=1.0*units.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> volume = state._volume(box_vectors)
        
        """

        # Compute volume of parallelepiped.
        [a,b,c] = box_vectors
        A = numpy.array([a/a.unit, b/a.unit, c/a.unit])
        volume = numpy.linalg.det(A) * a.unit**3
        return volume

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

