#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Expanded-ensemble simulation algorithm and variants.

DESCRIPTION

This module provides a general facility for running expanded-ensemble simulations, as well as
derived classes for special cases such as simulated tempering (in which the states differ only
in temperature) and simulated scaling (in which the state differ only by potential function).

Provided classes include:

* ExpandedEnsemble - Base class for general expanded-ensemble simulations among specified ThermodynamicState objects
* SimulatedTempering - Convenience subclass of ExpandedEnsemble for parallel tempering simulations (one System object, many temperatures/pressures)
* SimulatedScaling - Convenience subclass of ExpandedEnsemble for simulated scaling simulations (many System objects, same temperature/pressure)

NOTES

This implementation does not yet contain weight adaptation algorithms.

REFERENCES

Expanded ensembles:

*Lyubartsev AP, Martsinovskii AA, Shevkunov SV, Vorontsov-Velyaminov PN.
New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles.
J. Chem. Phys. 96:1776, 1992.

Simulated tempering:

Simulated scaling:

DEPENDENCIES

Use of this module requires the following

* NetCDF (compiled with netcdf4 support) and HDF5

http://www.unidata.ucar.edu/software/netcdf/
http://www.hdfgroup.org/HDF5/

* netcdf4-python (a Python interface for netcdf4)

http://code.google.com/p/netcdf4-python/

* numpy and scipy

http://www.scipy.org/

TODO

* Break ThermodynamicState out into a shared file.
* Add another layer of abstraction so that the base class uses generic log probabilities, rather than reduced potentials.
* Add support for HDF5 storage models.
* See if we can get scipy.io.netcdf interface working, or more easily support additional NetCDF implementations / autodetect available implementations.
* Use interface-based checking of arguments so that different implementations of the OpenMM API can be used.

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import math
import copy
import time

import numpy
import numpy.linalg

import simtk.openmm 
import simtk.unit as units

#import scipy.io.netcdf as netcdf # scipy pure Python netCDF interface - GIVES US TROUBLE FOR NOW
import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now
#import tables as hdf5 # HDF5 will be supported in the future

from thermodynamics import ThermodynamicState

#=============================================================================================
# REVISION CONTROL
#=============================================================================================

__version__ = "$Revision: $"

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# Exceptions
#=============================================================================================

class NotImplementedException(Exception):
    """
    Exception denoting that the requested feature has not yet been implemented.

    """
    pass

class ParameterException(Exception):
    """
    Exception denoting that an incorrect argument has been specified.

    """
    pass
    
#=============================================================================================
# Weight adaptation scheme.
#=============================================================================================

class WeightAdaptationScheme(object):
    """
    Abstract base class for adaptation of weights for expanded ensemble simulation.

    """

    def initialize(self, states):
        pass

    def resume(self, states):
        pass

    def store(self, states):
        pass
        
    
    def adapt_weights(self, state_history, energy_history, log_weight_history):
        """
        Adapt weights, given simulation history.

        ARGUMENTS

        state_history (numpy array) - state_history[iteration] is the history of previously visited states
        energy_history (numpy array) - energy_history[iteration,state_index] is the history of energies
        log_weight_history (numpy array) - log_weight_history[iteration,state_index] is the history of log weights used for each iteration

        """

        raise NotImplementedException("WeightAdaptationScheme is an abstract base class.  Please use a derived class, like WangLandau.")

class WangLandau(WeightAdaptationScheme):
    def __init__(self, log_weight_delta=1.0, flatness_fraction=0.8, switch_methods=True):
        """
        Wang-Landau weight adaptation scheme.

        Weight adaptation will switch to 1/niterations once 

        ARGUMENTS
        
        log_weight_delta (float) - value by which log weights are initially modified
        flatness_fraction (float) - state histograms are judged to be 'flat' and log_weight_delta updated when all all counts are greater than flatness_fraction * ncounts / nstates.

        OPTIONAL ARGUMENTS

        switch_methods (boolean) - if True, switch to the 1/t method (where t = number of iterations) once log_weight_delta becomes sufficiently small

        REFERENCES

        """

        self.log_weight_delta = log_weight_delta
        self.flatness_fraction = flatness_fraction
        self.switch_methods = switch_methods

        return
    
    def adapt_weights(self, state_history, energy_history, log_weight_history):
        """
        Adapt weights, given simulation history.

        ARGUMENTS

        state_history (numpy array) - state_history[iteration] is the history of previously visited states
        energy_history (numpy array) - energy_history[iteration,state_index] is the history of energies
        log_weight_history (numpy array) - log_weight_history[iteration,state_index] is the history of log weights used for each iteration

        """

        # Get dimensions.
        niterations = energy_history.shape[0]
        nstates = energy_history.shape[1]

        # Determine last time log weights have changed.
        weights_last_changed = niterations-1
        while ( (weights_last_changed > 0) and (log_weight_history[weights_last_changed,:] == log_weight_history[-1,:]) ):
            weights_last_changed -= 1
        weights_last_changed += 1

        # Compute histogram for this duration.
        histogram = numpy.zeros([nstates], numpy.float64)
        for iteration in range(weights_last_changed, niterations):
            state_index = state_history[iteration]
            histogram[state_index] += 1.0
        
        # Determine if flatness criterion has been 

        return log_weights    

#=============================================================================================
# Expanded-ensemble simulation.
#=============================================================================================

class ExpandedEnsemble(object):
    """
    Expanded-ensemble simulation facility.

    This base class provides a general expanded-ensemble simulation facility, allowing any set of thermodynamic states
    to be specified, along with a set of initial coordinates.  No distinction is made between one-dimensional and
    multidimensional state layout; by default, the state switching scheme scheme attempts to select the next state
    from the stationary distribution of states given fixed configuration.    

    (Modification of the 'exchange_scheme' setting will allow the tranditional 'neighbor swaps only' scheme to be used.)

    While this base class is fully functional, it does not make use of the special structure of simulated tempering or
    simulated scaling variants of expanded ensemble simulations.  The SimulatedTempering and SimulatedScaling classes
    should therefore be used for these algorithms, since they are more efficient and provide more convenient ways to
    initialize the simulation classes.

    Stored configurations, energies, swaps, and restart information are all written to a single output file using
    the platform portable, robust, and efficient NetCDF4 library.  Plans for future HDF5 support are pending.    

    THEORY

    Expanded ensemble simulations sample a joint configuration and thermodynamic state index pair (x,k) from

    \pi(x,k) \propto exp[-u_k(x) + g_k]

    where u_k is the reduced potential for state k, and g_k is the corresponding log biasing weight.
    
    ATTRIBUTES

    The following parameters (attributes) can be set after the object has been created, but before it has been
    initialized by a call to run():

    * collision_rate (units: 1/time) - the collision rate used for Langevin dynamics (default: 90 ps^-1)
    * constraint_tolerance (dimensionless) - relative constraint tolerance (default: 1e-6)
    * timestep (units: time) - timestep for Langevin dyanmics (default: 2 fs)
    * nsteps_per_iteration (dimensionless) - number of timesteps per iteration (default: 500)
    * number_of_iterations (dimensionless) - number of replica-exchange iterations to simulate (default: 100)
    * number_of_equilibration_iterations (dimensionless) - number of equilibration iterations before begininng exhanges (default: 25)
    * equilibration_timestep (units: time) - timestep for use in equilibration (default: 2 fs)
    * verbose (boolean) - show information on run progress (default: False)
    * exchange_scheme (string) - scheme used to switch states: 'swap-all' or 'swap-neighbors' (default: 'swap-all')
    * log_weights (dimensionless numpy array) - vector of log weights for states
    
    TODO

    * Replace hard-coded Langevin dynamics with general MCMC moves.
    * Allow parallel resource to be used, if available (likely via MPI).
    * Add support for and autodetection of other NetCDF4 interfaces.
    * Add HDF5 support.
    * Allow user to specify a weight adaptation scheme that is a subclass of a base class we provide, so that the user can provide their own implementation.

    EXAMPLES

    Simulated tempering example (replica exchange among temperatures)
    (This is just an illustrative example; use SimulatedTempering class for actual production simulated tempering simulations.)
    
    >>> # Create test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create thermodynamic states for parallel tempering with exponentially-spaced schedule.
    >>> import simtk.unit as units
    >>> import math
    >>> nreplicas = 4 # number of temperature replicas
    >>> T_min = 298.0 * units.kelvin # minimum temperature
    >>> T_max = 600.0 * units.kelvin # maximum temperature
    >>> T_i = [ T_min + (T_max - T_min) * (math.exp(float(i) / float(nreplicas-1)) - 1.0) / (math.e - 1.0) for i in range(nreplicas) ]
    >>> from thermodynamics import ThermodynamicState
    >>> states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(nreplicas) ]
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # use a temporary file for testing -- you will want to keep this file, since it stores output and checkpoint data
    >>> simulation = ExpandedEnsemble(states, coordinates, file.name) # initialize the expanded-ensemble simulation
    >>> simulation.number_of_iterations = 10 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    >>> simulation.minimize = False
    >>> simulation.number_of_equilibration_iterations = 0
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """    

    def __init__(self, states, coordinates, store_filename, box_vectors=None, protocol=None, log_weights=None, weight_adaptation_scheme=None, weight_adaptation_protocol=None, mm=None):
        """
        Initialize expanded-ensemble simulation facility.

        ARGUMENTS
        
        states (list of ThermodynamicState) - Thermodynamic states to include in expanded ensemble.
           Each state must have a system with the same number of atoms, and the same
           thermodynamic ensemble (combination of temperature, pressure, pH, etc.) must
           be defined for each.
        coordinates (Coordinate object) - One set of initial coordinates
        store_filename (string) - Name of file to bind simulation to use as storage for checkpointing and storage of results.

        OPTIONAL ARGUMENTS

        log_weights (numpy array) - log_weights[i] is the log weight of thermodynamic state i (default: None, for all zeros)
        weight_adaptation_scheme (string) - specified weight adaptation scheme in [None, 'wang-landau', 'waste-recycling'] (default: None)
        weight_adaptation_protocol (dict) - additional parameters for weight adaptation scheme (default: None)
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict.
           Provided keywords will be matched to object variables to replace defaults.

        mm (implementation of simtk.openmm) - OpenMM API implementation to use

        """

        # Select default OpenMM implementation if not specified.
        self.mm = mm
        if mm is None: self.mm = simtk.openmm

        # Determine number of thermodynamic states from the number of specified states.
        self.nstates = len(states)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in states:
            if not state.is_compatible_with(states[0]):
                # TODO: Give more helpful warning as to what is wrong?
                raise ParameterError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        # Make a deep copy of states.
        # TODO: Change this to deep copy when System supports it.
        #self.states = copy.deepcopy(states)
        self.states = states

        # Record store file filename
        self.store_filename = store_filename

        # Store a deep copy of provided coordinates as a numpy array.
        self.coordinates = units.Quantity(numpy.array(coordinates / coordinates.unit, numpy.float32), coordinates.unit)

        # Store box vectors.
        # TODO: Modify this when we have defined an object that stores configuration and box vectors.
        self.box_vectors = box_vectors
        
        # Store log weights.
        self.log_weights = log_weights

        # Store weight adaptation scheme.
        self.weight_adaptation_scheme = weight_adaptation_scheme
        self.weight_adaptation_protocol = weight_adaptation_protocol

        # Set default options.
        # These can be changed externally until object is initialized.
        self.collision_rate = 91.0 / units.picosecond 
        self.constraint_tolerance = 1.0e-6 
        self.timestep = 2.0 * units.femtosecond
        self.nsteps_per_iteration = 500
        self.number_of_iterations = 1
        self.equilibration_timestep = 1.0 * units.femtosecond
        self.number_of_equilibration_iterations = 10
        self.title = 'Expanded-ensemble simulation created using ExpandedEnsemble class of expanded.py on %s' % time.asctime(time.localtime())        
        self.minimize = True 
        self.platform = None
        self.energy_platform = None        
        self.exchange_scheme = 'swap-all' # mix all replicas thoroughly

        # To allow for parameters to be modified after object creation, class is not initialized until a call to self._initialize().
        self._initialized = False

        # Set verbosity.
        self.verbose = False
        self.show_energies = True
        self.show_mixing_statistics = True
        
        # Handle provided 'protocol' dict, replacing any options provided by caller in dictionary.
        # TODO: Look for 'verbose' key first.
        if protocol is not None:
            for key in protocol.keys(): # for each provided key
                if key in vars(self).keys(): # if this is also a simulation parameter                    
                    value = protocol[key]
                    if self.verbose: print "from protocol: %s -> %s" % (key, str(value))
                    vars(self)[key] = value # replace default simulation parameter with provided parameter

        return

    def __repr__(self):
        """
        Return a 'formal' representation that can be used to reconstruct the class, if possible.

        """

        # TODO: Can we make this a more useful expression?
        return "<instance of ExpandedEnsemble>"

    def __str__(self):
        """
        Show an 'informal' human-readable representation of the replica-exchange simulation.

        """
        
        r =  ""
        r += "Expanded-ensemble simulation\n"
        r += "\n"
        r += "%d states\n" % str(self.nstates)
        r += "file store: %s\n" % str(self.store_filename)
        r += "initialized: %s\n" % str(self._initialized)
        r += "log weights: %s\n" % str(self.log_weights)
        r += "\n"
        r += "PARAMETERS\n"
        r += "collision rate: %s\n" % str(self.collision_rate)
        r += "relative constraint tolerance: %s\n" % str(self.constraint_tolerance)
        r += "timestep: %s\n" % str(self.timestep)
        r += "number of steps/iteration: %s\n" % str(self.nsteps_per_iteration)
        r += "number of iterations: %s\n" % str(self.number_of_iterations)
        r += "equilibration timestep: %s\n" % str(self.equilibration_timestep)
        r += "number of equilibration iterations: %s\n" % str(self.number_of_equilibration_iterations)
        r += "\n"

        return r

    def run(self):
        """
        Run the replica-exchange simulation.

        Any parameter changes (via object attributes) that were made between object creation and calling this method become locked in
        at this point, and the object will create and bind to the store file.  If the store file already exists, the run will be resumed
        if possible; otherwise, an exception will be raised.

        """

        # Make sure we've initialized everything and bound to a storage file before we begin execution.
        if not self._initialized:
            self._initialize()

        # Main loop
        while (self.iteration < self.number_of_iterations):
            if self.verbose: print "\nIteration %d / %d" % (self.iteration+1, self.number_of_iterations)

            # Adapt weights.
            self._adapt_weights()

            # Attempt state swaps.
            self._mix()

            # Propagate dynamics.
            self._propagate()

            # Compute energies in all states.
            self._compute_energies()

            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()

            # Write to storage file.
            self._write_iteration_netcdf()
            
            # Increment iteration counter.
            self.iteration += 1

            # Show mixing statistics.
            if self.verbose:
                self._show_mixing_statistics()


        # Clean up and close storage files.
        self._finalize()

        return

    def _initialize(self):
        """
        Initialize the simulation, and bind to a storage file.

        """
        if self._initialized:
            print "Simulation has already been initialized."
            raise Error

        # Select OpenMM Platform.
        if self.platform is None:
            # Find fastest platform.
            fastest_speed = 0.0
            fastest_platform = simtk.openmm.Platform.getPlatformByName("Reference")                        
            for platform_index in range(simtk.openmm.Platform.getNumPlatforms()):
                platform = simtk.openmm.Platform.getPlatform(platform_index)
                speed = platform.getSpeed()
                if (speed > fastest_speed):
                    fastest_speed = speed
                    fastest_platform = platform
            self.platform = fastest_platform

        if self.energy_platform is None:
            self.energy_platform = self.platform

        # Determine number of alchemical states.
        self.nstates = len(self.states)

        # Determine number of atoms in systems.
        self.natoms = self.states[0].system.getNumParticles()
  
        # Allocate storage.
        self.replica_state      = 0
        self.u_l                = numpy.zeros([self.nstates], numpy.float32)        
        self.swap_Pij_accepted  = numpy.zeros([self.nstates, self.nstates], numpy.float32)
        self.Nij_proposed       = numpy.zeros([self.nstates,self.nstates], numpy.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = numpy.zeros([self.nstates,self.nstates], numpy.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1

        if self.log_weights is None:
            self.log_weights = numpy.zeros([self.nstates], numpy.float64)

        # Get box size if not specified.
        if self.box_vectors is None:
            self.box_vectors = self.states[self.replica_state].system.getDefaultPeriodicBoxVectors()
        
        # Check if netcdf file extists.
        if os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0):
            # Resume from NetCDF file.
            self._resume_from_netcdf()

            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()            
        else:
            # Minimize and equilibrate all replicas.
            self._minimize_and_equilibrate()
            
            # Initialize current iteration counter.
            self.iteration = 0
            
            # TODO: Perform a sanity check on the consistency of forces for all replicas to make sure the GPU resources are working properly.
            
            # Compute energies of all alchemical replicas
            self._compute_energies()
            
            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()

            # Initialize NetCDF file.
            self._initialize_netcdf()

            # Store initial state.
            self._write_iteration_netcdf()
  
        # Signal that the class has been initialized.
        self._initialized = True

        return

    def _finalize(self):
        """
        Do anything necessary to clean up.

        """

        self.ncfile.close()

        return

    def _adapt_weights(self):
        """
        Adapt expanded ensemble log weights, if requested.

        """

        if self.weight_adaptation_scheme is None:
            return
        elif self.weight_adaptation_scheme == 'wang-landau':
            pass
        else:
            raise ParameterException("Weight adaptation scheme '%s' unknown.  Choose valid 'weight_adaptation_scheme' parameter." % self.weight_adaptation_scheme)
        
        return

    def _propagate(self):
        """
        Propagate replica dynamics.

        """

        start_time = time.time()

        # Propagate all replicas.
        if self.verbose: print "Propagating dynamics for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / units.picoseconds)
        #if self.verbose: print "replica %d / %d" % (replica_index, self.nstates)
        # Retrieve state.
        state_index = self.replica_state # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state
        # Create integrator and context.
        integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)
        context = self.mm.Context(state.system, integrator, self.platform)                        
        # Set coordinates.
        context.setPositions(self.coordinates)
        # Set box vectors.
        context.setPeriodicBoxVectors(*self.box_vectors)
        # Assign Maxwell-Boltzmann velocities.
        velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature)
        context.setVelocities(velocities)
        # Run dynamics.
        integrator.step(self.nsteps_per_iteration)
        # Store final coordinates
        openmm_state = context.getState(getPositions=True)
        self.coordinates = openmm_state.getPositions(asNumpy=True)
        # Store box vectors.
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)
        # Clean up.
        del context
        del integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_replica = elapsed_time / float(self.nstates)
        ns_per_day = self.timestep * self.nsteps_per_iteration / time_per_replica * 24*60*60 / units.nanoseconds
        if self.verbose: print "Time to propagate all replicas %.3f s (%.3f per replica, %.3f ns/day).\n" % (elapsed_time, time_per_replica, ns_per_day)

        return

    def _minimize_and_equilibrate(self):
        """
        Minimize and equilibrate all replicas.

        """

        # Minimize
        if self.minimize:
            if self.verbose: print "Minimizing..."
            state_index = self.replica_state
            state = self.states[state_index] # thermodynamic state
            # Create integrator and context.
            integrator = self.mm.VerletIntegrator(self.equilibration_timestep)
            context = self.mm.Context(state.system, integrator, self.platform)
            # Set coordinates.
            context.setPositions(self.coordinates)
            # Set box vectors.
            context.setPeriodicBoxVectors(*self.box_vectors)
            # Apply constraints.
            tolerance = integrator.getConstraintTolerance()
            context.applyConstraints(tolerance)
            # Minimize energy.
            minimized_coordinates = self.mm.LocalEnergyMinimizer.minimize(context)
            # Apply constraints.
            context.applyConstraints(tolerance)
            # Store final coordinates
            openmm_state = context.getState(getPositions=True)
            self.coordinates = openmm_state.getPositions(asNumpy=True)
            # Store final box vectors.
            self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)                
            # Clean up.
            del integrator, context

        # Equilibrate    
        for iteration in range(self.number_of_equilibration_iterations):
            if self.verbose: print "equilibration iteration %d / %d" % (iteration, self.number_of_equilibration_iterations)
            # Retrieve thermodynamic state.
            state_index = self.replica_state
            state = self.states[state_index] # thermodynamic state
            # Create integrator and context.
            integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.equilibration_timestep)
            context = self.mm.Context(state.system, integrator, self.platform)                        
            # Set coordinates.
            context.setPositions(self.coordinates)
            # Set box vectors.
            context.setPeriodicBoxVectors(*self.box_vectors)
            # Assign Maxwell-Boltzmann velocities.
            velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature)
            context.setVelocities(velocities)
            # Run dynamics.
            integrator.step(self.nsteps_per_iteration)
            # Store final coordinates
            openmm_state = context.getState(getPositions=True)
            self.coordinates = openmm_state.getPositions(asNumpy=True)
            # Store final box vectors.
            self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)                
            # Clean up.
            del context
            del integrator

        return

    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        TODO

        * We have to re-order Context initialization if we have variable box volume
        * Parallel implementation
        * We could speed this up if we did not create a new context for each state and replica.
        
        """

        start_time = time.time()
        
        if self.verbose: print "Computing energies..."
        for state_index in range(self.nstates):
            self.u_l[state_index] = self.states[state_index].reduced_potential(self.coordinates, box_vectors=self.box_vectors, platform=self.energy_platform)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy= elapsed_time / float(self.nstates)**2 
        if self.verbose: print "Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy)

        return

    def _log_sum(self, log_a_n):
        """
        Compute log(sum(exp(log_a_n))) in a numerically stable manner.

        ARGUMENTS

        log_a_n (numpy array) - logarithms of terms a_n

        RETURNS

        log_sum (float) - log(sum(exp(log_a_n)))

        """
        max_arg = log_a_n.max()
        log_sum = numpy.log(numpy.sum(numpy.exp(log_a_n - max_arg))) + max_arg
        return log_sum
        
    def _mix_all_swap(self):
        """
        Attempt Gibbs sampling update of state.

        """

        if self.verbose: print "Updating state using Gibbs sampling." 

        # Take note of initial state.
        istate = self.replica_state

        # Determine stationary probabilities of current configuration at each state.
        # TODO: We may need to set P_i[istate] = 0 for any states istate where energy is nan.
        log_P_i = - self.u_l[:] + self.log_weights[:]
        log_P_i = log_P_i - self._log_sum(log_P_i)
        P_i = numpy.exp(log_P_i)
        P_i[:] = P_i[:] / P_i.sum()

        # Sample from stationary probability.
        r = numpy.random.rand()
        jstate = 0
        while (r > P_i[jstate]):
            r -= P_i[jstate]
            jstate += 1

        # Assign new replica state.
        self.replica_state = jstate

        # Record swap statistics.
        # TODO: I'm not sure this is the right way to do accounting in Gibbs sampling.
        self.Nij_proposed[istate,:] = P_i
        self.Nij_proposed[:,istate] = P_i
        self.Nij_accepted[istate,jstate] += P_i[jstate]
        self.Nij_accepted[jstate,istate] += P_i[jstate]

        return

    def _mix_neighbor_swap(self):
        """
        Attempt exchanges between neighboring replicas only.

        """

        if self.verbose: print "Will attempt neighbor swaps."

        istate = self.replica_state
        r = numpy.random.rand()
        if (r < 0.5):
            jstate = istate - 1
        else:
            jstate = istate + 1 

        # Reject swap attempt if any energies are nan.
        log_P_accept = - self.u_l[jstate] + self.u_l[istate] + self.log_weights[jstate] - self.log_weights[istate]

        # Record that this move has been proposed.
        self.Nij_proposed[istate,jstate] += 1
        self.Nij_proposed[jstate,istate] += 1

        # Accept or reject.
        if (log_P_accept >= 0.0 or (numpy.random.rand() < math.exp(log_P_accept))):
            # Update state.
            self.replica_state = jstate
            
            # Accumulate statistics
            self.Nij_accepted[istate,jstate] += 1
            self.Nij_accepted[jstate,istate] += 1

        return

    def _mix(self):
        """
        Attempt to swap to different states according to user-specified scheme.
        
        """

        if self.verbose: print "Mixing states..."

        # Reset storage to keep track of swap attempts this iteration.
        self.Nij_proposed[:,:] = 0
        self.Nij_accepted[:,:] = 0

        # Perform swap attempts according to requested scheme.
        start_time = time.time()                    
        if self.exchange_scheme == 'swap-neighbors':
            self._mix_neighbor_swap() 
        elif self.exchange_scheme == 'swap-all':
            self._mix_all_swap()
        elif self.exchange_scheme == 'none':
            pass
        else:
            raise ParameterException("State exchange scheme '%s' unknown.  Choose valid 'exchange_scheme' parameter." % self.exchange_scheme)
        end_time = time.time()

        # Estimate cumulative transition probabilities between all states.        
        Nij_accepted = self.ncfile.variables['accepted'][:,:,:].sum(0) + self.Nij_accepted
        Nij_proposed = self.ncfile.variables['proposed'][:,:,:].sum(0) + self.Nij_proposed
        swap_Pij_accepted = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for istate in range(self.nstates):
            Ni = Nij_proposed[istate,:].sum()
            if (Ni == 0):
                swap_Pij_accepted[istate,istate] = 1.0
            else:
                swap_Pij_accepted[istate,istate] = 1.0 - float(Nij_accepted[istate,:].sum() - Nij_accepted[istate,istate]) / float(Ni)
                for jstate in range(self.nstates):
                    if istate != jstate:
                        swap_Pij_accepted[istate,jstate] = float(Nij_accepted[istate,jstate]) / float(Ni)

        # Report on mixing.
        if self.verbose:
            print "Mixing of states took %.3f s" % (end_time - start_time)
                
        return

    def _show_mixing_statistics(self):
        """
        Print summary of mixing statistics.

        """

        # Don't print anything until we've accumulated some statistics.
        if self.iteration < 2:
            return

        # Don't print anything if there is only one state.
        if (self.nstates < 2):
            return
        
        # Compute statistics of transitions.
        Nij = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for iteration in range(self.iteration - 1):
            istate = self.ncfile.variables['states'][iteration]
            jstate = self.ncfile.variables['states'][iteration+1]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
        Tij = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for istate in range(self.nstates):
            Tij[istate,istate] = 1.0
            if (Nij[istate,:].sum() > 0.0):
                Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

        if self.show_mixing_statistics:
            # Print observed transition probabilities.
            PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
            print "Cumulative symmetrized state mixing transition matrix:"
            print "%6s" % "",
            for jstate in range(self.nstates):
                print "%6d" % jstate,
            print ""
            for istate in range(self.nstates):
                print "%-6d" % istate,
                for jstate in range(self.nstates):
                    P = Tij[istate,jstate]
                    if (P >= PRINT_CUTOFF):
                        print "%6.3f" % P,
                    else:
                        print "%6s" % "",
                print ""

        # Estimate second eigenvalue and equilibration time.
        mu = numpy.linalg.eigvals(Tij)
        mu = -numpy.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            print "Perron eigenvalue is unity; Markov chain is decomposable."
        else:
            print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))

        return

    def _initialize_netcdf(self):
        """
        Initialize NetCDF file for storage.
        
        """    

        # Open NetCDF 4 file for writing.
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'w', version=2)
        ncfile = netcdf.Dataset(self.store_filename, 'w', version=2)        

        # Create dimensions.
        ncfile.createDimension('iteration', 0) # unlimited number of iterations
        ncfile.createDimension('state', self.nstates) # number of states
        ncfile.createDimension('atom', self.natoms) # number of atoms in system
        ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        setattr(ncfile, 'tile', self.title)
        setattr(ncfile, 'application', 'YANK')
        setattr(ncfile, 'program', 'yank.py')
        setattr(ncfile, 'programVersion', __version__)
        setattr(ncfile, 'Conventions', 'YANK')
        setattr(ncfile, 'ConventionVersion', '0.1')
        
        # Create variables.
        ncvar_positions = ncfile.createVariable('positions', 'f', ('iteration','atom','spatial'))
        ncvar_states    = ncfile.createVariable('states', 'i', ('iteration',))
        ncvar_energies  = ncfile.createVariable('energies', 'f', ('iteration','state'))        
        ncvar_proposed  = ncfile.createVariable('proposed', 'l', ('iteration','state','state')) 
        ncvar_accepted  = ncfile.createVariable('accepted', 'l', ('iteration','state','state'))
        ncvar_log_weights  = ncfile.createVariable('log_weights', 'f', ('iteration','state'))
        ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f', ('iteration','spatial','spatial'))        
        ncvar_volumes  = ncfile.createVariable('volumes', 'f', ('iteration',))
        
        # Define units for variables.
        setattr(ncvar_positions, 'units', 'nm')
        setattr(ncvar_states,    'units', 'none')
        setattr(ncvar_energies,  'units', 'kT')
        setattr(ncvar_proposed,  'units', 'none')
        setattr(ncvar_accepted,  'units', 'none')
        setattr(ncvar_log_weights,  'units', 'none')                        
        setattr(ncvar_box_vectors, 'units', 'nm')
        setattr(ncvar_volumes, 'units', 'nm**3')
        
        # Define long (human-readable) names for variables.
        setattr(ncvar_positions, "long_name", "positions[iteration][atom][spatial] is position of coordinate 'spatial' of atom 'atom' for iteration 'iteration'.")
        setattr(ncvar_states,    "long_name", "states[iteration] is the state index (0..nstates-1) of iteration 'iteration'.")
        setattr(ncvar_energies,  "long_name", "energies[iteration][state] is the reduced (unitless) energy from iteration 'iteration' evaluated at state 'state'.")
        setattr(ncvar_proposed,  "long_name", "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_accepted,  "long_name", "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_log_weights,  "long_name", "log_weights[iteration][state] is the log weight used for state 'state' during iteration 'iteration'.")
        setattr(ncvar_box_vectors, "long_name", "box_vectors[iteration][i][j] is dimension j of box vector i from iteration 'iteration-1'.")
        setattr(ncvar_volumes, "long_name", "volume[iteration] is the box volume from iteration 'iteration-1'.")
        
        # Force sync to disk to avoid data loss.
        ncfile.sync()

        # Store netcdf file handle.
        self.ncfile = ncfile
        
        return
    
    def _write_iteration_netcdf(self):
        """
        Write positions, states, and energies of current iteration to NetCDF file.
        
        """

        # Store replica positions.
        self.ncfile.variables['positions'][self.iteration,:,:] = self.coordinates[:,:] / units.nanometers
            
        # Store box vectors and volume.
        for i in range(3):
            self.ncfile.variables['box_vectors'][self.iteration,i,:] = (self.box_vectors[i] / units.nanometers)
        volume = self.states[self.replica_state]._volume(self.box_vectors)
        self.ncfile.variables['volumes'][self.iteration] = volume / (units.nanometers**3)

        # Store state information.
        self.ncfile.variables['states'][self.iteration] = self.replica_state

        # Store energies.
        self.ncfile.variables['energies'][self.iteration,:] = self.u_l[:]

        # Store log weights.
        self.ncfile.variables['log_weights'][self.iteration,:] = self.log_weights[:]

        # Store mixing statistics.
        # TODO: Write mixing statistics for this iteration?
        self.ncfile.variables['proposed'][self.iteration,:,:] = self.Nij_proposed[:,:]
        self.ncfile.variables['accepted'][self.iteration,:,:] = self.Nij_accepted[:,:]        

        # Force sync to disk to avoid data loss.
        self.ncfile.sync()

        return

    def _resume_from_netcdf(self):
        """
        Resume execution by reading current positions and energies from a NetCDF file.
        
        """
        
        debug = True

        # Open NetCDF file for reading
        if debug: print "Reading NetCDF file '%s'..." % self.store_filename
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'r') # Scientific.IO.NetCDF
        ncfile = netcdf.Dataset(self.store_filename, 'r') # netCDF4
        
        # TODO: Perform sanity check on file before resuming

        # Get current dimensions.
        self.iteration = ncfile.variables['energies'].shape[0] - 1 
        self.nstates = ncfile.variables['energies'].shape[1]
        self.natoms = ncfile.variables['positions'].shape[1]
        if debug: print "iteration = %d, nstates = %d, natoms = %d" % (self.iteration, self.nstates, self.natoms)

        # Restore log weights.
        self.log_weights = ncfile.variables['log_weights'][self.iteration,:].astype(numpy.float64).copy()
        if debug: print self.log_weights

        # Restore positions.
        x = ncfile.variables['positions'][self.iteration,:,:].astype(numpy.float64).copy()
        self.coordinates = units.Quantity(x, units.nanometers)
        if debug: print self.coordinates
        
        # Restore box vectors.
        x = ncfile.variables['box_vectors'][self.iteration,replica_index,:,:].astype(numpy.float64).copy()
        x = units.Quantity(x, units.nanometers)
        self.box_vectors = (x[0,:], x[1,:], x[2,:])
        if debug: print self.box_vectors

        # Restore state information.
        self.replica_state = ncfile.variables['states'][self.iteration]
        if debug: print self.replica_state

        # Restore energies.
        self.u_l = ncfile.variables['energies'][self.iteration,:].copy()
        if debug: print self.u_l
        
        # Close NetCDF file.
        ncfile.close()        

        # We will work on the next iteration.
        self.iteration += 1
        
        # Reopen NetCDF file for appending, and maintain handle.
        #self.ncfile = netcdf.NetCDFFile(self.store_filename, 'a')
        self.ncfile = netcdf.Dataset(self.store_filename, 'a')        
        
        return

    def _show_energies(self):
        """
        Show energies and log weights (in units of kT) for all replicas at all states.

        """

        # print header
        print "%-24s %16s" % ("reduced potential (kT)", "current state"),
        for state_index in range(self.nstates):
            print " state %3d" % state_index,
        print ""

        # print energies in kT
        print "replica %16s %16d" % ("", self.replica_state),
        for state_index in range(self.nstates):
            print "%10.1f" % (self.u_l[state_index]),
        print ""

        # print log weights
        print "log weights %12s %16s" % ("", ""),        
        for state_index in range(self.nstates):
            print "%10.1f" % (self.log_weights[state_index]),
        print ""

        return

    def _assign_Maxwell_Boltzmann_velocities(self, system, temperature):
        """
        Generate Maxwell-Boltzmann velocities.

        ARGUMENTS

        system (simtk.openmm.System) - the system for which velocities are to be assigned
        temperature (simtk.unit.Quantity with units temperature) - the temperature at which velocities are to be assigned

        RETURN VALUES

        velocities (simtk.unit.Quantity wrapping numpy array of dimension natoms x 3 with units of distance/time) - drawn from the Maxwell-Boltzmann distribution at the appropriate temperature

        TODO

        This could be sped up by introducing vector operations.

        """

        # Get number of atoms
        natoms = system.getNumParticles()

        # Create storage for velocities.        
        velocities = units.Quantity(numpy.zeros([natoms, 3], numpy.float32), units.nanometer / units.picosecond) # velocities[i,k] is the kth component of the velocity of atom i
  
        # Compute thermal energy and inverse temperature from specified temperature.
        kT = kB * temperature # thermal energy
        beta = 1.0 / kT # inverse temperature
  
        # Assign velocities from the Maxwell-Boltzmann distribution.
        for atom_index in range(natoms):
            mass = system.getParticleMass(atom_index) # atomic mass
            sigma = units.sqrt(kT / mass) # standard deviation of velocity distribution for each coordinate for this atom
            for k in range(3):
                velocities[atom_index,k] = sigma * numpy.random.normal()

        # Return velocities
        return velocities

#=============================================================================================
# Simulated tempering
#=============================================================================================

class SimulatedTempering(ExpandedEnsemble):
    """
    Simulated tempering simulation facility.

    DESCRIPTION

    This class provides a facility for simulated tempering simulations.  It is a subclass of ExpandedEnsemble, but provides
    various convenience methods and efficiency improvements for simulated tempering simulations, so should be preferred for
    this type of simulation.  In particular, the System only need be specified once, while the temperatures (or a temperature
    range) is used to automatically build a set of ThermodynamicState objects for expanded-ensemble simulation.  Efficiency
    improvements make use of the fact that the reduced potentials are linear in inverse temperature.
    
    EXAMPLES

    Simulated tempering of alanine dipeptide in implicit solvent.
    
    >>> # Create alanine dipeptide test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * units.kelvin
    >>> Tmax = 600.0 * units.kelvin
    >>> nreplicas = 4
    >>> simulation = SimulatedTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    def __init__(self, system, coordinates, store_filename, box_vectors=None, protocol=None, Tmin=None, Tmax=None, ntemps=None, temperatures=None, mm=None):
        """
        Initialize a simulated tempering simulation object.

        ARGUMENTS
        
        system (simtk.openmm.System) - the system to simulate
        coordinates (simtk.unit.Quantity of numpy natoms x 3 array of units length) - coordinates to initiate simulation from
        store_filename (string) -  name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        box_vectors - box vectors to initiate simulation from (default: None)
        Tmin, Tmax, ntemps - min and max temperatures, and number of temperatures for exponentially-spaced temperature selection (default: None)
        temperatures (list of simtk.unit.Quantity with units of temperature) - if specified, this list of temperatures will be used instead of (Tmin, Tmax, ntemps) (default: None)
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict.  Provided keywords will be matched to object variables to replace defaults. (default: None)

        NOTES

        Either (Tmin, Tmax, ntempts) must all be specified or the list of 'temperatures' must be specified.

        """
        # Create thermodynamic states from temperatures.
        if temperatures is not None:
            print "Using provided temperatures"
            self.temperatures = temperatures
        elif (Tmin is not None) and (Tmax is not None) and (ntemps is not None):
            self.temperatures = [ Tmin + (Tmax - Tmin) * (math.exp(float(i) / float(ntemps-1)) - 1.0) / (math.e - 1.0) for i in range(ntemps) ]
        else:
            raise ValueError("Either 'temperatures' or 'Tmin', 'Tmax', and 'ntemps' must be provided.")

        states = [ ThermodynamicState(system=system, temperature=self.temperatures[i]) for i in range(ntemps) ]

        # Initialize replica-exchange simlulation.
        ExpandedEnsemble.__init__(self, states, coordinates, store_filename, box_vectors=box_vectors, protocol=protocol, mm=mm)

        # Override title.
        self.title = 'Parallel tempering simulation created using ParallelTempering class of repex.py on %s' % time.asctime(time.localtime())
        
        return

    def _compute_energies(self):
        """
        Compute reduced potentials of all replicas at all states (temperatures).

        NOTES

        Because only the temperatures differ among replicas, we replace the generic O(N^2) replica-exchange implementation with an O(N) implementation.

        TODO

        * See if we can store the potential energy and avoid ever having to recompute potential (even just once) for this call.
        
        """

        start_time = time.time()
        if self.verbose: print "Computing energies..."

        # Create an integrator and context.
        state = self.states[0]
        integrator = self.mm.VerletIntegrator(self.timestep)
        context = self.mm.Context(state.system, integrator, self.platform)
        
        # Set coordinates.
        context.setPositions(self.coordinates)

        # Compute potential energy.
        openmm_state = context.getState(getEnergy=True)            
        potential_energy = openmm_state.getPotentialEnergy()           

        # Compute energies at this state for all replicas.
        for state_index in range(self.nstates):
            # Compute reduced potential
            beta = 1.0 / (kB * self.states[state_index].temperature)
            self.u_l[state_index] = beta * potential_energy

        # Clean up.
        del context
        del integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.verbose: print "Time to compute all energies %.3f s.\n" % (elapsed_time)

        return

#=============================================================================================
# Simulated scaling
#=============================================================================================

class SimulatedScaling(ExpandedEnsemble):
    """
    Simulated scaling simulation facility.

    DESCRIPTION

    This class provides an implementation of a simulated scaling simulation based on the ExpandedEnsemble facility.
    It provides several convenience classes and efficiency improvements, and should be preferentially used for simulated
    scaling simulations over ExpandedEnsemble whenever possible.
    
    EXAMPLES

    Simulated scaling example where all thermodynamic states have identical potential energy functions.
    
    >>> # Create reference system
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Copy reference system.
    >>> systems = [reference_system for index in range(10)]
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Create reference state.
    >>> from thermodynamics import ThermodynamicState            
    >>> reference_state = ThermodynamicState(reference_system, temperature=298.0*units.kelvin)
    >>> simulation = SimulatedScaling(reference_state, systems, coordinates, store_filename)
    >>> simulation.number_of_iterations = 10 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    >>> simulation.minimize = False
    >>> simulation.number_of_equilibration_iterations = 0
    >>> # Run simulation.
    >>> simulation.run() # run the simulation
    
    """

    def __init__(self, reference_state, systems, coordinates, store_filename, box_vectors=None, protocol=None, mm=None):
        """
        Initialize a simulated scaling simulation object.

        ARGUMENTS

        reference_state (ThermodynamicState) - reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems (list of simtk.openmm.System) - list of systems to simulate (one per replica)
        coordinates (simtk.unit.Quantity of numpy natoms x 3 with units length) - coordinates to simulate from
        store_filename (string) - name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.

        """
        # Create thermodynamic states from systems.        
        states = list()
        for system in systems:
            #state = copy.deepcopy(reference_state) # TODO: Use deep copy once this works
            state = reference_state
            #state.system = copy.deepcopy(system) # TODO: Use deep copy once this works
            state.system = system
            states.append(state)

        # Initialize replica-exchange simlulation.
        ExpandedEnsemble.__init__(self, states, coordinates, store_filename, box_vectors, protocol=protocol, mm=mm)

        # Override title.
        self.title = 'Simulated scaling simulation created using ExpandedEnsemble class of repex.py on %s' % time.asctime(time.localtime())
        
        return

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

