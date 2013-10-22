#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Replica-exchange simulation algorithms and specific variants.

DESCRIPTION

This module provides a general facility for running replica-exchange simulations, as well as
derived classes for special cases such as parallel tempering (in which the states differ only
in temperature) and Hamiltonian exchange (in which the state differ only by potential function).

Provided classes include:

* ReplicaExchange - Base class for general replica-exchange simulations among specified ThermodynamicState objects
* ParallelTempering - Convenience subclass of ReplicaExchange for parallel tempering simulations (one System object, many temperatures/pressures)
* HamiltonianExchange - Convenience subclass of ReplicaExchange for Hamiltonian exchange simulations (many System objects, same temperature/pressure)

DEPENDENCIES

Use of this module requires the following

* Python 2.6 or later

  http://www.python.org

* OpenMM with Python wrappers

  http://simtk.org/home/openmm

* PyOpenMM Python tools for OpenMM

  http://simtk.org/home/openmm

* NetCDF (compiled with netcdf4 support) and HDF5 (on which 

  http://www.unidata.ucar.edu/software/netcdf/
  http://www.hdfgroup.org/HDF5/

* netcdf4-python (a Python interface for netcdf4)

  http://code.google.com/p/netcdf4-python/

* numpy and scipy

  http://www.scipy.org/

* mpi4py (if MPI support is desired)

http://mpi4py.scipy.org/

Note that this must be compiled against the appropriate MPI implementation.

NOTES

* MPI is now supported through mpi4py.

TODO

* Add analysis facility accessible by user.
* Give up on Context caching and revert to serial Context creation/destruction if we run out of GPU memory (issuing an alert).
* Store replica self-energies and/or -ln q(x) for simulation (for analyzing correlation times).
* Add analysis facility.
* Allow user to call initialize() externally and get the NetCDF file handle to add additional data?
* Store / restore parameters and System objects from NetCDF file for resuming and later analysis.
* Sampling support:
  * Short-term: Add support for user to specify a callback to create the Integrator to use ('integrator_factory' or 'integrator_callback').
  * Longer-term: Allow a more complex MCMC sampling scheme consisting of one or more moves to be specified through mcmc.py facility.
* Allow different choices of temperature handling during exchange attempts: 
  * scale velocities (exchanging only on potential energies) - make this the default?
  * randomize velocities (exchanging only on potential energies)
  * exchange on total energies, preserving velocities (requires more replicas)
* Add control over number of times swaps are attempted when mixing replicas, or compute best guess automatically
* Add support for online analysis (potentially by separate threads or while GPU is running?)
* Add another layer of abstraction so that the base class uses generic log probabilities, rather than reduced potentials?
* Add support for HDF5 storage models.
* Support for other NetCDF interfaces via autodetection/fallback:
  * scipy.io.netcdf 
* Use interface-based checking of arguments so that different implementations of the OpenMM API (such as pyopenmm) can be used.
* Eliminate file closures in favor of syncs to avoid closing temporary files in the middle of a run.

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
# Replica-exchange simulation
#=============================================================================================

class ReplicaExchange(object):
    """
    Replica-exchange simulation facility.

    This base class provides a general replica-exchange simulation facility, allowing any set of thermodynamic states
    to be specified, along with a set of initial coordinates to be assigned to the replicas in a round-robin fashion.
    No distinction is made between one-dimensional and multidimensional replica layout; by default, the replica mixing
    scheme attempts to mix *all* replicas to minimize slow diffusion normally found in multidimensional replica exchange
    simulations.  (Modification of the 'replica_mixing_scheme' setting will allow the tranditional 'neighbor swaps only'
    scheme to be used.)

    While this base class is fully functional, it does not make use of the special structure of parallel tempering or
    Hamiltonian exchange variants of replica exchange.  The ParallelTempering and HamiltonianExchange classes should
    therefore be used for these algorithms, since they are more efficient and provide more convenient ways to initialize
    the simulation classes.

    Stored configurations, energies, swaps, and restart information are all written to a single output file using
    the platform portable, robust, and efficient NetCDF4 library.  Plans for future HDF5 support are pending.    
    
    ATTRIBUTES

    The following parameters (attributes) can be set after the object has been created, but before it has been
    initialized by a call to run():

    * collision_rate (units: 1/time) - the collision rate used for Langevin dynamics (default: 90 ps^-1)
    * constraint_tolerance (dimensionless) - relative constraint tolerance (default: 1e-6)
    * timestep (units: time) - timestep for Langevin dyanmics (default: 2 fs)
    * nsteps_per_iteration (dimensionless) - number of timesteps per iteration (default: 500)
    * number_of_iterations (dimensionless) - number of replica-exchange iterations to simulate (default: 100)
    * number_of_equilibration_iterations (dimensionless) - number of equilibration iterations before beginning exchanges (default: 0)
    * equilibration_timestep (units: time) - timestep for use in equilibration (default: 2 fs)
    * verbose (boolean) - show information on run progress (default: False)
    * replica_mixing_scheme (string) - scheme used to swap replicas: 'swap-all' or 'swap-neighbors' (default: 'swap-all')
    * online_analysis (boolean) - if True, analysis will occur each iteration (default: False)
    
    TODO

    * Improve speed of Maxwell-Boltzmann velocity assignment.
    * Replace hard-coded Langevin dynamics with general MCMC moves.
    * Allow parallel resource to be used, if available (likely via Parallel Python).
    * Add support for and autodetection of other NetCDF4 interfaces.
    * Add HDF5 support.

    EXAMPLES

    Parallel tempering simulation of alanine dipeptide in implicit solvent (replica exchange among temperatures)
    (This is just an illustrative example; use ParallelTempering class for actual production parallel tempering simulations.)
    
    >>> # Create test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create thermodynamic states for parallel tempering with exponentially-spaced schedule.
    >>> import simtk.unit as units
    >>> import math
    >>> nreplicas = 3 # number of temperature replicas
    >>> T_min = 298.0 * units.kelvin # minimum temperature
    >>> T_max = 600.0 * units.kelvin # maximum temperature
    >>> T_i = [ T_min + (T_max - T_min) * (math.exp(float(i) / float(nreplicas-1)) - 1.0) / (math.e - 1.0) for i in range(nreplicas) ]
    >>> from thermodynamics import ThermodynamicState
    >>> states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(nreplicas) ]
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # use a temporary file for testing -- you will want to keep this file, since it stores output and checkpoint data
    >>> # Create simulation.
    >>> simulation = ReplicaExchange(states, coordinates, file.name) # initialize the replica-exchange simulation
    >>> simulation.minimize = False
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.run() # run the simulation

    """    

    # Options to store.
    options_to_store = ['collision_rate', 'constraint_tolerance', 'timestep', 'nsteps_per_iteration', 'number_of_iterations', 'equilibration_timestep', 'number_of_equilibration_iterations', 'title', 'minimize', 'replica_mixing_scheme', 'online_analysis', 'verbose', 'show_mixing_statistics']

    def __init__(self, states=None, coordinates=None, store_filename=None, protocol=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize replica-exchange simulation facility.

        ARGUMENTS
        
        states (list of ThermodynamicState) - Thermodynamic states to simulate, where one replica is allocated per state.
           Each state must have a system with the same number of atoms, and the same
           thermodynamic ensemble (combination of temperature, pressure, pH, etc.) must
           be defined for each.
        coordinates (Coordinate object or iterable container of Coordinate objects) - One or more sets of initial coordinates
           to be initially assigned to replicas in a round-robin fashion, provided simulation is not resumed from store file.
           Currently, coordinates must be specified as a list of simtk.unit.Quantity-wrapped numpy arrays.
        store_filename (string) - Name of file to bind simulation to use as storage for checkpointing and storage of results.

        OPTIONAL ARGUMENTS
        
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.
        mm (implementation of simtk.openmm) - OpenMM API implementation to use (default: simtk.openmm)
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)
        metadata (dict) - metadata to store in a 'metadata' group in store file

        NOTES
        
        If store_filename exists, the simulation will try to resume from this file.
        If thermodynamic state information can be found in the store file, this will be read, and 'states' will be ignored.
        If the store file exists, coordinates will be read from this file, and 'coordinates' will be ignored.        
        The provided 'protocol' will override any options restored from the store file.

        """

        # To allow for parameters to be modified after object creation, class is not initialized until a call to self._initialize().
        self._initialized = False

        # Select default OpenMM implementation if not specified.
        self.mm = mm
        if mm is None: self.mm = simtk.openmm

        # Set MPI communicator (or None if not used).
        self.mpicomm = mpicomm
        
        # Set default options.
        # These can be changed externally until object is initialized.
        self.collision_rate = 91.0 / units.picosecond 
        self.constraint_tolerance = 1.0e-6 
        self.timestep = 2.0 * units.femtosecond
        self.nsteps_per_iteration = 500
        self.number_of_iterations = 1
        self.equilibration_timestep = 1.0 * units.femtosecond
        self.number_of_equilibration_iterations = 1
        self.title = 'Replica-exchange simulation created using ReplicaExchange class of repex.py on %s' % time.asctime(time.localtime())        
        self.minimize = True 
        self.minimize_tolerance = 1.0 * units.kilojoules_per_mole / units.nanometers # if specified, set minimization tolerance
        self.minimize_maxIterations = 0 # if nonzero, set maximum iterations
        self.platform = None
        self.replica_mixing_scheme = 'swap-all' # mix all replicas thoroughly
        self.online_analysis = False # if True, analysis will occur each iteration
        self.show_energies = True
        self.show_mixing_statistics = True

        # Set verbosity.
        self.verbose = False
        if protocol and ('verbose' in protocol): self.verbose = protocol['verbose']
        # TODO: Find a better solution to setting verbosity.
        self.verbose_root = self.verbose # True only if self.verbose is True and we are root node (if MPI)
        if self.mpicomm and self.mpicomm.rank != 0: self.verbose_root = False
        
        # Record store file filename
        self.store_filename = store_filename

        # Check if netcdf file exists.
        self.resume = os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0)
        if self.mpicomm: self.resume = self.mpicomm.bcast(self.resume, root=0) # use whatever root node decides

        # Try to restore thermodynamic states and run options from the NetCDF file.
        states_restored = False 
        options_restored = False
        if self.resume: 
            print "Attempting to resume by reading thermodynamic states and options..." 
            ncfile = netcdf.Dataset(self.store_filename, 'r')            
            states_restored = self._restore_thermodynamic_states(ncfile)
            options_restored = self._restore_options(ncfile)
            ncfile.close()

        # Make a deep copy of specified states if we cannot restore them from file.
        if (not states_restored) and (states is not None):            
            # TODO: Debug why deepcopy isn't working.
            #self.states = copy.deepcopy(states)
            self.states = states
            
        if (not states_restored) and (states is None):
            raise ParameterError("Could not restore thermodynamic states from store file, and no states specified.")

        # Determine number of replicas from the number of specified thermodynamic states.
        self.nreplicas = len(self.states)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.states:
            if not state.is_compatible_with(self.states[0]):
                raise ParameterError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        if not self.resume:
            # Distribute coordinate information to replicas in a round-robin fashion.
            # We have to explicitly check to see if z is a list or a set here because it turns out that numpy 2D arrays are iterable as well.
            # TODO: Handle case where coordinates are passed in as a list of tuples, or list of lists, or list of Vec3s, etc.
            if type(coordinates) in [type(list()), type(set())]:
                self.provided_coordinates = [ units.Quantity(numpy.array(coordinate_set / coordinate_set.unit), coordinate_set.unit) for coordinate_set in coordinates ] 
            else:
                self.provided_coordinates = [ units.Quantity(numpy.array(coordinates / coordinates.unit), coordinates.unit) ]            
                    
        # Handle provided 'protocol' dict, replacing any options provided by caller in dictionary.
        # TODO: Look for 'verbose' key first.
        if protocol is not None:
            for key in protocol.keys(): # for each provided key
                if key in vars(self).keys(): # if this is also a simulation parameter                    
                    value = protocol[key]
                    if self.verbose: print "from protocol: %s -> %s" % (key, str(value))
                    vars(self)[key] = value # replace default simulation parameter with provided parameter

        # Store metadata to store in store file.
        self.metadata = metadata

        return

    def __repr__(self):
        """
        Return a 'formal' representation that can be used to reconstruct the class, if possible.

        """

        # TODO: Can we make this a more useful expression?
        return "<instance of ReplicaExchange>"

    def __str__(self):
        """
        Show an 'informal' human-readable representation of the replica-exchange simulation.

        """
        
        r =  ""
        r += "Replica-exchange simulation\n"
        r += "\n"
        r += "%d replicas\n" % str(self.nreplicas)
        r += "%d coordinate sets provided\n" % len(self.provided_coordinates)
        r += "file store: %s\n" % str(self.store_filename)
        r += "initialized: %s\n" % str(self._initialized)
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
        run_start_time = time.time()              
        run_start_iteration = self.iteration
        while (self.iteration < self.number_of_iterations):
            if self.verbose: print "\nIteration %d / %d" % (self.iteration+1, self.number_of_iterations)
            initial_time = time.time()

            # Attempt replica swaps to sample from equilibrium permuation of states associated with replicas.
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states.
            self._compute_energies()

            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()

            # Analysis.
            if self.online_analysis:
                self._analysis()

            # Write to storage file.
            self._write_iteration_netcdf()
            
            # Increment iteration counter.
            self.iteration += 1

            # Show mixing statistics.
            if self.verbose and self.show_mixing_statistics:
                self._show_mixing_statistics()

            # Show timing statistics.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.number_of_iterations - self.iteration)
            estimated_total_time = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.number_of_iterations)
            estimated_finish_time = final_time + estimated_time_remaining
            if self.verbose: 
                print "Iteration took %.3f s." % elapsed_time
                print "Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time)))
            
            # Perform sanity checks to see if we should terminate here.
            self._run_sanity_checks()

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

        # TODO: If no platform is specified, get the default platform.
        if self.platform is None:
            print "No platform is specified, using default."
            print "Not implemented."
            return Error

        # Turn off verbosity if not master node.
        if self.mpicomm:
            # Have each node report that it is initialized.
            if self.verbose:
                print "Initialized node %d / %d" % (self.mpicomm.rank, self.mpicomm.size)
            # Turn off verbosity for all nodes but root.
            if self.mpicomm.rank != 0:
                self.verbose = False
  
        # Display papers to be cited.
        if self.verbose:
            self._display_citations()

        # Determine number of alchemical states.
        self.nstates = len(self.states)

        # Create cached Context objects.
        if self.verbose: print "Creating and caching Context objects..."
        MAX_SEED = (1<<31) - 1 # maximum seed value (max size of signed C long)
        seed = int(numpy.random.randint(MAX_SEED)) # TODO: Is this the right maximum value to use?
        if self.mpicomm:
            # Compile all kernels on master process to avoid nvcc deadlocks.
            # TODO: Fix this when nvcc fix comes out.
            initial_time = time.time()
            if self.mpicomm.rank == 0:
                for state_index in range(self.nstates):
                    print "Master node compiling kernels for state %d / %d..." % (state_index, self.nstates) # DEBUG
                    state = self.states[state_index]
                    integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)                    
                    integrator.setRandomNumberSeed(seed + self.mpicomm.rank) 
                    context = self.mm.Context(state.system, integrator, self.platform)
                    self.mm.LocalEnergyMinimizer.minimize(context, 0, 1) # DEBUG
                    del context, integrator
            self.mpicomm.barrier()
            final_time = time.time()
            elapsed_time = final_time - initial_time
            print "Barrier complete.  Compiling kernels took %.1f s." % elapsed_time # DEBUG

            # Create cached contexts for only the states this process will handle.
            initial_time = time.time()
            for state_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                print "Node %d / %d creating Context for state %d..." % (self.mpicomm.rank, self.mpicomm.size, state_index) # DEBUG
                state = self.states[state_index]
                try:
                    state._integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)                    
                    state._integrator.setRandomNumberSeed(seed + self.mpicomm.rank) 
                    initial_context_time = time.time() # DEBUG
                    if self.platform:
                        print "Node %d / %d: Using platform %s" % (self.mpicomm.rank, self.mpicomm.size, self.platform.getName())
                        state._context = self.mm.Context(state.system, state._integrator, self.platform)
                    else:
                        print "Node %d / %d: No platform specified." % (self.mpicomm.rank, self.mpicomm.size)
                        state._context = self.mm.Context(state.system, state._integrator)                    
                    print "Node %d / %d: Context creation took %.3f s" % (self.mpicomm.rank, self.mpicomm.size, time.time() - initial_context_time) # DEBUG
                except Exception as e:
                    print e
            print "Note %d / %d: Context creation done.  Waiting for MPI barrier..." % (self.mpicomm.rank, self.mpicomm.size) # DEBUG
            self.mpicomm.barrier()
            print "Barrier complete." # DEBUG
        else:
            # Serial version.
            initial_time = time.time()
            for (state_index, state) in enumerate(self.states):  
                if self.verbose: print "Creating Context for state %d..." % state_index
                state._integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)
                state._integrator.setRandomNumberSeed(seed)
                initial_context_time = time.time() # DEBUG
                if self.platform:
                    state._context = self.mm.Context(state.system, state._integrator, self.platform)
                else:
                    state._context = self.mm.Context(state.system, state._integrator)
                print "Context creation took %.3f s" % (time.time() - initial_context_time) # DEBUG            
        final_time = time.time()
        elapsed_time = final_time - initial_time
        if self.verbose: print "%.3f s elapsed." % elapsed_time

        # Determine number of atoms in systems.
        self.natoms = self.states[0].system.getNumParticles()
  
        # Allocate storage.
        self.replica_coordinates = list() # replica_coordinates[i] is the configuration currently held in replica i
        self.replica_box_vectors = list() # replica_box_vectors[i] is the set of box vectors currently held in replica i  
        self.replica_states     = numpy.zeros([self.nstates], numpy.int32) # replica_states[i] is the state that replica i is currently at
        self.u_kl               = numpy.zeros([self.nstates, self.nstates], numpy.float32)        
        self.swap_Pij_accepted  = numpy.zeros([self.nstates, self.nstates], numpy.float32)
        self.Nij_proposed       = numpy.zeros([self.nstates,self.nstates], numpy.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = numpy.zeros([self.nstates,self.nstates], numpy.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1

        # Distribute coordinate information to replicas in a round-robin fashion, making a deep copy.
        if not self.resume:
            self.replica_coordinates = [ copy.deepcopy(self.provided_coordinates[replica_index % len(self.provided_coordinates)]) for replica_index in range(self.nstates) ]

        # Assign default box vectors.
        self.replica_box_vectors = list()
        for state in self.states:
            [a,b,c] = state.system.getDefaultPeriodicBoxVectors()
            box_vectors = units.Quantity(numpy.zeros([3,3], numpy.float32), units.nanometers)
            box_vectors[0,:] = a
            box_vectors[1,:] = b
            box_vectors[2,:] = c
            self.replica_box_vectors.append(box_vectors)
        
        # Assign initial replica states.
        for replica_index in range(self.nstates):
            self.replica_states[replica_index] = replica_index

        if self.resume:
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
            
            # TODO: Perform any GPU sanity checks here.
            
            # Compute energies of all alchemical replicas
            self._compute_energies()
            
            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()

            # Initialize NetCDF file.
            self._initialize_netcdf()

            # Store initial state.
            self._write_iteration_netcdf()

        # Analysis objet starts off empty.
        # TODO: Use an empty dict instead?
        self.analysis = None

        # Signal that the class has been initialized.
        self._initialized = True

        return

    def _finalize(self):
        """
        Do anything necessary to finish run except close files.

        """

        if self.mpicomm:
            # Only the root node needs to clean up.
            if self.mpicomm.rank != 0: return
        
        if hasattr(self, 'ncfile') and self.ncfile:
            self.ncfile.sync()

        return

    def __del__(self):
        """
        Clean up, closing files.

        """
        self._finalize()

        if self.mpicomm:
            # Only the root node needs to clean up.
            if self.mpicomm.rank != 0: return

        if hasattr(self, 'ncfile') and self.ncfile:
            self.ncfile.close()

        return

    def _display_citations(self):
        """
        Display papers to be cited.

        TODO:

        * Add original citations for various replica-exchange schemes.
        * Show subset of OpenMM citations based on what features are being used.

        """
        
        openmm_citations = """\
        Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing units. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
        Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
        Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
        Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w"""
        
        gibbs_citations = """\
        Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing. J. Chem. Phys., in press. arXiv: 1105.5749"""

        mbar_citations = """\
        Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105, 2008. DOI: 10.1063/1.2978177"""

        print "Please cite the following:"
        print ""
        print openmm_citations
        if self.replica_mixing_scheme == 'swap-all':
            print gibbs_citations
        if self.online_analysis:
            print mbar_citations

        return

    def _propagate_replica(self, replica_index):
        """
        Propagate the replica corresponding to the specified replica index.
        Caching is used.

        ARGUMENTS

        replica_index (int) - the replica to propagate

        RETURNS

        elapsed_time (float) - time (in seconds) to propagate replica

        """

        start_time = time.time()

        # Retrieve state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state

        # Retrieve integrator and context from thermodynamic state.
        integrator = state._integrator
        context = state._context

        # Set coordinates.
        coordinates = self.replica_coordinates[replica_index]            
        context.setPositions(coordinates)
        setpositions_end_time = time.time()
        # Set box vectors.
        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        # Assign Maxwell-Boltzmann velocities.
        velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature) 
        context.setVelocities(velocities)
        setvelocities_end_time = time.time()
        # Run dynamics.
        integrator.step(self.nsteps_per_iteration)
        integrator_end_time = time.time() 
        # Store final coordinates
        getstate_start_time = time.time()
        openmm_state = context.getState(getPositions=True)
        getstate_end_time = time.time()
        self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
        # Store box vectors.
        self.replica_box_vectors[replica_index] = openmm_state.getPeriodicBoxVectors(asNumpy=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        positions_elapsed_time = setpositions_end_time - start_time
        velocities_elapsed_time = setvelocities_end_time - setpositions_end_time
        integrator_elapsed_time = integrator_end_time - setvelocities_end_time
        getstate_elapsed_time = getstate_end_time - integrator_end_time
        if self.verbose: print "Replica %d/%d: integrator elapsed time %.3f s (positions %.3f s | velocities %.3f s | integrate+getstate %.3f s)." % (replica_index, self.nreplicas, elapsed_time, positions_elapsed_time, velocities_elapsed_time, integrator_elapsed_time+getstate_elapsed_time)

        return elapsed_time

    def _propagate_replicas_mpi(self):
        """
        Propagate all replicas using MPI communicator.

        It is presumed all nodes have the correct configurations in the correct replica slots, but that state indices may be unsynchronized.

        TODO

        * Move synchronization of state information to mix_replicas?
        * Broadcast from root node only?

        """

        # Propagate all replicas.
        if self.verbose: print "Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / units.picoseconds)

        # Run just this node's share of states.
        if self.verbose and (self.mpicomm.rank == 0): print "Running trajectories..."
        start_time = time.time()
        # replica_lookup = { self.replica_states[replica_index] : replica_index for replica_index in range(self.nstates) } # replica_lookup[state_index] is the replica index currently at state 'state_index' # requires Python 2.7 features
        replica_lookup = dict( (self.replica_states[replica_index], replica_index) for replica_index in range(self.nstates) ) # replica_lookup[state_index] is the replica index currently at state 'state_index' # Python 2.6 compatible
        replica_indices = [ replica_lookup[state_index] for state_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size) ] # list of replica indices for this node to propagate
        for replica_index in replica_indices:
            if self.verbose: print "Node %3d/%3d propagating replica %3d state %3d..." % (self.mpicomm.rank, self.mpicomm.size, replica_index, self.replica_states[replica_index]) # DEBUG
            self._propagate_replica(replica_index)
        end_time = time.time()        
        elapsed_time = end_time - start_time
        # Collect elapsed time.
        node_elapsed_times = self.mpicomm.gather(elapsed_time, root=0) # barrier
        if self.verbose and (self.mpicomm.rank == 0):
            node_elapsed_times = numpy.array(node_elapsed_times)
            end_time = time.time()        
            elapsed_time = end_time - start_time
            barrier_wait_times = elapsed_time - node_elapsed_times
            print "Running trajectories: elapsed time %.3f s (barrier time min %.3f s | max %.3f s | avg %.3f s)" % (elapsed_time, barrier_wait_times.min(), barrier_wait_times.max(), barrier_wait_times.mean())
            print "Total time spent waiting for GPU: %.3f s" % (node_elapsed_times.sum())

        # Send final configurations and box vectors back to all nodes.
        if self.verbose and (self.mpicomm.rank == 0): print "Synchronizing trajectories..."
        start_time = time.time()
        replica_indices_gather = self.mpicomm.allgather(replica_indices)
        replica_coordinates_gather = self.mpicomm.allgather([ self.replica_coordinates[replica_index] for replica_index in replica_indices ])
        replica_box_vectors_gather = self.mpicomm.allgather([ self.replica_box_vectors[replica_index] for replica_index in replica_indices ])
        for (source, replica_indices) in enumerate(replica_indices_gather):
            for (index, replica_index) in enumerate(replica_indices):
                self.replica_coordinates[replica_index] = replica_coordinates_gather[source][index]
                self.replica_box_vectors[replica_index] = replica_box_vectors_gather[source][index]
        end_time = time.time()
        if self.verbose and (self.mpicomm.rank == 0): print "Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time)

        return
        
    def _propagate_replicas_serial(self):        
        """
        Propagate all replicas using serial execution.

        """

        # Propagate all replicas.
        if self.verbose: print "Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / units.picoseconds)
        for replica_index in range(self.nstates):
            self._propagate_replica(replica_index)

        return

    def _propagate_replicas(self):
        """
        Propagate all replicas.

        TODO

        * Report on efficiency of dyanmics (fraction of time wasted to overhead).

        """
        start_time = time.time()

        if self.mpicomm:
            self._propagate_replicas_mpi()
        else:
            self._propagate_replicas_serial()

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_replica = elapsed_time / float(self.nstates)
        ns_per_day = self.timestep * self.nsteps_per_iteration / time_per_replica * 24*60*60 / units.nanoseconds
        if self.verbose: print "Time to propagate all replicas: %.3f s (%.3f per replica, %.3f ns/day)." % (elapsed_time, time_per_replica, ns_per_day)

        return

    def _minimize_replica(self, replica_index):
        """
        Minimize the specified replica.

        """
        # Retrieve thermodynamic state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state
        # Retrieve integrator and context.
        # TODO: This needs to be adapted in case Integrator and Context objects are not cached.
        integrator = state._integrator
        context = state._context
        # Set coordinates.
        coordinates = self.replica_coordinates[replica_index]            
        context.setPositions(coordinates)
        # Set box vectors.
        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        # Minimize energy.
        minimized_coordinates = self.mm.LocalEnergyMinimizer.minimize(context, self.minimize_tolerance, self.minimize_maxIterations)
        # Store final coordinates
        openmm_state = context.getState(getPositions=True)
        self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
        # Clean up.
        del integrator, context

        return
            
    def _minimize_and_equilibrate(self):
        """
        Minimize and equilibrate all replicas.

        """

        # Minimize
        if self.minimize:
            if self.verbose: print "Minimizing all replicas..."

            if self.mpicomm:
                # MPI implementation.
                
                # Minimize this node's share of replicas.
                start_time = time.time()
                for replica_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                    if self.verbose and (self.mpicomm.rank == 0): print "node %d / %d : minimizing replica %d / %d" % (self.mpicomm.rank, self.mpicomm.size, replica_index, self.nstates)
                    self._minimize_replica(replica_index)
                end_time = time.time()
                self.mpicomm.barrier()
                if self.verbose and (self.mpicomm.rank == 0): print "Running trajectories: elapsed time %.3f s" % (end_time - start_time)

                # Send final configurations and box vectors back to all nodes.
                if self.verbose and (self.mpicomm.rank == 0): print "Synchronizing trajectories..."
                replica_coordinates_gather = self.mpicomm.allgather(self.replica_coordinates[self.mpicomm.rank:self.nstates:self.mpicomm.size])
                replica_box_vectors_gather = self.mpicomm.allgather(self.replica_box_vectors[self.mpicomm.rank:self.nstates:self.mpicomm.size])        
                for replica_index in range(self.nstates):
                    source = replica_index % self.mpicomm.size # node with trajectory data
                    index = replica_index // self.mpicomm.size # index within trajectory batch
                    self.replica_coordinates[replica_index] = replica_coordinates_gather[source][index]
                    self.replica_box_vectors[replica_index] = replica_box_vectors_gather[source][index]
                if self.verbose and (self.mpicomm.rank == 0): print "Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time)

            else:
                # Serial implementation.
                for replica_index in range(self.nstates):
                    self._minimize_replica(replica_index)

        # Equilibrate
        production_timestep = self.timestep
        for iteration in range(self.number_of_equilibration_iterations):
            if self.verbose: print "equilibration iteration %d / %d" % (iteration, self.number_of_equilibration_iterations)
            self._propagate_replicas()
        self.timestep = production_timestep
            
        return

    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        TODO

        * We have to re-order Context initialization if we have variable box volume
        * Parallel implementation
        
        """

        start_time = time.time()
        
        if self.verbose: print "Computing energies..."

        if self.mpicomm:
            # MPI version.

            # Compute energies for this node's share of states.
            for state_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                for replica_index in range(self.nstates):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_coordinates[replica_index], box_vectors=self.replica_box_vectors[replica_index], platform=self.platform)

            # Send final energies to all nodes.
            energies_gather = self.mpicomm.allgather(self.u_kl[:,self.mpicomm.rank:self.nstates:self.mpicomm.size])
            for state_index in range(self.nstates):
                source = state_index % self.mpicomm.size # node with trajectory data
                index = state_index // self.mpicomm.size # index within trajectory batch
                self.u_kl[:,state_index] = energies_gather[source][:,index]

        else:
            # Serial version.
            for state_index in range(self.nstates):
                for replica_index in range(self.nstates):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_coordinates[replica_index], box_vectors=self.replica_box_vectors[replica_index], platform=self.platform)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy= elapsed_time / float(self.nstates)**2 
        if self.verbose: print "Time to compute all energies %.3f s (%.3f per energy calculation)." % (elapsed_time, time_per_energy)

        return

    def _mix_all_replicas(self):
        """
        Attempt exchanges between all replicas to enhance mixing.

        TODO

        * Adjust nswap_attempts based on how many we can afford to do and not have mixing take a substantial fraction of iteration time.
        
        """

        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
        nswap_attempts = self.nstates**5 # number of swaps to attempt (ideal, but too slow!)
        nswap_attempts = self.nstates**3 # best compromise for pure Python?
        
        if self.verbose: print "Will attempt to swap all pairs of replicas, using a total of %d attempts." % nswap_attempts

        # Attempt swaps to mix replicas.
        for swap_attempt in range(nswap_attempts):
            # Choose replicas to attempt to swap.
            i = numpy.random.randint(self.nstates) # Choose replica i uniformly from set of replicas.
            j = numpy.random.randint(self.nstates) # Choose replica j uniformly from set of replicas.

            # Determine which states these resplicas correspond to.
            istate = self.replica_states[i] # state in replica slot i
            jstate = self.replica_states[j] # state in replica slot j

            # Reject swap attempt if any energies are nan.
            if (numpy.isnan(self.u_kl[i,jstate]) or numpy.isnan(self.u_kl[j,istate]) or numpy.isnan(self.u_kl[i,istate]) or numpy.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (numpy.random.rand() < math.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

        return

    def _mix_all_replicas_weave(self):
        """
        Attempt exchanges between all replicas to enhance mixing.
        Acceleration by 'weave' from scipy is used to speed up mixing by ~ 400x.
        
        """

        # TODO: Replace this with a different acceleration scheme to achieve better performance?

        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
        # TODO: Alternatively, use timing to figure out how many swaps we can do and still keep overhead to ~ 1% of iteration time?
        # nswap_attempts = self.nstates**5 # number of swaps to attempt (ideal, but too slow!)
        nswap_attempts = self.nstates**4 # number of swaps to attempt
        # Handled in C code below.
        
        if self.verbose: print "Will attempt to swap all pairs of replicas using weave-accelerated code, using a total of %d attempts." % nswap_attempts

        from scipy import weave

        # TODO: Replace drand48 with numpy random generator.
        code = """
        // Determine number of swap attempts.
        // TODO: Replace this with analytical result computed to guarantee sufficient mixing.        
        //long nswap_attempts = nstates*nstates*nstates*nstates*nstates; // K**5
        //long nswap_attempts = nstates*nstates*nstates; // K**3
        long nswap_attempts = nstates*nstates*nstates*nstates; // K**4

        // Attempt swaps.
        for(long swap_attempt = 0; swap_attempt < nswap_attempts; swap_attempt++) {
            // Choose replicas to attempt to swap.
            int i = (long)(drand48() * nstates); 
            int j = (long)(drand48() * nstates);

            // Determine which states these resplicas correspond to.            
            int istate = REPLICA_STATES1(i); // state in replica slot i
            int jstate = REPLICA_STATES1(j); // state in replica slot j

            // Reject swap attempt if any energies are nan.
            if ((std::isnan(U_KL2(i,jstate)) || std::isnan(U_KL2(j,istate)) || std::isnan(U_KL2(i,istate)) || std::isnan(U_KL2(j,jstate))))
               continue;

            // Compute log probability of swap.
            double log_P_accept = - (U_KL2(i,jstate) + U_KL2(j,istate)) + (U_KL2(i,istate) + U_KL2(j,jstate));

            // Record that this move has been proposed.
            NIJ_PROPOSED2(istate,jstate) += 1;
            NIJ_PROPOSED2(jstate,istate) += 1;

            // Accept or reject.
            if (log_P_accept >= 0.0 || (drand48() < exp(log_P_accept))) {
                // Swap states in replica slots i and j.
                int tmp = REPLICA_STATES1(i);
                REPLICA_STATES1(i) = REPLICA_STATES1(j);
                REPLICA_STATES1(j) = tmp;
                // Accumulate statistics
                NIJ_ACCEPTED2(istate,jstate) += 1;
                NIJ_ACCEPTED2(jstate,istate) += 1;
            }

        }
        """

        # Stage input temporarily.
        nstates = self.nstates
        replica_states = self.replica_states
        u_kl = self.u_kl
        Nij_proposed = self.Nij_proposed
        Nij_accepted = self.Nij_accepted

        # Execute inline C code with weave.
        info = weave.inline(code, ['nstates', 'replica_states', 'u_kl', 'Nij_proposed', 'Nij_accepted'], headers=['<math.h>', '<stdlib.h>'], verbose=2)

        # Store results.
        self.replica_states = replica_states
        self.Nij_proposed = Nij_proposed
        self.Nij_accepted = Nij_accepted

        return

    def _mix_neighboring_replicas(self):
        """
        Attempt exchanges between neighboring replicas only.

        """

        if self.verbose: print "Will attempt to swap only neighboring replicas."

        # Attempt swaps of pairs of replicas using traditional scheme (e.g. [0,1], [2,3], ...)
        offset = numpy.random.randint(2) # offset is 0 or 1
        for istate in range(offset, self.nstates-1, 2):
            jstate = istate + 1 # second state to attempt to swap with i

            # Determine which replicas these states correspond to.
            i = None
            j = None
            for index in range(self.nstates):
                if self.replica_states[index] == istate: i = index
                if self.replica_states[index] == jstate: j = index                

            # Reject swap attempt if any energies are nan.
            if (numpy.isnan(self.u_kl[i,jstate]) or numpy.isnan(self.u_kl[j,istate]) or numpy.isnan(self.u_kl[i,istate]) or numpy.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (numpy.random.rand() < math.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

        return

    def _mix_replicas(self):
        """
        Attempt to swap replicas according to user-specified scheme.
        
        """

        if (self.mpicomm) and (self.mpicomm.rank != 0):
            # Non-root nodes receive state information.
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)
            return

        if self.verbose: print "Mixing replicas..."        

        # Reset storage to keep track of swap attempts this iteration.
        self.Nij_proposed[:,:] = 0
        self.Nij_accepted[:,:] = 0

        # Perform swap attempts according to requested scheme.
        start_time = time.time()                    
        if self.replica_mixing_scheme == 'swap-neighbors':
            self._mix_neighboring_replicas()        
        elif self.replica_mixing_scheme == 'swap-all':
            # Try to use weave-accelerated mixing code if possible, otherwise fall back to Python-accelerated code.            
            try:
                self._mix_all_replicas_weave()            
            except:
                self._mix_all_replicas()
        elif self.replica_mixing_scheme == 'none':
            # Don't mix replicas.
            pass
        else:
            raise ParameterException("Replica mixing scheme '%s' unknown.  Choose valid 'replica_mixing_scheme' parameter." % self.replica_mixing_scheme)
        end_time = time.time()

        # Determine fraction of swaps accepted this iteration.        
        nswaps_attempted = self.Nij_proposed.sum()
        nswaps_accepted = self.Nij_accepted.sum()
        swap_fraction_accepted = 0.0
        if (nswaps_attempted > 0): swap_fraction_accepted = float(nswaps_accepted) / float(nswaps_attempted);            
        if self.verbose: print "Accepted %d / %d attempted swaps (%.1f %%)" % (nswaps_accepted, nswaps_attempted, swap_fraction_accepted * 100.0)

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

        if self.mpicomm:
            # Root node will share state information with all replicas.
            if self.verbose: print "Sharing state information..."
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)

        # Report on mixing.
        if self.verbose:
            print "Mixing of replicas took %.3f s" % (end_time - start_time)
                
        return

    def _show_mixing_statistics(self):
        """
        Print summary of mixing statistics.

        """

        # Only root node can print.
        if self.mpicomm and (self.mpicomm.rank != 0):
            return

        # Don't print anything until we've accumulated some statistics.
        if self.iteration < 2:
            return

        # Don't print anything if there is only one replica.
        if (self.nreplicas < 2):
            return
        
        initial_time = time.time()

        # Compute statistics of transitions.
        Nij = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for iteration in range(self.iteration - 1):
            for ireplica in range(self.nstates):
                istate = self.ncfile.variables['states'][iteration,ireplica]
                jstate = self.ncfile.variables['states'][iteration+1,ireplica]
                Nij[istate,jstate] += 0.5
                Nij[jstate,istate] += 0.5
        Tij = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for istate in range(self.nstates):
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

        # Show time consumption statistics.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "Time to compute mixing statistics %.3f s" % elapsed_time

        return

    def _initialize_netcdf(self):
        """
        Initialize NetCDF file for storage.
        
        """    
        
        # Only root node should set up NetCDF file.
        if self.mpicomm:
            if self.mpicomm.rank != 0: return

        # Open NetCDF 4 file for writing.
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'w', version=2)
        #ncfile = netcdf.Dataset(self.store_filename, 'w', version=2)        
        ncfile = netcdf.Dataset(self.store_filename, 'w', version='NETCDF4')
        
        # Create dimensions.
        ncfile.createDimension('iteration', 0) # unlimited number of iterations
        ncfile.createDimension('replica', self.nreplicas) # number of replicas
        ncfile.createDimension('atom', self.natoms) # number of atoms in system
        ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        setattr(ncfile, 'title', self.title)
        setattr(ncfile, 'application', 'YANK')
        setattr(ncfile, 'program', 'yank.py')
        setattr(ncfile, 'programVersion', __version__)
        setattr(ncfile, 'Conventions', 'YANK')
        setattr(ncfile, 'ConventionVersion', '0.1')
        
        # Create variables.
        ncvar_positions = ncfile.createVariable('positions', 'f', ('iteration','replica','atom','spatial'))
        ncvar_states    = ncfile.createVariable('states', 'i', ('iteration','replica'))
        ncvar_energies  = ncfile.createVariable('energies', 'f', ('iteration','replica','replica'))        
        ncvar_proposed  = ncfile.createVariable('proposed', 'l', ('iteration','replica','replica'))
        ncvar_accepted  = ncfile.createVariable('accepted', 'l', ('iteration','replica','replica'))                
        ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f', ('iteration','replica','spatial','spatial'))        
        ncvar_volumes  = ncfile.createVariable('volumes', 'f', ('iteration','replica'))
        
        # Define units for variables.
        setattr(ncvar_positions, 'units', 'nm')
        setattr(ncvar_states,    'units', 'none')
        setattr(ncvar_energies,  'units', 'kT')
        setattr(ncvar_proposed,  'units', 'none')
        setattr(ncvar_accepted,  'units', 'none')                
        setattr(ncvar_box_vectors, 'units', 'nm')
        setattr(ncvar_volumes, 'units', 'nm**3')

        # Define long (human-readable) names for variables.
        setattr(ncvar_positions, "long_name", "positions[iteration][replica][atom][spatial] is position of coordinate 'spatial' of atom 'atom' from replica 'replica' for iteration 'iteration'.")
        setattr(ncvar_states,    "long_name", "states[iteration][replica] is the state index (0..nstates-1) of replica 'replica' of iteration 'iteration'.")
        setattr(ncvar_energies,  "long_name", "energies[iteration][replica][state] is the reduced (unitless) energy of replica 'replica' from iteration 'iteration' evaluated at state 'state'.")
        setattr(ncvar_proposed,  "long_name", "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_accepted,  "long_name", "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_box_vectors, "long_name", "box_vectors[iteration][replica][i][j] is dimension j of box vector i for replica 'replica' from iteration 'iteration-1'.")
        setattr(ncvar_volumes, "long_name", "volume[iteration][replica] is the box volume for replica 'replica' from iteration 'iteration-1'.")

        # Create timestamp variable.
        ncvar_timestamp = ncfile.createVariable('timestamp', str, ('iteration',))

        # Create group for performance statistics.
        ncgrp_timings = ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',)) # total iteration time (seconds)
        ncvar_iteration_time = ncgrp_timings.createVariable('mixing', 'f', ('iteration',)) # time for mixing
        ncvar_iteration_time = ncgrp_timings.createVariable('propagate', 'f', ('iteration','replica')) # total time to propagate each replica
        
        # Store thermodynamic states.
        self._store_thermodynamic_states(ncfile)

        # Store run options
        self._store_options(ncfile)

        # Store metadata.
        if self.metadata:
            self._store_metadata(ncfile, 'metadata', self.metadata)

        # Force sync to disk to avoid data loss.
        ncfile.sync()

        # Store netcdf file handle.
        self.ncfile = ncfile
        
        return
    
    def _write_iteration_netcdf(self):
        """
        Write positions, states, and energies of current iteration to NetCDF file.
        
        """

        if self.mpicomm:
            # Only the root node will write data.
            if self.mpicomm.rank != 0: return

        initial_time = time.time()

        # Store replica positions.
        for replica_index in range(self.nstates):
            coordinates = self.replica_coordinates[replica_index]
            x = coordinates / units.nanometers
            self.ncfile.variables['positions'][self.iteration,replica_index,:,:] = x[:,:]
            
        # Store box vectors and volume.
        for replica_index in range(self.nstates):
            state_index = self.replica_states[replica_index]
            state = self.states[state_index]
            box_vectors = self.replica_box_vectors[replica_index]
            for i in range(3):
                self.ncfile.variables['box_vectors'][self.iteration,replica_index,i,:] = (box_vectors[i] / units.nanometers)
            volume = state._volume(box_vectors)
            self.ncfile.variables['volumes'][self.iteration,replica_index] = volume / (units.nanometers**3)

        # Store state information.
        self.ncfile.variables['states'][self.iteration,:] = self.replica_states[:]

        # Store energies.
        self.ncfile.variables['energies'][self.iteration,:,:] = self.u_kl[:,:]

        # Store mixing statistics.
        # TODO: Write mixing statistics for this iteration?
        self.ncfile.variables['proposed'][self.iteration,:,:] = self.Nij_proposed[:,:]
        self.ncfile.variables['accepted'][self.iteration,:,:] = self.Nij_accepted[:,:]        

        # Store timestamp this iteration was written.
        self.ncfile.variables['timestamp'][self.iteration] = time.ctime()

        # Force sync to disk to avoid data loss.
        presync_time = time.time()
        self.ncfile.sync()

        # Print statistics.
        final_time = time.time()
        sync_time = final_time - presync_time
        elapsed_time = final_time - initial_time
        if self.verbose: print "Writing data to NetCDF file took %.3f s (%.3f s for sync)" % (elapsed_time, sync_time)

        return

    def _run_sanity_checks(self):
        """
        Run some checks on current state information to see if something has gone wrong that precludes continuation.

        """

        abort = False

        # Check positions.
        for replica_index in range(self.nreplicas):
            coordinates = self.replica_coordinates[replica_index]
            x = coordinates / units.nanometers
            if numpy.any(numpy.isnan(x)):
                print "nan encountered in replica %d coordinates." % replica_index
                abort = True

        # Check energies.
        if numpy.any(numpy.isnan(self.u_kl)):
            print "nan encountered in u_kl state energies"
            abort = True

        if abort:
            if self.mpicomm:
                self.mpicomm.Abort()
            else:
                raise Exception("Aborting.")

        return

    def _store_thermodynamic_states(self, ncfile):
        """
        Store the thermodynamic states in a NetCDF file.

        """
        if self.verbose: print "Storing thermodynamic states in NetCDF file..."
        initial_time = time.time()
            
        # Create a group to store state information.
        ncgrp_stateinfo = ncfile.createGroup('thermodynamic_states')

        # Get number of states.
        ncvar_nstates = ncgrp_stateinfo.createVariable('nstates', int)
        ncvar_nstates.assignValue(self.nstates)


        # Temperatures.
        ncvar_temperatures = ncgrp_stateinfo.createVariable('temperatures', 'f', ('replica',))
        setattr(ncvar_temperatures, 'units', 'K')
        setattr(ncvar_temperatures, 'long_name', "temperatures[state] is the temperature of thermodynamic state 'state'")
        for state_index in range(self.nstates):
            ncvar_temperatures[state_index] = self.states[state_index].temperature / units.kelvin

        # Pressures.
        if self.states[0].pressure is not None:
            ncvar_temperatures = ncgrp_stateinfo.createVariable('pressures', 'f', ('replica',))
            setattr(ncvar_temperatures, 'units', 'atm')
            setattr(ncvar_temperatures, 'long_name', "pressures[state] is the external pressure of thermodynamic state 'state'")
            for state_index in range(self.nstates):
                ncvar_temperatures[state_index] = self.states[state_index].pressure / units.atmospheres                

        # TODO: Store other thermodynamic variables store in ThermodynamicState?  Generalize?
                
        # Systems.
        ncvar_serialized_states = ncgrp_stateinfo.createVariable('systems', str, ('replica',), zlib=True)
        setattr(ncvar_serialized_states, 'long_name', "systems[state] is the serialized OpenMM System corresponding to the thermodynamic state 'state'")
        for state_index in range(self.nstates):
            if self.verbose: print "Serializing state %d..." % state_index
            serialized = self.states[state_index].system.__getstate__()
            if self.verbose: print "Serialized state is %d B | %.3f KB | %.3f MB" % (len(serialized), len(serialized) / 1024.0, len(serialized) / 1024.0 / 1024.0)
            ncvar_serialized_states[state_index] = serialized
        final_time = time.time() 
        elapsed_time = final_time - initial_time

        if self.verbose: print "Serializing thermodynamic states took %.3f s." % elapsed_time
        
        return

    def _restore_thermodynamic_states(self, ncfile):
        """
        Restore the thermodynamic states from a NetCDF file.

        """
        if self.verbose_root: print "Restoring thermodynamic states from NetCDF file..."
        initial_time = time.time()
            
        # Make sure this NetCDF file contains thermodynamic state information.
        if not 'thermodynamic_states' in ncfile.groups:
            # Not found, signal failure.
            return False

        # Create a group to store state information.
        ncgrp_stateinfo = ncfile.groups['thermodynamic_states']

        # Get number of states.
        self.nstates = int(ncgrp_stateinfo.variables['nstates'][0])

        # Read state information.
        self.states = list()
        for state_index in range(self.nstates):
            # Populate a new ThermodynamicState object.
            state = ThermodynamicState()
            # Read temperature.
            state.temperature = ncgrp_stateinfo.variables['temperatures'][state_index] * units.kelvin
            # Read pressure, if present.
            if 'pressures' in ncgrp_stateinfo.variables:
                state.pressure = ncgrp_stateinfo.variables['pressures'][state_index] * units.atmospheres
            # Reconstitute System object.
            state.system = self.mm.System() 
            state.system.__setstate__(str(ncgrp_stateinfo.variables['systems'][state_index]))
            # Store state.
            self.states.append(state)

        final_time = time.time() 
        elapsed_time = final_time - initial_time
        if self.verbose_root: print "Restoring thermodynamic states from NetCDF file took %.3f s." % elapsed_time

        return True

    def _store_options(self, ncfile):
        """
        Store run parameters in NetCDF file.

        """

        if self.verbose_root: print "Storing run parameters in NetCDF file..."

        # Create scalar dimension if not already present.
        if 'scalar' not in ncfile.dimensions:
            ncfile.createDimension('scalar', 1) # scalar dimension
        
        # Create a group to store state information.
        ncgrp_options = ncfile.createGroup('options')

        # Store run parameters.
        for option_name in self.options_to_store:
            # Get option value.
            option_value = getattr(self, option_name)
            # If Quantity, strip off units first.
            option_unit = None
            if type(option_value) == units.Quantity:
                option_unit = option_value.unit
                option_value = option_value / option_unit
            # Store the Python type.
            option_type = type(option_value)
            # Handle booleans
            if type(option_value) == bool:
                option_value = int(option_value)
            # Store the variable.
            if self.verbose_root: print "Storing option: %s -> %s (type: %s)" % (option_name, option_value, str(option_type))
            if type(option_value) == str:
                ncvar = ncgrp_options.createVariable(option_name, type(option_value), 'scalar')
                packed_data = numpy.empty(1, 'O')
                packed_data[0] = option_value
                ncvar[:] = packed_data
            else:
                ncvar = ncgrp_options.createVariable(option_name, type(option_value))
                ncvar.assignValue(option_value)
            if option_unit: setattr(ncvar, 'units', str(option_unit))
            setattr(ncvar, 'type', option_type.__name__) 

        return

    def _restore_options(self, ncfile):
        """
        Restore run parameters from NetCDF file.
        """

        if self.verbose_root: print "Attempting to restore options from NetCDF file..."

        # Make sure this NetCDF file contains option information
        if not 'options' in ncfile.groups:
            # Not found, signal failure.
            return False

        # Find the group.
        ncgrp_options = ncfile.groups['options']

        # Load run parameters.
        for option_name in ncgrp_options.variables.keys():
            # Get NetCDF variable.
            option_ncvar = ncgrp_options.variables[option_name]
            # Get option value.
            option_value = option_ncvar[0]
            # Cast to Python type.
            type_name = getattr(option_ncvar, 'type') 
            option_value = eval(type_name + '(' + repr(option_value) + ')')
            # If Quantity, assign units.
            if hasattr(option_ncvar, 'units'):
                option_unit_name = getattr(option_ncvar, 'units')
                if option_unit_name[0] == '/': option_unit_name = '1' + option_unit_name
                option_unit = eval(option_unit_name, vars(units))
                option_value = units.Quantity(option_value, option_unit)
            # Store option.
            if self.verbose_root: print "Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value))
            setattr(self, option_name, option_value)
            
        # Signal success.
        return True

    def _store_metadata(self, ncfile, groupname, metadata):
        """
        Store metadata in NetCDF file.
        
        """

        # Create group.
        ncgrp = ncfile.createGroup(groupname)

        # Store metadata.
        for (key, value) in metadata.iteritems():
            # TODO: Handle more sophisticated types.
            ncvar = ncgrp.createVariable(key, type(value))
            ncvar.assignValue(value)

        return

    def _resume_from_netcdf(self):
        """
        Resume execution by reading current positions and energies from a NetCDF file.
        
        """

        # Open NetCDF file for reading
        if self.verbose: print "Reading NetCDF file '%s'..." % self.store_filename
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'r') # Scientific.IO.NetCDF
        ncfile = netcdf.Dataset(self.store_filename, 'r') # netCDF4
        
        # TODO: Perform sanity check on file before resuming

        # Get current dimensions.
        self.iteration = ncfile.variables['positions'].shape[0] - 1 
        self.nstates = ncfile.variables['positions'].shape[1]
        self.natoms = ncfile.variables['positions'].shape[2]
        if self.verbose: print "iteration = %d, nstates = %d, natoms = %d" % (self.iteration, self.nstates, self.natoms)

        # Restore positions.
        self.replica_coordinates = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['positions'][self.iteration,replica_index,:,:].astype(numpy.float64).copy()
            coordinates = units.Quantity(x, units.nanometers)
            self.replica_coordinates.append(coordinates)
        
        # Restore box vectors.
        self.replica_box_vectors = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['box_vectors'][self.iteration,replica_index,:,:].astype(numpy.float64).copy()
            box_vectors = units.Quantity(x, units.nanometers)
            self.replica_box_vectors.append(box_vectors)

        # Restore state information.
        self.replica_states = ncfile.variables['states'][self.iteration,:].copy()

        # Restore energies.
        self.u_kl = ncfile.variables['energies'][self.iteration,:,:].copy()
        
        # Close NetCDF file.
        ncfile.close()        

        # We will work on the next iteration.
        self.iteration += 1

        if (self.mpicomm is None) or (self.mpicomm.rank == 0):
            # Reopen NetCDF file for appending, and maintain handle.
            self.ncfile = netcdf.Dataset(self.store_filename, 'a')
        else:
            self.ncfile = None
        
        return

    def _show_energies(self):
        """
        Show energies (in units of kT) for all replicas at all states.

        """

        # Only root node can print.
        if self.mpicomm and (self.mpicomm.rank != 0):
            return

        # print header
        print "%-24s %16s" % ("reduced potential (kT)", "current state"),
        for state_index in range(self.nstates):
            print " state %3d" % state_index,
        print ""

        # print energies in kT
        for replica_index in range(self.nstates):
            print "replica %-16d %16d" % (replica_index, self.replica_states[replica_index]),
            for state_index in range(self.nstates):
                u = self.u_kl[replica_index,state_index]
                if (u > 1e6):
                    print "%10.3e" % u,
                else:
                    print "%10.1f" % u,
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

        """

        # Get number of atoms
        natoms = system.getNumParticles()

        # Decorate System object with vector of masses for efficiency.
        if not hasattr(system, 'masses'):
            masses = simtk.unit.Quantity(numpy.zeros([natoms,3], numpy.float64), units.amu)
            for atom_index in range(natoms):
                mass = system.getParticleMass(atom_index) # atomic mass
                masses[atom_index,:] = mass
            setattr(system, 'masses', masses)

        # Retrieve masses.
        masses = getattr(system, 'masses')
  
        # Compute thermal energy and velocity scaling factors.
        kT = kB * temperature # thermal energy
        sigma2 = kT / masses

        # Assign velocities from the Maxwell-Boltzmann distribution.
        # TODO: This is wacky because units.sqrt cannot operate on numpy vectors.
        
        # NEW (working) CODE
        velocity_unit = units.nanometers / units.picoseconds
        velocities = units.Quantity(numpy.sqrt(sigma2 / (velocity_unit**2)) * numpy.random.randn(natoms, 3), velocity_unit)
        
        # OLD (broken) CODE
        #velocities = units.Quantity(numpy.sqrt(sigma2/sigma2.unit) * numpy.random.randn(natoms, 3), units.sqrt(sigma2.unit))

        # DEBUG: Print kinetic energy.
        #kinetic_energy = units.sum(units.sum(0.5 * masses * velocities**2)) 
        #print kinetic_energy / units.kilocalories_per_mole
    
        # Return velocities
        return velocities

    def _analysis(self):
        """
        Perform online analysis each iteration.

        UNDER CONSTRUCTION

        Every iteration, this will update the estimate of the state relative free energy differences and statistical uncertainties.
        We can additionally request further analysis.

        """

        # Only root node can perform analysis.
        if self.mpicomm and (self.mpicomm.rank != 0): return

        # Determine how many iterations there are data available for.
        replica_states = self.ncfile.variables['states'][:,:]
        u_nkl_replica = self.ncfile.variables['energies'][:,:,:]

        niterations_completed = replica_states.shape[0]
        nstates = replica_states.shape[1]

        # Deconvolute replicas and compute total simulation effective self-energy timeseries.
        u_kln = numpy.zeros([nstates, nstates, niterations_completed], numpy.float32)
        u_n = numpy.zeros([niterations_completed], numpy.float64)
        for iteration in range(niterations_completed):
            state_indices = replica_states[iteration,:] 
            u_n[iteration] = 0.0
            for replica_index in range(nstates):
                state_index = state_indices[replica_index]
                u_n[iteration] += u_nkl_replica[iteration,replica_index,state_index]
                u_kln[state_index,:,iteration] = u_nkl_replica[iteration,replica_index,:]

        # Determine optimal equilibration time, statistical inefficiency, and effectively uncorrelated sample indices.
        import timeseries
        [equilibration_end, g, Neff_max, indices] = timeseries.detectEquilibration(u_n)
        N_k = indices.size * numpy.ones([nstates], numpy.int32)

        # Next, analyze with pymbar, initializing with last estimate of free energies.
        import pymbar
        if hasattr(self, 'f_k'):
            mbar = pymbar.MBAR(u_kln[:,:,indices], N_k, f_k_initial=self.f_k)
        else:
            mbar = pymbar.MBAR(u_kln[:,:,indices], N_k)            

        # Store free energies.
        self.f_k = mbar.f_k

        # Store free energy differences and uncertainties.
        #[Delta_f_ij, dDelta_f_ij] = mbar.getFreeEnergyDifferences()

        # Compute entropy and enthalpy.
        [Delta_f_ij, dDelta_f_ij, Delta_u_ij, dDelta_u_ij, Delta_s_ij, dDelta_s_ij] = mbar.computeEntropyAndEnthalpy()

        # Store analysis summary.
        # TODO: Convert this to an object?
        analysis = dict()
        analysis['equilibration_end'] = equilibration_end
        analysis['g'] = g
        analysis['indices'] = indices
        analysis['Delta_f_ij'] = Delta_f_ij
        analysis['dDelta_f_ij'] = dDelta_f_ij
        analysis['Delta_u_ij'] = Delta_u_ij
        analysis['dDelta_u_ij'] = dDelta_u_ij
        analysis['Delta_s_ij'] = Delta_s_ij
        analysis['dDelta_s_ij'] = dDelta_s_ij


        self.analysis = analysis

        return

    def analyze(self):
        """
        Analyze the current simulation and return estimated free energies.
        
        RETURNS
        
        analysis (dict) - analysis object containing end of equilibrated region, statistical inefficiency, and free energy differences:
        
        KEYS
        
        equilibration_end (int) - the last iteration in the discarded equilibrated region
        g (float) - estimated statistical inefficiency of production region
        indices (list of int) - equilibrated, effectively uncorrelated iteration indices used in analysis
        Delta_f_ij (numpy array of nstates x nstates) - Delta_f_ij[i,j] is the free energy difference f_j - f_i in units of kT
        dDelta_f_ij (numpy array of nstates x nstates) - dDelta_f_ij[i,j] is estimated standard error of Delta_f_ij[i,j]
        Delta_u_ij
        dDelta_u_ij
        Delta_s_ij
        dDelta_s_ij
        
        """
        # Update analysis on root node.
        self._analysis()

        if self.mpicomm: self.analysis = self.mpicomm.bcast(self.analysis, root=0) # broadcast analysis from root node
        
        # Return analysis object
        return self.analysis

#=============================================================================================
# Parallel tempering
#=============================================================================================

class ParallelTempering(ReplicaExchange):
    """
    Parallel tempering simulation facility.

    DESCRIPTION

    This class provides a facility for parallel tempering simulations.  It is a subclass of ReplicaExchange, but provides
    various convenience methods and efficiency improvements for parallel tempering simulations, so should be preferred for
    this type of simulation.  In particular, the System only need be specified once, while the temperatures (or a temperature
    range) is used to automatically build a set of ThermodynamicState objects for replica-exchange.  Efficiency improvements
    make use of the fact that the reduced potentials are linear in inverse temperature.
    
    EXAMPLES

    Parallel tempering of alanine dipeptide in implicit solvent.
    
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
    >>> nreplicas = 3
    >>> simulation = ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.minimize = False
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    Parallel tempering of alanine dipeptide in explicit solvent at 1 atm.
    
    >>> # Create alanine dipeptide system
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideExplicit()
    >>> # Add Monte Carlo barsostat to system (must be same pressure as simulation).
    >>> import simtk.openmm as openmm
    >>> pressure = 1.0 * units.atmosphere
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * units.kelvin
    >>> Tmax = 600.0 * units.kelvin    
    >>> nreplicas = 3
    >>> simulation = ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, pressure=pressure, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False # don't minimize first
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    def __init__(self, system, coordinates, store_filename, protocol=None, Tmin=None, Tmax=None, ntemps=None, temperatures=None, pressure=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize a parallel tempering simulation object.

        ARGUMENTS
        
        system (simtk.openmm.System) - the system to simulate
        coordinates (simtk.unit.Quantity of numpy natoms x 3 array of units length, or list) - coordinate set(s) for one or more replicas, assigned in a round-robin fashion
        store_filename (string) -  name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        Tmin, Tmax, ntemps - min and max temperatures, and number of temperatures for exponentially-spaced temperature selection (default: None)
        temperatures (list of simtk.unit.Quantity with units of temperature) - if specified, this list of temperatures will be used instead of (Tmin, Tmax, ntemps) (default: None)
        pressure (simtk.unit.Quantity with units of pressure) - if specified, a MonteCarloBarostat will be added (or modified) to perform NPT simulations
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict.  Provided keywords will be matched to object variables to replace defaults. (default: None)
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)

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
        
        states = [ ThermodynamicState(system=system, temperature=self.temperatures[i], pressure=pressure) for i in range(ntemps) ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm, mpicomm=mpicomm, metadata=metadata)

        # Override title.
        self.title = 'Parallel tempering simulation created using ParallelTempering class of repex.py on %s' % time.asctime(time.localtime())
        
        return

    def _compute_energies(self):
        """
        Compute reduced potentials of all replicas at all states (temperatures).

        NOTES

        Because only the temperatures differ among replicas, we replace the generic O(N^2) replica-exchange implementation with an O(N) implementation.
        
        """

        start_time = time.time()
        if self.verbose: print "Computing energies..."

        if self.mpicomm:
            # MPI implementation

            # NOTE: This version incurs the overhead of context creation/deletion.
            # TODO: Use cached contexts instead.
            
            # Create an integrator and context.
            state = self.states[0]
            integrator = self.mm.VerletIntegrator(self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)

            for replica_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                # Set coordinates.
                context.setPositions(self.replica_coordinates[replica_index])
                # Compute potential energy.
                openmm_state = context.getState(getEnergy=True)            
                potential_energy = openmm_state.getPotentialEnergy()           
                # Compute energies at this state for all replicas.
                for state_index in range(self.nstates):
                    # Compute reduced potential
                    beta = 1.0 / (kB * self.states[state_index].temperature)
                    self.u_kl[replica_index,state_index] = beta * potential_energy

            # Gather energies.
            energies_gather = self.mpicomm.allgather(self.u_kl[self.mpicomm.rank:self.nstates:self.mpicomm.size,:])
            for replica_index in range(self.nstates):
                source = replica_index % self.mpicomm.size # node with trajectory data
                index = replica_index // self.mpicomm.size # index within trajectory batch
                self.u_kl[replica_index,:] = energies_gather[source][index]

            # Clean up.
            del context, integrator
                
        else:
            # Serial implementation.
            # NOTE: This version incurs the overhead of context creation/deletion.
            # TODO: Use cached contexts instead.

            # Create an integrator and context.
            state = self.states[0]
            integrator = self.mm.VerletIntegrator(self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)
        
            # Compute reduced potentials for all configurations in all states.
            for replica_index in range(self.nstates):
                # Set coordinates.
                context.setPositions(self.replica_coordinates[replica_index])
                # Compute potential energy.
                openmm_state = context.getState(getEnergy=True)            
                potential_energy = openmm_state.getPotentialEnergy()           
                # Compute energies at this state for all replicas.
                for state_index in range(self.nstates):
                    # Compute reduced potential
                    beta = 1.0 / (kB * self.states[state_index].temperature)
                    self.u_kl[replica_index,state_index] = beta * potential_energy

            # Clean up.
            del context, integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy = elapsed_time / float(self.nstates)
        if self.verbose: print "Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy)

        return

#=============================================================================================
# Hamiltonian exchange
#=============================================================================================

class HamiltonianExchange(ReplicaExchange):
    """
    Hamiltonian exchange simulation facility.

    DESCRIPTION

    This class provides an implementation of a Hamiltonian exchange simulation based on the ReplicaExchange facility.
    It provides several convenience classes and efficiency improvements, and should be preferentially used for Hamiltonian
    exchange simulations over ReplicaExchange when possible.
    
    EXAMPLES
    
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
    >>> # Create simulation.
    >>> simulation = HamiltonianExchange(reference_state, systems, coordinates, store_filename)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False
    >>> # Run simulation.
    >>> simulation.run() # run the simulation
    
    """

    def __init__(self, reference_state, systems, coordinates, store_filename, protocol=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize a Hamiltonian exchange simulation object.

        ARGUMENTS

        reference_state (ThermodynamicState) - reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems (list of simtk.openmm.System) - list of systems to simulate (one per replica)
        coordinates (simtk.unit.Quantity of numpy natoms x 3 with units length) -  coordinates (or a list of coordinates objects) for initial assignment of replicas (will be used in round-robin assignment)
        store_filename (string) - name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)        

        """

        if systems is None:
            states = None
        else:
            # Create thermodynamic states from systems.        
            states = [ ThermodynamicState(system=system, temperature=reference_state.temperature, pressure=reference_state.pressure, mm=mm) for system in systems ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm, mpicomm=mpicomm, metadata=metadata)

        # Override title.
        self.title = 'Hamiltonian exchange simulation created using HamiltonianExchange class of repex.py on %s' % time.asctime(time.localtime())
        
        return

    

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

