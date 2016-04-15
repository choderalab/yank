#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
repex
=====

Replica-exchange simulation algorithms and specific variants.

DESCRIPTION

This module provides a general facility for running replica-exchange simulations, as well as
derived classes for special cases such as parallel tempering (in which the states differ only
in temperature) and Hamiltonian exchange (in which the state differ only by potential function).

Provided classes include:

* ReplicaExchange - Base class for general replica-exchange simulations among specified ThermodynamicState objects
* ParallelTempering - Convenience subclass of ReplicaExchange for parallel tempering simulations (one System object, many temperatures/pressures)
* HamiltonianExchange - Convenience subclass of ReplicaExchange for Hamiltonian exchange simulations (many System objects, same temperature/pressure)

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
* Add another layer of abstraction so that the base class uses generic log probabilities, rather than reduced potentials?
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

from simtk import openmm
from simtk import unit

import os, os.path
import math
import copy
import time
import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
import mdtraj as md
import netCDF4 as netcdf

from utils import is_terminal_verbose

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA # Boltzmann constant

# TODO: Fix MAX_SEED when we determine what maximum allowed seed is.
#MAX_SEED = 4294967 # maximum seed for OpenMM setRandomNumberSeed
MAX_SEED = 2**31 - 1 # maximum seed for OpenMM setRandomNumberSeed

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
# Thermodynamic state description
#=============================================================================================

class ThermodynamicState(object):
    """
    Data specifying a thermodynamic state obeying Boltzmann statistics.

    Examples
    --------
    Specify an NVT state for a water box at 298 K.

    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> waterbox = testsystems.WaterBox()
    >>> [system, positions] = [waterbox.system, waterbox.positions]
    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin)

    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin, pressure=1.0*unit.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    Notes
    -----
    This state object cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    TODO
    ----
    * Implement a more fundamental ProbabilityState as a base class?
    * Implement pH.

    """

    def __init__(self, system=None, temperature=None, pressure=None):
        """
        Initialize the thermodynamic state.

        Parameters
        ----------

        system : simtk.openmm.System, optional, default=None
           A System object describing the potential energy function for the system
           Note: Only a shallow copy is made.
        temperature : simtk.unit.Quantity compatible with 'kelvin', optional, default=None
           The temperature for a system with constant temperature
        pressure : simtk.unit.Quantity compatible with 'atmospheres', optional, default=None
           The pressure for constant-pressure systems (default: None)

        """

        # Initialize.
        self.system = None          # the System object governing the potential energy computation
        self.temperature = None     # the temperature
        self.pressure = None        # the pressure, or None if not isobaric

        # Store provided values.
        if system is not None:
            self.system = system
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

        return

    def reduced_potential(self, positions, box_vectors=None, platform=None, context=None):
        """
        Compute the reduced potential for the given positions in this thermodynamic state.

        Parameters
        ----------
        positions : simtk.unit.Quantity of Nx3 numpy.array
           Positions[n,k] is kth coordinate of particle n
        box_vectors : tuple of Vec3 or ???, optional, default=None
           Periodic box vectors, if present.
        platform : simtk.openmm.Platform, optional, default=None
           Platform to use, or None if default.
        context : simtk.openmm.Context, optional, default=none
           If specified, use this Context.

        Returns
        -------
        u : float
           The unitless reduced potential (which can be considered to have units of kT)

        Examples
        --------
        Compute the reduced potential of a Lennard-Jones cluster at 100 K.

        >>> from simtk import unit
        >>> from openmmtools import testsystems
        >>> testsystem = testsystems.LennardJonesCluster()
        >>> [system, positions] = [testsystem.system, testsystem.positions]
        >>> state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin)
        >>> potential = state.reduced_potential(positions)

        Compute the reduced potential of a Lennard-Jones fluid at 100 K and 1 atm.

        >>> testsystem = testsystems.LennardJonesFluid()
        >>> [system, positions] = [testsystem.system, testsystem.positions]
        >>> state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin, pressure=1.0*unit.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(positions, box_vectors)

        Compute the reduced potential of a water box at 298 K and 1 atm.

        >>> testsystem = testsystems.WaterBox()
        >>> [system, positions] = [testsystem.system, testsystem.positions]
        >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin, pressure=1.0*unit.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(positions, box_vectors)

        Notes
        -----
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

        References
        ----------
        [1] Shirts MR and Chodera JD. Statistically optimal analysis of equilibrium states. J Chem Phys 129:124105, 2008.

        TODO
        ----
        * Instead of requiring configuration and box_vectors be passed separately, develop a Configuration or Snapshot class.

        """

        # If pressure is specified, ensure box vectors have been provided.
        if (self.pressure is not None) and (box_vectors is None):
            raise ParameterException("box_vectors must be specified if constant-pressure ensemble.")

        # Compute potential energy.
        potential_energy = self._compute_potential_energy(positions, box_vectors=box_vectors, platform=platform, context=context)

        # Compute inverse temperature.
        beta = 1.0 / (kB * self.temperature)

        # Compute reduced potential.
        reduced_potential = beta * potential_energy
        if self.pressure is not None:
            reduced_potential += beta * self.pressure * self._volume(box_vectors) * unit.AVOGADRO_CONSTANT_NA

        return reduced_potential

    def _compute_potential_energy(self, positions, box_vectors=None, platform=None, context=None):
        """
        Compute the potential energy for the given positions in this thermodynamic state.

        Parameters
        ----------
        positions : simtk.unit.Quantity of Nx3 numpy.array
           Positions[n,k] is kth coordinate of particle n
        box_vectors : tuple of Vec3 or ???, optional, default=None
           Periodic box vectors, if present.
        platform : simtk.openmm.Platform, optional, default=None
           Platform to use, or None if default.
        context : simtk.openmm.Context, optional, default=none
           If specified, use this Context.

        Returns
        -------
        potential_energy : simtk.unit.Quantity with units compatible with kilojoules_per_mole
           The unit-bearing potential energy.

        """

        # Create OpenMM context to compute potential energy.
        cleanup_context = False
        if context == None:
            integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
            if platform:
                context = openmm.Context(self.system, integrator, platform)
            else:
                context = openmm.Context(self.system, integrator)
            cleanup_context = True

        # Must set box vectors first.
        if box_vectors is not None: context.setPeriodicBoxVectors(*box_vectors)

        # Set positions.
        context.setPositions(positions)

        # Retrieve potential energy.
        potential_energy = context.getState(getEnergy=True).getPotentialEnergy()

        # Clean up if we created a Context and Integrator.
        if cleanup_context:
            del context, integrator

        return potential_energy

    def is_compatible_with(self, state):
        """
        Determine whether another state is in the same thermodynamic ensemble (e.g. NVT, NPT).

        ARGUMENTS

        state (ThermodynamicState) - thermodynamic state whose compatibility is to be determined

        RETURNS

        is_compatible (boolean) - True if 'state' is of the same ensemble (e.g. both NVT, both NPT), False otherwise

        EXAMPLES

        Create NVT and NPT states.

        >>> from simtk import unit
        >>> from openmmtools import testsystems
        >>> testsystem = testsystems.LennardJonesCluster()
        >>> [system, positions] = [testsystem.system, testsystem.positions]
        >>> nvt_state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin)
        >>> npt_state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin, pressure=1.0*unit.atmospheres)

        Test compatibility.

        >>> test1 = nvt_state.is_compatible_with(nvt_state)
        >>> test2 = nvt_state.is_compatible_with(npt_state)
        >>> test3 = npt_state.is_compatible_with(nvt_state)
        >>> test4 = npt_state.is_compatible_with(npt_state)

        """

        is_compatible = True

        # Make sure systems have the same number of atoms.
        if ((self.system != None) and (state.system != None)):
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

        Examples
        --------
        Create an NVT state.

        >>> from simtk import unit
        >>> from openmmtools import testsystems
        >>> testsystem = testsystems.LennardJonesCluster()
        >>> [system, positions] = [testsystem.system, testsystem.positions]
        >>> state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin)

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

        r = "<ThermodynamicState object"
        if self.temperature is not None:
            r += ", temperature = %s" % str(self.temperature)
        if self.pressure is not None:
            r += ", pressure = %s" % str(self.pressure)
        r += ">"

        return r

    def _volume(self, box_vectors):
        """
        Return the volume of the current configuration.

        Returns
        -------
        volume : simtk.unit.Quantity
           The volume of the system (in units of length^3), or None if no box positions are defined

        Examples
        --------
        Compute the volume of a Lennard-Jones fluid at 100 K and 1 atm.

        >>> from openmmtools import testsystems
        >>> testsystem = testsystems.LennardJonesFluid()
        >>> [system, positions] = [testsystem.system, testsystem.positions]
        >>> state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin, pressure=1.0*unit.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> volume = state._volume(box_vectors)

        """

        # Compute volume of parallelepiped.
        [a,b,c] = box_vectors
        A = np.array([a/a.unit, b/a.unit, c/a.unit])
        volume = np.linalg.det(A) * a.unit**3
        return volume

#=============================================================================================
# Replica-exchange simulation
#=============================================================================================

class ReplicaExchange(object):
    """
    Replica-exchange simulation facility.

    This base class provides a general replica-exchange simulation facility, allowing any set of thermodynamic states
    to be specified, along with a set of initial positions to be assigned to the replicas in a round-robin fashion.
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

    Attributes
    ----------
    The following parameters (attributes) can be set after the object has been created, but before it has been
    initialized by a call to run():

    collision_rate : simtk.unit.Quantity (units: 1/time)
       The collision rate used for Langevin dynamics (default: 90 ps^-1)
    constraint_tolerance : float
       Relative constraint tolerance (default: 1e-6)
    timestep : simtk.unit.Quantity (units: time)
       Timestep for Langevin dyanmics (default: 2 fs)
    nsteps_per_iteration : int
       Number of timesteps per iteration (default: 500)
    number_of_iterations : int
       Number of replica-exchange iterations to simulate (default: 100)
    number_of_equilibration_iterations : int
       Number of equilibration iterations before beginning exchanges (default: 0)
    equilibration_timestep : simtk.unit.Quantity (units: time)
       Timestep for use in equilibration (default: 2 fs)
    title : str
       Title for the simulation.
    minimize : bool
       Minimize configurations before running the simulation (default: True)
    minimize_tolerance : simtk.unit.Quantity (units: energy/mole/length)
       Set minimization tolerance (default: 1.0 * unit.kilojoules_per_mole / unit.nanometers).
    minimize_max_iterations : int
       Maximum number of iterations for minimization.
    replica_mixing_scheme : str
       Specify how to mix replicas. Supported schemes are 'swap-neighbors' and
       'swap-all' (default: 'swap-all').
    online_analysis : bool
       If True, analysis will occur each iteration (default: False).
    online_analysis_min_iterations : int
       Minimum number of iterations needed to begin online analysis (default: 20).
    show_energies : bool
       If True, will print energies at each iteration (default: True).
    show_mixing_statistics : bool
       If True, will show mixing statistics at each iteration (default: True).

    TODO
    ----
    * Replace hard-coded Langevin dynamics with general MCMC moves.
    * Allow parallel resource to be used, if available (likely via Parallel Python).
    * Add support for and autodetection of other NetCDF4 interfaces.
    * Add HDF5 support.

    Examples
    --------
    Parallel tempering simulation of alanine dipeptide in implicit solvent (replica exchange among temperatures)
    (This is just an illustrative example; use ParallelTempering class for actual production parallel tempering simulations.)

    >>> # Create test system.
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideImplicit()
    >>> [system, positions] = [testsystem.system, testsystem.positions]
    >>> # Create thermodynamic states for parallel tempering with exponentially-spaced schedule.
    >>> from simtk import unit
    >>> import math
    >>> nreplicas = 3 # number of temperature replicas
    >>> T_min = 298.0 * unit.kelvin # minimum temperature
    >>> T_max = 600.0 * unit.kelvin # maximum temperature
    >>> T_i = [ T_min + (T_max - T_min) * (math.exp(float(i) / float(nreplicas-1)) - 1.0) / (math.e - 1.0) for i in range(nreplicas) ]
    >>> states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(nreplicas) ]
    >>> import tempfile
    >>> store_filename = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
    >>> # Create simulation.
    >>> simulation = ReplicaExchange(store_filename)
    >>> simulation.create(states, positions) # initialize the replica-exchange simulation
    >>> simulation.minimize = False
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * unit.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.run() # run the simulation
    >>> del simulation # clean up

    Extend the simulation

    >>> simulation = ReplicaExchange(store_filename)
    >>> simulation.resume()
    >>> simulation.number_of_iterations = 4 # extend
    >>> simulation.run()

    Clean up.

    >>> os.remove(store_filename)

    """

    default_parameters = {'collision_rate': 5.0 / unit.picosecond,
                          'constraint_tolerance': 1.0e-6,
                          'timestep': 2.0 * unit.femtosecond,
                          'nsteps_per_iteration': 500,
                          'number_of_iterations': 1,
                          'equilibration_timestep': 1.0 * unit.femtosecond,
                          'number_of_equilibration_iterations': 1,
                          'title': 'Replica-exchange simulation created using ReplicaExchange class of repex.py on %s' % time.asctime(time.localtime()),
                          'minimize': True,
                          'minimize_tolerance': 1.0 * unit.kilojoules_per_mole / unit.nanometers,
                          'minimize_max_iterations': 0,
                          'replica_mixing_scheme': 'swap-all',
                          'online_analysis': False,
                          'online_analysis_min_iterations': 20,
                          'show_energies': True,
                          'show_mixing_statistics': True
                          }

    # Options to store.
    options_to_store = ['collision_rate', 'constraint_tolerance', 'timestep', 'nsteps_per_iteration', 'number_of_iterations', 'equilibration_timestep', 'number_of_equilibration_iterations', 'title', 'minimize', 'replica_mixing_scheme', 'online_analysis', 'show_mixing_statistics']

    def __init__(self, store_filename, mpicomm=None, mm=None, **kwargs):
        """
        Initialize replica-exchange simulation facility.

        Parameters
        ----------
        store_filename : string
           Name of file to bind simulation to use as storage for checkpointing and storage of results.
        mm : implementation of simtk.openmm, optional, default=simtk.openmm
           OpenMM API implementation to use
        mpicomm : mpi4py communicator, optional, default=None
           MPI communicator, if parallel execution is desired

        Other Parameters
        ----------------
        **kwargs
            Parameters in ReplicaExchange.default_parameters corresponding public attributes.

        """
        # To allow for parameters to be modified after object creation, class is not initialized until a call to self._initialize().
        self._initialized = False

        # Select default OpenMM implementation if not specified.
        self.mm = mm
        if mm is None: self.mm = openmm

        # Set MPI communicator (or None if not used).
        self.mpicomm = mpicomm

        # Set default options.
        # These can be changed externally until object is initialized.
        self.platform = None
        self.platform_name = None
        self.integrator = None # OpenMM integrator to use for propagating dynamics

        # Initialize keywords parameters and check for unknown keywords parameters
        for par, default in self.default_parameters.items():
            setattr(self, par, kwargs.pop(par, default))
        if kwargs:
            raise TypeError('got an unexpected keyword arguments {}'.format(
                ', '.join(kwargs.keys())))

        # Record store file filename
        self.store_filename = store_filename

        # Check if netcdf file exists, assuming we want to resume if one exists.
        self._resume = os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0)
        if self.mpicomm:
            logger.debug('Node {}/{}: MPI bcast - sharing self._resume'.format(
                    self.mpicomm.rank, self.mpicomm.size))
            self._resume = self.mpicomm.bcast(self._resume, root=0)  # use whatever root node decides

        return

    def create(self, states, positions, options=None, metadata=None):
        """
        Create new replica-exchange simulation.

        Parameters
        ----------
        states : list of ThermodynamicState
           Thermodynamic states to simulate, where one replica is allocated per state.
           Each state must have a system with the same number of atoms, and the same
           thermodynamic ensemble (combination of temperature, pressure, pH, etc.) must
           be defined for each.
        positions : Coordinate object or iterable container of Coordinate objects)
           One or more sets of initial positions
           to be initially assigned to replicas in a round-robin fashion, provided simulation is not resumed from store file.
           Currently, positions must be specified as a list of simtk.unit.Quantity-wrapped np arrays.
        options : dict, optional, default=None
           Optional dict to use for specifying simulation options. Provided keywords will be matched to object variables to replace defaults.
        metadata : dict, optional, default=None
           metadata to store in a 'metadata' group in store file

        """

        # Check if netcdf file exists.
        file_exists = os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0)
        if self.mpicomm:
            logger.debug('Node {}/{}: MPI bcast - sharing file_exists'.format(
                    self.mpicomm.rank, self.mpicomm.size))
            file_exists = self.mpicomm.bcast(file_exists, root=0)  # use whatever root node decides
        if file_exists:
            raise RuntimeError("NetCDF file %s already exists; cowardly refusing to overwrite." % self.store_filename)
        self._resume = False

        # TODO: Make a deep copy of specified states once this is fixed in OpenMM.
        # self.states = copy.deepcopy(states)
        self.states = states

        # Determine number of replicas from the number of specified thermodynamic states.
        self.nreplicas = len(self.states)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.states:
            if not state.is_compatible_with(self.states[0]):
                raise ParameterError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        # Distribute coordinate information to replicas in a round-robin fashion.
        # We have to explicitly check to see if z is a list or a set here because it turns out that np 2D arrays are iterable as well.
        # TODO: Handle case where positions are passed in as a list of tuples, or list of lists, or list of Vec3s, etc.
        if type(positions) in [type(list()), type(set())]:
            self.provided_positions = [ unit.Quantity(np.array(coordinate_set / coordinate_set.unit), coordinate_set.unit) for coordinate_set in positions ]
        else:
            self.provided_positions = [ unit.Quantity(np.array(positions / positions.unit), positions.unit) ]

        # Handle provided 'options' dict, replacing any options provided by caller in dictionary.
        if options is not None:
            for key in options.keys(): # for each provided key
                if key in vars(self).keys(): # if this is also a simulation parameter
                    value = options[key]
                    logger.debug("from options: %s -> %s" % (key, str(value)))
                    vars(self)[key] = value # replace default simulation parameter with provided parameter

        # Store metadata to store in store file.
        self.metadata = metadata

        # Initialize NetCDF file.
        self._initialize_create()

        return

    def resume(self, options=None):
        """
        Parameters
        ----------
        options : dict, optional, default=None
           will override any options restored from the store file.

        """
        self._resume = True

        # Check if netcdf file exists.
        file_exists = os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0)
        if self.mpicomm:
            logger.debug('Node {}/{}: MPI bcast - sharing file_exists'.format(
                    self.mpicomm.rank, self.mpicomm.size))
            file_exists = self.mpicomm.bcast(file_exists, root=0)  # use whatever root node decides
        if not file_exists:
            raise Exception("NetCDF file %s does not exist; cannot resume." % self.store_filename)

        # Try to restore thermodynamic states and run options from the NetCDF file.
        ncfile = netcdf.Dataset(self.store_filename, 'r')
        self._restore_thermodynamic_states(ncfile)
        self._restore_options(ncfile)
        self._restore_metadata(ncfile)
        ncfile.close()

        # Determine number of replicas from the number of specified thermodynamic states.
        self.nreplicas = len(self.states)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.states:
            if not state.is_compatible_with(self.states[0]):
                raise ParameterError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        # Handle provided 'options' dict, replacing any options provided by caller in dictionary.
        # TODO: Check to make sure that only allowed overrides are specified.
        if options:
            for key in options.keys(): # for each provided key
                if key in vars(self).keys(): # if this is also a simulation parameter
                    value = options[key]
                    logger.debug("from options: %s -> %s" % (key, str(value)))
                    vars(self)[key] = value # replace default simulation parameter with provided parameter

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
        r += "%d coordinate sets provided\n" % len(self.provided_positions)
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

    @classmethod
    def _status_from_ncfile(cls, ncfile):
        """
        Return status dict of current calculation.

        Returns
        -------
        status : dict
           Returns a dict of useful information about current simulation progress.

        """
        status = dict()

        status['number_of_iterations'] = ncfile.variables['positions'].shape[0]
        status['nstates'] = ncfile.variables['positions'].shape[1]
        status['natoms'] = ncfile.variables['positions'].shape[2]

        return status

    @classmethod
    def status_from_store(cls, store_filename):
        """
        Return status dict of calculation on disk.

        Parameters
        ----------
        store_filename : str
           The name of the NetCDF storage filename.

        Returns
        -------
        status : dict
           Returns a dict of useful information about current simulation progress.

        """
        ncfile = netcdf.Dataset(store_filename, 'r')
        status = ReplicaExchange._status_from_ncfile(ncfile)
        ncfile.close()
        return status

    def status(self):
        """
        Return status dict of current calculation.

        Returns
        -------
        status : dict
           Returns a dict of useful information about current simulation progress.

        """
        ncfile = netcdf.Dataset(self.store_filename, 'r')
        status = ReplicaExchange._status_from_ncfile(self.ncfile)
        ncfile.close()

        return status

    def run(self, niterations_to_run=None):
        """
        Run the replica-exchange simulation.

        Any parameter changes (via object attributes) that were made between object creation and calling this method become locked in
        at this point, and the object will create and bind to the store file.  If the store file already exists, the run will be resumed
        if possible; otherwise, an exception will be raised.

        Parameters
        ----------
        niterations_to_run : int, optional, default=None
           If specfied, only at most the specified number of iterations will be run.

        """
        if not self._initialized:
            self._initialize_resume()

        # Main loop
        run_start_time = time.time()
        run_start_iteration = self.iteration
        iteration_limit = self.number_of_iterations
        if niterations_to_run:
            iteration_limit = min(self.iteration + niterations_to_run, iteration_limit)
        while (self.iteration < iteration_limit):
            logger.debug("\nIteration %d / %d" % (self.iteration+1, self.number_of_iterations))
            initial_time = time.time()

            # Attempt replica swaps to sample from equilibrium permuation of states associated with replicas.
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states.
            self._compute_energies()

            # Show energies.
            if self.show_energies:
                self._show_energies()

            # Write iteration to storage file.
            self._write_iteration_netcdf()

            # Increment iteration counter.
            self.iteration += 1

            # Show mixing statistics.
            if self.show_mixing_statistics:
                self._show_mixing_statistics()

            # Perform online analysis.
            if self.online_analysis:
                self._analysis()

            # Show timing statistics if debug level is activated
            if logger.isEnabledFor(logging.DEBUG):
                final_time = time.time()
                elapsed_time = final_time - initial_time
                estimated_time_remaining = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.number_of_iterations - self.iteration)
                estimated_total_time = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.number_of_iterations)
                estimated_finish_time = final_time + estimated_time_remaining
                logger.debug("Iteration took %.3f s." % elapsed_time)
                logger.debug("Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time))))

            # Perform sanity checks to see if we should terminate here.
            self._run_sanity_checks()

        # Clean up and close storage files.
        self._finalize()

        return

    def _determine_fastest_platform(self, system):
        """
        Determine fastest OpenMM platform for given system.

        Parameters
        ----------
        system : simtk.openmm.System
           The system for which the fastest OpenMM Platform object is to be determined.

        Returns
        -------
        platform : simtk.openmm.Platform
           The fastest OpenMM Platform for the specified system.

        """
        timestep = 1.0 * unit.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator)
        platform = context.getPlatform()
        del context, integrator
        return platform

    def _initialize_create(self):
        """
        Initialize the simulation and create a storage file, closing it after completion.

        """

        if self._initialized:
            raise Error("Simulation has already been initialized.")

        # Extract a representative system.
        representative_system = self.states[0].system

        # Turn off verbosity if not master node.
        if self.mpicomm:
            # Have each node report that it is initialized.
            # TODO this doesn't work on worker nodes since they report only warning entries and higher
            logger.debug("Initialized node %d / %d" % (self.mpicomm.rank, self.mpicomm.size))

        # Display papers to be cited.
        if is_terminal_verbose():
            self._display_citations()

        # Determine number of alchemical states.
        self.nstates = len(self.states)

        # Determine number of atoms in systems.
        self.natoms = representative_system.getNumParticles()

        # Allocate storage.
        self.replica_positions = list() # replica_positions[i] is the configuration currently held in replica i
        self.replica_box_vectors = list() # replica_box_vectors[i] is the set of box vectors currently held in replica i
        self.replica_states     = np.zeros([self.nstates], np.int64) # replica_states[i] is the state that replica i is currently at
        self.u_kl               = np.zeros([self.nstates, self.nstates], np.float64)
        self.swap_Pij_accepted  = np.zeros([self.nstates, self.nstates], np.float64)
        self.Nij_proposed       = np.zeros([self.nstates,self.nstates], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = np.zeros([self.nstates,self.nstates], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1

        # Distribute coordinate information to replicas in a round-robin fashion, making a deep copy.
        if not self._resume:
            self.replica_positions = [ copy.deepcopy(self.provided_positions[replica_index % len(self.provided_positions)]) for replica_index in range(self.nstates) ]

        # Assign default box vectors.
        self.replica_box_vectors = list()
        for state in self.states:
            [a,b,c] = state.system.getDefaultPeriodicBoxVectors()
            box_vectors = unit.Quantity(np.zeros([3,3], np.float32), unit.nanometers)
            box_vectors[0,:] = a
            box_vectors[1,:] = b
            box_vectors[2,:] = c
            self.replica_box_vectors.append(box_vectors)

        # Assign initial replica states.
        for replica_index in range(self.nstates):
            self.replica_states[replica_index] = replica_index

        # Initialize current iteration counter.
        self.iteration = 0

        # Initialize NetCDF file.
        self._initialize_netcdf()

        # Store initial state.
        self._write_iteration_netcdf()

        # Close NetCDF file.
        self.ncfile.close()
        self.ncfile = None

        return

    def _initialize_resume(self):
        """
        Initialize the simulation, and bind to a storage file.

        """

        if self._initialized:
            raise Error("Simulation has already been initialized.")

        # Extract a representative system.
        representative_system = self.states[0].system

        # Turn off verbosity if not master node.
        if self.mpicomm:
            # Have each node report that it is initialized.
            # TODO this doesn't work on worker nodes since they report only warning entries and higher
            logger.debug("Initialized node %d / %d" % (self.mpicomm.rank, self.mpicomm.size))

        # Display papers to be cited.
        if is_terminal_verbose():
            self._display_citations()

        # Determine number of alchemical states.
        self.nstates = len(self.states)

        # Determine number of atoms in systems.
        self.natoms = representative_system.getNumParticles()

        # If no platform is specified, instantiate a platform, or try to use the fastest platform.
        if self.platform is None:
            # Handle OpenMM platform selection.
            # TODO: Can we handle this more gracefully, or push this off to ReplicaExchange?
            if self.platform_name:
                self.platform = openmm.Platform.getPlatformByName(self.platform_name)
            else:
                self.platform = self._determine_fastest_platform(representative_system)

            # Use only a single CPU thread if we are using the CPU platform.
            # TODO: Since there is an environment variable that can control this, we may want to avoid doing this.
            if (self.platform.getName() == 'CPU') and self.mpicomm:
                logger.debug("Setting 'CpuThreads' to 1 because MPI is active.")
                self.platform.setPropertyDefaultValue('CpuThreads', '1')

        # Allocate storage.
        self.replica_positions = list() # replica_positions[i] is the configuration currently held in replica i
        self.replica_box_vectors = list() # replica_box_vectors[i] is the set of box vectors currently held in replica i
        self.replica_states     = np.zeros([self.nstates], np.int32) # replica_states[i] is the state that replica i is currently at
        self.u_kl               = np.zeros([self.nstates, self.nstates], np.float64)
        self.swap_Pij_accepted  = np.zeros([self.nstates, self.nstates], np.float64)
        self.Nij_proposed       = np.zeros([self.nstates,self.nstates], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = np.zeros([self.nstates,self.nstates], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1

        # Distribute coordinate information to replicas in a round-robin fashion, making a deep copy.
        if not self._resume:
            self.replica_positions = [ copy.deepcopy(self.provided_positions[replica_index % len(self.provided_positions)]) for replica_index in range(self.nstates) ]

        # Assign default box vectors.
        self.replica_box_vectors = list()
        for state in self.states:
            [a,b,c] = state.system.getDefaultPeriodicBoxVectors()
            box_vectors = unit.Quantity(np.zeros([3,3], np.float32), unit.nanometers)
            box_vectors[0,:] = a
            box_vectors[1,:] = b
            box_vectors[2,:] = c
            self.replica_box_vectors.append(box_vectors)

        # Assign initial replica states.
        for replica_index in range(self.nstates):
            self.replica_states[replica_index] = replica_index

        # Resume from NetCDF file.
        self._resume_from_netcdf()

        # Show energies.
        if self.show_energies:
            self._show_energies()

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

        if hasattr(self, 'ncfile'):
            if self.ncfile is not None:
                self.ncfile.close()
                self.ncfile = None

        return

    def _display_citations(self):
        """
        Display papers to be cited.

        TODO:

        * Add original citations for various replica-exchange schemes.
        * Show subset of OpenMM citations based on what features are being used.

        """

        openmm_citations = """\
        Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
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

        # If temperature and pressure are specified, make sure MonteCarloBarostat is attached.
        if state.temperature and state.pressure:
            forces = { state.system.getForce(index).__class__.__name__ : state.system.getForce(index) for index in range(state.system.getNumForces()) }

            if 'MonteCarloAnisotropicBarostat' in forces:
                raise Exception('MonteCarloAnisotropicBarostat is unsupported.')

            if 'MonteCarloBarostat' in forces:
                barostat = forces['MonteCarloBarostat']
                # Set temperature and pressure.
                barostat.setTemperature(state.temperature)
                barostat.setDefaultPressure(state.pressure)
                barostat.setRandomNumberSeed(int(np.random.randint(0, MAX_SEED)))
            else:
                # Create barostat and add it to the system if it doesn't have one already.
                barostat = openmm.MonteCarloBarostat(state.pressure, state.temperature)
                barostat.setRandomNumberSeed(int(np.random.randint(0, MAX_SEED)))
                state.system.addForce(barostat)

        # Create Context and integrator.
        integrator = openmm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)
        integrator.setRandomNumberSeed(int(np.random.randint(0, MAX_SEED)))
        if self.platform:
            context = openmm.Context(state.system, integrator, self.platform)
        else:
            context = openmm.Context(state.system, integrator)

        # Set box vectors.
        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        # Set positions.
        positions = self.replica_positions[replica_index]
        context.setPositions(positions)
        setpositions_end_time = time.time()
        # Assign Maxwell-Boltzmann velocities.
        context.setVelocitiesToTemperature(state.temperature, int(np.random.randint(0, MAX_SEED)))
        setvelocities_end_time = time.time()
        # Run dynamics.
        integrator.step(self.nsteps_per_iteration)
        integrator_end_time = time.time()
        # Store final positions
        getstate_start_time = time.time()
        openmm_state = context.getState(getPositions=True,enforcePeriodicBox=True)
        getstate_end_time = time.time()
        self.replica_positions[replica_index] = openmm_state.getPositions(asNumpy=True)
        # Store box vectors.
        self.replica_box_vectors[replica_index] = openmm_state.getPeriodicBoxVectors(asNumpy=True)

        # Clean up.
        del context, integrator

        # Compute timing.
        end_time = time.time()
        elapsed_time = end_time - start_time
        positions_elapsed_time = setpositions_end_time - start_time
        velocities_elapsed_time = setvelocities_end_time - setpositions_end_time
        integrator_elapsed_time = integrator_end_time - setvelocities_end_time
        getstate_elapsed_time = getstate_end_time - integrator_end_time
        logger.debug("Replica %d/%d: integrator elapsed time %.3f s (positions %.3f s | velocities %.3f s | integrate+getstate %.3f s)." % (replica_index, self.nreplicas, elapsed_time, positions_elapsed_time, velocities_elapsed_time, integrator_elapsed_time+getstate_elapsed_time))

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
        logger.debug("Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / unit.picoseconds))

        # Run just this node's share of states.
        logger.debug("Running trajectories...")
        start_time = time.time()
        # replica_lookup = { self.replica_states[replica_index] : replica_index for replica_index in range(self.nstates) } # replica_lookup[state_index] is the replica index currently at state 'state_index' # requires Python 2.7 features
        replica_lookup = dict( (self.replica_states[replica_index], replica_index) for replica_index in range(self.nstates) ) # replica_lookup[state_index] is the replica index currently at state 'state_index' # Python 2.6 compatible
        replica_indices = [ replica_lookup[state_index] for state_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size) ] # list of replica indices for this node to propagate
        for replica_index in replica_indices:
            logger.debug("Node %3d/%3d propagating replica %3d state %3d..." % (self.mpicomm.rank, self.mpicomm.size, replica_index, self.replica_states[replica_index]))
            self._propagate_replica(replica_index)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Collect elapsed time.
        node_elapsed_times = self.mpicomm.gather(elapsed_time, root=0) # barrier
        if self.mpicomm.rank == 0 and logger.isEnabledFor(logging.DEBUG):
            node_elapsed_times = np.array(node_elapsed_times)
            end_time = time.time()
            elapsed_time = end_time - start_time
            barrier_wait_times = elapsed_time - node_elapsed_times
            logger.debug("Running trajectories: elapsed time %.3f s (barrier time min %.3f s | max %.3f s | avg %.3f s)" % (elapsed_time, barrier_wait_times.min(), barrier_wait_times.max(), barrier_wait_times.mean()))
            logger.debug("Total time spent waiting for GPU: %.3f s" % (node_elapsed_times.sum()))

        # Send final configurations and box vectors back to all nodes.
        logger.debug("Synchronizing trajectories...")
        start_time = time.time()
        replica_indices_gather = self.mpicomm.allgather(replica_indices)
        replica_positions_gather = self.mpicomm.allgather([ self.replica_positions[replica_index] for replica_index in replica_indices ])
        replica_box_vectors_gather = self.mpicomm.allgather([ self.replica_box_vectors[replica_index] for replica_index in replica_indices ])
        for (source, replica_indices) in enumerate(replica_indices_gather):
            for (index, replica_index) in enumerate(replica_indices):
                self.replica_positions[replica_index] = replica_positions_gather[source][index]
                self.replica_box_vectors[replica_index] = replica_box_vectors_gather[source][index]
        end_time = time.time()
        logger.debug("Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time))

        return

    def _propagate_replicas_serial(self):
        """
        Propagate all replicas using serial execution.

        """

        # Propagate all replicas.
        logger.debug("Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / unit.picoseconds))
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
        ns_per_day = self.timestep * self.nsteps_per_iteration / time_per_replica * 24*60*60 / unit.nanoseconds
        logger.debug("Time to propagate all replicas: %.3f s (%.3f per replica, %.3f ns/day)." % (elapsed_time, time_per_replica, ns_per_day))

        return

    def _minimize_replica(self, replica_index):
        """
        Minimize the specified replica.

        """
        # Retrieve thermodynamic state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state
        # Create integrator and context.
        integrator = self.mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = self.mm.Context(state.system, integrator, self.platform)
        # Set box vectors.
        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        # Set positions.
        positions = self.replica_positions[replica_index]
        context.setPositions(positions)
        # Minimize energy.
        minimized_positions = self.mm.LocalEnergyMinimizer.minimize(context, self.minimize_tolerance, self.minimize_max_iterations)
        # Store final positions
        self.replica_positions[replica_index] = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
        # Clean up.
        del integrator, context

        return

    def _minimize_and_equilibrate(self):
        """
        Minimize and equilibrate all replicas.

        """

        # Minimize
        if self.minimize:
            logger.debug("Minimizing all replicas...")

            if self.mpicomm:
                # MPI implementation.
                logger.debug("MPI implementation.")
                # Minimize this node's share of replicas.
                start_time = time.time()
                for replica_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                    logger.debug("node %d / %d : minimizing replica %d / %d" % (self.mpicomm.rank, self.mpicomm.size, replica_index, self.nstates))
                    self._minimize_replica(replica_index)
                end_time = time.time()
                debug_msg = 'Node {}/{}: MPI barrier'.format(self.mpicomm.rank, self.mpicomm.size)
                logger.debug(debug_msg + ' - waiting for the minimization to be completed.')
                self.mpicomm.barrier()
                logger.debug("Running trajectories: elapsed time %.3f s" % (end_time - start_time))

                # Send final configurations and box vectors back to all nodes.
                logger.debug("Synchronizing trajectories...")
                replica_positions_gather = self.mpicomm.allgather(self.replica_positions[self.mpicomm.rank:self.nstates:self.mpicomm.size])
                replica_box_vectors_gather = self.mpicomm.allgather(self.replica_box_vectors[self.mpicomm.rank:self.nstates:self.mpicomm.size])
                for replica_index in range(self.nstates):
                    source = replica_index % self.mpicomm.size # node with trajectory data
                    index = replica_index // self.mpicomm.size # index within trajectory batch
                    self.replica_positions[replica_index] = replica_positions_gather[source][index]
                    self.replica_box_vectors[replica_index] = replica_box_vectors_gather[source][index]
                logger.debug("Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time))

            else:
                # Serial implementation.
                logger.debug("Serial implementation.")
                for replica_index in range(self.nstates):
                    logger.debug("minimizing replica %d / %d" % (replica_index, self.nstates))
                    self._minimize_replica(replica_index)

        # Equilibrate
        production_timestep = self.timestep
        for iteration in range(self.number_of_equilibration_iterations):
            logger.debug("equilibration iteration %d / %d" % (iteration, self.number_of_equilibration_iterations))
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

        logger.debug("Computing energies...")

        if self.mpicomm:
            # MPI version.

            # Compute energies for this node's share of states.
            for state_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                for replica_index in range(self.nstates):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_positions[replica_index], box_vectors=self.replica_box_vectors[replica_index], platform=self.platform)

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
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_positions[replica_index], box_vectors=self.replica_box_vectors[replica_index], platform=self.platform)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy= elapsed_time / float(self.nstates)**2
        logger.debug("Time to compute all energies %.3f s (%.3f per energy calculation)." % (elapsed_time, time_per_energy))

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

        logger.debug("Will attempt to swap all pairs of replicas, using a total of %d attempts." % nswap_attempts)

        # Attempt swaps to mix replicas.
        for swap_attempt in range(nswap_attempts):
            # Choose replicas to attempt to swap.
            i = np.random.randint(self.nstates) # Choose replica i uniformly from set of replicas.
            j = np.random.randint(self.nstates) # Choose replica j uniformly from set of replicas.

            # Determine which states these resplicas correspond to.
            istate = self.replica_states[i] # state in replica slot i
            jstate = self.replica_states[j] # state in replica slot j

            # Reject swap attempt if any energies are nan.
            if (np.isnan(self.u_kl[i,jstate]) or np.isnan(self.u_kl[j,istate]) or np.isnan(self.u_kl[i,istate]) or np.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (np.random.rand() < math.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

        return

    def _mix_all_replicas_cython(self):
        """
        Attempt to exchange all replicas to enhance mixing, calling code written in Cython.
        """

        from . import mixing
        from mixing._mix_replicas import _mix_replicas_cython

        replica_states = md.utils.ensure_type(self.replica_states, np.int64, 1, "Replica States")
        u_kl = md.utils.ensure_type(self.u_kl, np.float64, 2, "Reduced Potentials")
        Nij_proposed = md.utils.ensure_type(self.Nij_proposed, np.int64, 2, "Nij Proposed")
        Nij_accepted = md.utils.ensure_type(self.Nij_accepted, np.int64, 2, "Nij accepted")
        _mix_replicas_cython(self.nstates**4, self.nstates, replica_states, u_kl, Nij_proposed, Nij_accepted)

        #replica_states = np.array(self.replica_states, np.int64)
        #u_kl = np.array(self.u_kl, np.float64)
        #Nij_proposed = np.array(self.Nij_proposed, np.int64)
        #Nij_accepted = np.array(self.Nij_accepted, np.int64)
        #_mix_replicas._mix_replicas_cython(self.nstates**4, self.nstates, replica_states, u_kl, Nij_proposed, Nij_accepted)

        self.replica_states = replica_states
        self.Nij_proposed = Nij_proposed
        self.Nij_accepted = Nij_accepted

    def _mix_neighboring_replicas(self):
        """
        Attempt exchanges between neighboring replicas only.

        """

        logger.debug("Will attempt to swap only neighboring replicas.")

        # Attempt swaps of pairs of replicas using traditional scheme (e.g. [0,1], [2,3], ...)
        offset = np.random.randint(2) # offset is 0 or 1
        for istate in range(offset, self.nstates-1, 2):
            jstate = istate + 1 # second state to attempt to swap with i

            # Determine which replicas these states correspond to.
            i = None
            j = None
            for index in range(self.nstates):
                if self.replica_states[index] == istate: i = index
                if self.replica_states[index] == jstate: j = index

            # Reject swap attempt if any energies are nan.
            if (np.isnan(self.u_kl[i,jstate]) or np.isnan(self.u_kl[j,istate]) or np.isnan(self.u_kl[i,istate]) or np.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (np.random.rand() < math.exp(log_P_accept))):
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
            logger.debug('Node {}/{}: MPI bcast - sharing replica_states'.format(
                    self.mpicomm.rank, self.mpicomm.size))
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)
            return

        logger.debug("Mixing replicas...")

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
                self._mix_all_replicas_cython()
            except ValueError as e:
                logger.warning(e.message)
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
        logger.debug("Accepted %d / %d attempted swaps (%.1f %%)" % (nswaps_accepted, nswaps_attempted, swap_fraction_accepted * 100.0))

        # Estimate cumulative transition probabilities between all states.
        Nij_accepted = self.ncfile.variables['accepted'][:,:,:].sum(0) + self.Nij_accepted
        Nij_proposed = self.ncfile.variables['proposed'][:,:,:].sum(0) + self.Nij_proposed
        swap_Pij_accepted = np.zeros([self.nstates,self.nstates], np.float64)
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
            logger.debug('Node {}/{}: MPI bcast - sharing replica_states'.format(
                    self.mpicomm.rank, self.mpicomm.size))
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)

        # Report on mixing.
        logger.debug("Mixing of replicas took %.3f s" % (end_time - start_time))

        return

    def _accumulate_mixing_statistics(self):
        """Return the mixing transition matrix Tij."""
        try:
            return self._accumulate_mixing_statistics_update()
        except AttributeError:
            pass
        except ValueError:
            logger.info("Inconsistent transition count matrix detected, recalculating from scratch.")

        return self._accumulate_mixing_statistics_full()

    def _accumulate_mixing_statistics_full(self):
        """Compute statistics of transitions iterating over all iterations of repex."""
        states = self.ncfile.variables['states']
        self._Nij = np.zeros([self.nstates, self.nstates], np.float64)
        for iteration in range(states.shape[0]-1):
            for ireplica in range(self.nstates):
                istate = states[iteration, ireplica]
                jstate = states[iteration + 1, ireplica]
                self._Nij[istate, jstate] += 0.5
                self._Nij[jstate, istate] += 0.5

        Tij = np.zeros([self.nstates, self.nstates], np.float64)
        for istate in range(self.nstates):
            Tij[istate] = self._Nij[istate] / self._Nij[istate].sum()

        return Tij

    def _accumulate_mixing_statistics_update(self):
        """Compute statistics of transitions updating Nij of last iteration of repex."""

        states = self.ncfile.variables['states']
        if self._Nij.sum() != (states.shape[0] - 2) * self.nstates:  # n_iter - 2 = (n_iter - 1) - 1.  Meaning that you have exactly one new iteration to process.
            raise(ValueError("Inconsistent transition count matrix detected.  Perhaps you tried updating twice in a row?"))

        for ireplica in range(self.nstates):
            istate = states[self.iteration-2, ireplica]
            jstate = states[self.iteration-1, ireplica]
            self._Nij[istate, jstate] += 0.5
            self._Nij[jstate, istate] += 0.5

        Tij = np.zeros([self.nstates, self.nstates], np.float64)
        for istate in range(self.nstates):
            Tij[istate] = self._Nij[istate] / self._Nij[istate].sum()

        return Tij

    def _show_mixing_statistics(self):

        if self.iteration < 2:
            return
        if self.mpicomm and self.mpicomm.rank != 0:
            return  # only root node have access to ncfile
        if not logger.isEnabledFor(logging.DEBUG):
            return

        Tij = self._accumulate_mixing_statistics()

        # Print observed transition probabilities.
        PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
        logger.debug("Cumulative symmetrized state mixing transition matrix:")
        str_row = "%6s" % ""
        for jstate in range(self.nstates):
            str_row += "%6d" % jstate
        logger.debug(str_row)
        for istate in range(self.nstates):
            str_row = "%-6d" % istate
            for jstate in range(self.nstates):
                P = Tij[istate,jstate]
                if (P >= PRINT_CUTOFF):
                    str_row += "%6.3f" % P
                else:
                    str_row += "%6s" % ""
            logger.debug(str_row)

        # Estimate second eigenvalue and equilibration time.
        mu = np.linalg.eigvals(Tij)
        mu = -np.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            logger.debug("Perron eigenvalue is unity; Markov chain is decomposable.")
        else:
            logger.debug("Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1])))

    def _initialize_netcdf(self):
        """
        Initialize NetCDF file for storage.

        """

        # Only root node should set up NetCDF file.
        if self.mpicomm:
            if self.mpicomm.rank != 0: return

        # Open NetCDF 4 file for writing.
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
        setattr(ncfile, 'programVersion', 'unknown') # TODO: Include actual version.
        setattr(ncfile, 'Conventions', 'YANK')
        setattr(ncfile, 'ConventionVersion', '0.1')

        # Create variables.
        ncvar_positions = ncfile.createVariable('positions', 'f4', ('iteration','replica','atom','spatial'), zlib=True, chunksizes=(1,self.nreplicas,self.natoms,3))
        ncvar_states    = ncfile.createVariable('states', 'i4', ('iteration','replica'), zlib=False, chunksizes=(1,self.nreplicas))
        ncvar_energies  = ncfile.createVariable('energies', 'f8', ('iteration','replica','replica'), zlib=False, chunksizes=(1,self.nreplicas,self.nreplicas))
        ncvar_proposed  = ncfile.createVariable('proposed', 'i4', ('iteration','replica','replica'), zlib=False, chunksizes=(1,self.nreplicas,self.nreplicas))
        ncvar_accepted  = ncfile.createVariable('accepted', 'i4', ('iteration','replica','replica'), zlib=False, chunksizes=(1,self.nreplicas,self.nreplicas))
        ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f4', ('iteration','replica','spatial','spatial'), zlib=False, chunksizes=(1,self.nreplicas,3,3))
        ncvar_volumes  = ncfile.createVariable('volumes', 'f8', ('iteration','replica'), zlib=False, chunksizes=(1,self.nreplicas))

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
        ncvar_timestamp = ncfile.createVariable('timestamp', str, ('iteration',), zlib=False, chunksizes=(1,))

        # Create group for performance statistics.
        ncgrp_timings = ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',), zlib=False, chunksizes=(1,)) # total iteration time (seconds)
        ncvar_iteration_time = ncgrp_timings.createVariable('mixing', 'f', ('iteration',), zlib=False, chunksizes=(1,)) # time for mixing
        ncvar_iteration_time = ncgrp_timings.createVariable('propagate', 'f', ('iteration','replica'), zlib=False, chunksizes=(1,self.nreplicas)) # total time to propagate each replica

        # Store thermodynamic states.
        self._store_thermodynamic_states(ncfile)

        # Store run options
        self._store_options(ncfile)

        # Store metadata.
        if self.metadata:
            self._store_metadata(ncfile)

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
            positions = self.replica_positions[replica_index]
            x = positions / unit.nanometers
            self.ncfile.variables['positions'][self.iteration,replica_index,:,:] = x[:,:]

        # Store box vectors and volume.
        for replica_index in range(self.nstates):
            state_index = self.replica_states[replica_index]
            state = self.states[state_index]
            box_vectors = self.replica_box_vectors[replica_index]
            for i in range(3):
                self.ncfile.variables['box_vectors'][self.iteration,replica_index,i,:] = (box_vectors[i] / unit.nanometers)
            volume = state._volume(box_vectors)
            self.ncfile.variables['volumes'][self.iteration,replica_index] = volume / (unit.nanometers**3)

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
        logger.debug("Writing data to NetCDF file took %.3f s (%.3f s for sync)" % (elapsed_time, sync_time))

        return

    def _run_sanity_checks(self):
        """
        Run some checks on current state information to see if something has gone wrong that precludes continuation.

        """

        abort = False

        # Check positions.
        for replica_index in range(self.nreplicas):
            positions = self.replica_positions[replica_index]
            x = positions / unit.nanometers
            if np.any(np.isnan(x)):
                logger.warning("nan encountered in replica %d positions." % replica_index)
                abort = True

        # Check energies.
        for replica_index in range(self.nreplicas):
            if np.any(np.isnan(self.u_kl[replica_index,:])):
                logger.warning("nan encountered in u_kl state energies for replica %d" % replica_index)
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
        logger.debug("Storing thermodynamic states in NetCDF file...")
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
            ncvar_temperatures[state_index] = self.states[state_index].temperature / unit.kelvin

        # Pressures.
        if self.states[0].pressure is not None:
            ncvar_temperatures = ncgrp_stateinfo.createVariable('pressures', 'f', ('replica',))
            setattr(ncvar_temperatures, 'units', 'atm')
            setattr(ncvar_temperatures, 'long_name', "pressures[state] is the external pressure of thermodynamic state 'state'")
            for state_index in range(self.nstates):
                ncvar_temperatures[state_index] = self.states[state_index].pressure / unit.atmospheres

        # TODO: Store other thermodynamic variables store in ThermodynamicState?  Generalize?

        # Systems.
        ncvar_serialized_states = ncgrp_stateinfo.createVariable('systems', str, ('replica',), zlib=True)
        setattr(ncvar_serialized_states, 'long_name', "systems[state] is the serialized OpenMM System corresponding to the thermodynamic state 'state'")
        for state_index in range(self.nstates):
            logger.debug("Serializing state %d..." % state_index)
            serialized = self.states[state_index].system.__getstate__()
            logger.debug("Serialized state is %d B | %.3f KB | %.3f MB" % (len(serialized), len(serialized) / 1024.0, len(serialized) / 1024.0 / 1024.0))
            ncvar_serialized_states[state_index] = serialized
        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger.debug("Serializing thermodynamic states took %.3f s." % elapsed_time)

        return

    def _restore_thermodynamic_states(self, ncfile):
        """
        Restore the thermodynamic states from a NetCDF file.

        """
        logger.debug("Restoring thermodynamic states from NetCDF file...")
        initial_time = time.time()

        # Make sure this NetCDF file contains thermodynamic state information.
        if not 'thermodynamic_states' in ncfile.groups:
            raise Exception("Could not restore thermodynamic states from %s" % self.store_filename)

        # Create a group to store state information.
        ncgrp_stateinfo = ncfile.groups['thermodynamic_states']

        # Get number of states.
        self.nstates = ncgrp_stateinfo.variables['nstates'].getValue()

        # Read state information.
        self.states = list()
        for state_index in range(self.nstates):
            # Populate a new ThermodynamicState object.
            state = ThermodynamicState()
            # Read temperature.
            state.temperature = float(ncgrp_stateinfo.variables['temperatures'][state_index]) * unit.kelvin
            # Read pressure, if present.
            if 'pressures' in ncgrp_stateinfo.variables:
                state.pressure = float(ncgrp_stateinfo.variables['pressures'][state_index]) * unit.atmospheres
            # Reconstitute System object.
            state.system = self.mm.System()
            state.system.__setstate__(str(ncgrp_stateinfo.variables['systems'][state_index]))
            # Store state.
            self.states.append(state)

        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("Restoring thermodynamic states from NetCDF file took %.3f s." % elapsed_time)

        return True

    def _store_dict_in_netcdf(self, ncgrp, options):
        """
        Store the contents of a dict in a NetCDF file.

        Parameters
        ----------
        ncgrp : ncfile.Dataset group
            The group in which to store options.
        options : dict
            The dict to store.

        """
        from utils import typename
        import collections
        for option_name in options.keys():
            # Get option value.
            option_value = options[option_name]
            # If Quantity, strip off units first.
            option_unit = None
            if type(option_value) == unit.Quantity:
                option_unit = option_value.unit
                option_value = option_value / option_unit
            # Store the Python type.
            option_type = type(option_value)
            option_type_name = typename(option_type)
            # Handle booleans
            if type(option_value) == bool:
                option_value = int(option_value)
            # Store the variable.
            if type(option_value) == str:
                logger.debug("Storing option: %s -> %s (type: %s)" % (option_name, option_value, option_type_name))
                ncvar = ncgrp.createVariable(option_name, type(option_value), 'scalar')
                packed_data = np.empty(1, 'O')
                packed_data[0] = option_value
                ncvar[:] = packed_data
                setattr(ncvar, 'type', option_type_name)
            elif isinstance(option_value, collections.Iterable):
                nelements = len(option_value)
                logger.debug("Storing option: %s -> %s (type: %s, array of length %d)" % (option_name, option_value, option_type_name, nelements))
                element_type = type(option_value[0])
                element_type_name = typename(element_type)
                ncgrp.createDimension(option_name, nelements) # unlimited number of iterations
                ncvar = ncgrp.createVariable(option_name, element_type, (option_name,))
                for (i, element) in enumerate(option_value):
                    ncvar[i] = element
                setattr(ncvar, 'type', element_type_name)
            elif option_value is None:
                logger.debug("Storing option: %s -> %s (None)" % (option_name, option_value))
                ncvar = ncgrp.createVariable(option_name, int)
                ncvar.assignValue(0)
                setattr(ncvar, 'type', option_type_name)
            else:
                logger.debug("Storing option: %s -> %s (type: %s, other)" % (option_name, option_value, option_type_name))
                ncvar = ncgrp.createVariable(option_name, type(option_value))
                ncvar.assignValue(option_value)
                setattr(ncvar, 'type', option_type_name)
            if option_unit: setattr(ncvar, 'units', str(option_unit))

        return

    def _restore_dict_from_netcdf(self, ncgrp):
        """
        Restore dict from NetCDF.

        Parameters
        ----------
        ncgrp : netcdf.Dataset group
            The NetCDF group to restore from.

        Returns
        -------
        options : dict
            The restored options as a dict.

        """
        options = dict()

        import numpy
        for option_name in ncgrp.variables.keys():
            # Get NetCDF variable.
            option_ncvar = ncgrp.variables[option_name]
            type_name = getattr(option_ncvar, 'type')
            # Get option value.
            if type_name == 'NoneType':
                option_value = None
                logger.debug("Restoring option: %s -> %s (None)" % (option_name, str(option_value)))
            elif option_ncvar.shape == ():
                option_value = option_ncvar.getValue()
                # Cast to python types.
                if type_name == 'bool':
                    option_value = bool(option_value)
                elif type_name == 'int':
                    option_value = int(option_value)
                elif type_name == 'float':
                    option_value = float(option_value)
                elif type_name == 'str':
                    option_value = str(option_value)
                logger.debug("Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value)))
            elif (option_ncvar.shape[0] >= 0):
                option_value = np.array(option_ncvar[:], eval(type_name))
                # TODO: Deal with values that are actually scalar constants.
                # TODO: Cast to appropriate type
                logger.debug("Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value)))
            else:
                option_value = option_ncvar[0]
                option_value = eval(type_name + '(' + repr(option_value) + ')')
                logger.debug("Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value)))

            # If Quantity, assign unit.
            if hasattr(option_ncvar, 'units'):
                option_unit_name = getattr(option_ncvar, 'units')
                if option_unit_name[0] == '/':
                    option_value = eval(str(option_value) + option_unit_name, unit.__dict__)
                else:
                    option_value = eval(str(option_value) + '*' + option_unit_name, unit.__dict__)
            # Store option.
            options[option_name] = option_value

        return options

    def _store_options(self, ncfile):
        """
        Store run parameters in NetCDF file.

        """

        logger.debug("Storing run parameters in NetCDF file...")

        # Create scalar dimension if not already present.
        if 'scalar' not in ncfile.dimensions:
            ncfile.createDimension('scalar', 1) # scalar dimension

        # Create a group to store state information.
        ncgrp_options = ncfile.createGroup('options')

        # Build dict of options to store.
        options = dict()
        for option_name in self.options_to_store:
            option_value = getattr(self, option_name)
            options[option_name] = option_value

        # Store options.
        self._store_dict_in_netcdf(ncgrp_options, options)

        return

    def _restore_options(self, ncfile):
        """
        Restore run parameters from NetCDF file.

        """

        logger.debug("Attempting to restore options from NetCDF file...")

        # Make sure this NetCDF file contains option information
        if not 'options' in ncfile.groups:
            raise Exception("options not found in NetCDF file.")

        # Find the group.
        ncgrp_options = ncfile.groups['options']

        # Restore options as dict.
        options = self._restore_dict_from_netcdf(ncgrp_options)

        # Set these as attributes.
        for option_name in options.keys():
            setattr(self, option_name, options[option_name])

        # Signal success.
        return True

    def _store_metadata(self, ncfile):
        """
        Store metadata in NetCDF file.

        Parameters
        ----------
        ncfile : netcdf.Dataset
            The NetCDF file in which metadata is to be stored.

        """
        ncgrp = ncfile.createGroup('metadata')
        self._store_dict_in_netcdf(ncgrp, self.metadata)
        return

    def _restore_metadata(self, ncfile):
        """
        Restore metadata from NetCDF file.

        Parameters
        ----------
        ncfile : netcdf.Dataset
            The NetCDF file in which metadata is to be stored.

        """
        self.metadata = None
        if 'metadata' in ncfile.groups:
            ncgrp = ncfile.groups['metadata']
            self.metadata = self._restore_dict_from_netcdf(ncgrp)

    def _resume_from_netcdf(self):
        """
        Resume execution by reading current positions and energies from a NetCDF file.

        """

        # Check to make sure file exists.
        if not os.path.exists(self.store_filename):
            raise Exception("Store file %s does not exist." % self.store_filename)

        # Open NetCDF file for reading
        logger.debug("Reading NetCDF file '%s'..." % self.store_filename)
        ncfile = netcdf.Dataset(self.store_filename, 'r') # netCDF4

        # TODO: Perform sanity check on file before resuming

        # Get current dimensions.
        self.iteration = ncfile.variables['positions'].shape[0] - 1
        self.nstates = ncfile.variables['positions'].shape[1]
        self.natoms = ncfile.variables['positions'].shape[2]
        self.nreplicas = self.nstates
        logger.debug("iteration = %d, nstates = %d, natoms = %d" % (self.iteration, self.nstates, self.natoms))

        # Restore positions.
        self.replica_positions = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['positions'][self.iteration,replica_index,:,:].astype(np.float64).copy()
            positions = unit.Quantity(x, unit.nanometers)
            self.replica_positions.append(positions)

        # Restore box vectors.
        self.replica_box_vectors = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['box_vectors'][self.iteration,replica_index,:,:].astype(np.float64).copy()
            box_vectors = unit.Quantity(x, unit.nanometers)
            self.replica_box_vectors.append(box_vectors)

        # Restore state information.
        self.replica_states = ncfile.variables['states'][self.iteration,:].copy()

        # Restore energies.
        self.u_kl = ncfile.variables['energies'][self.iteration,:,:].copy()

        # On first iteration, we need to do some initialization.
        if self.iteration == 0:
            # Perform sanity checks to see if we should terminate here.
            self._run_sanity_checks()

            # Minimize and equilibrate all replicas.
            self._minimize_and_equilibrate()

            # Compute energies of all alchemical replicas
            self._compute_energies()

            # Show energies.
            if self.show_energies:
                self._show_energies()

            # Re-store initial state.
            # TODO: Sort this logic out.
            #self.ncfile = ncfile
            #self._write_iteration_netcdf()
            #self.ncfile = None

        # Run sanity checks.
        # TODO: Refine this.
        self._run_sanity_checks()
        #self._compute_energies() # recompute energies?
        #self._run_sanity_checks()

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

        if not logger.isEnabledFor(logging.DEBUG):
            return

        # print header
        str_row = "%-24s %16s" % ("reduced potential (kT)", "current state")
        for state_index in range(self.nstates):
            str_row += " state %3d" % state_index
        logger.debug(str_row)

        # print energies in kT
        for replica_index in range(self.nstates):
            str_row = "replica %-16d %16d" % (replica_index, self.replica_states[replica_index])
            for state_index in range(self.nstates):
                u = self.u_kl[replica_index,state_index]
                if (u > 1e6):
                    str_row += "%10.3e" % u
                else:
                    str_row += "%10.1f" % u
            logger.debug(str_row)

        return

    def _compute_trace(self):
        """
        Compute trace for replica ensemble minus log probability.

        Extract timeseries of u_n = - log q(X_n) from store file

        where q(X_n) = \pi_{k=1}^K u_{s_{nk}}(x_{nk})

        with X_n = [x_{n1}, ..., x_{nK}] is the current collection of replica configurations
        s_{nk} is the current state of replica k at iteration n
        u_k(x) is the kth reduced potential

        Returns
        -------
        u_n : numpy array of numpy.float64
        u   _n[n] is -log q(X_n)

        TODO
        ----
        * Later, we should have this quantity computed and stored on the fly in the store file.
        But we may want to do this without breaking backward compatibility.

        """

        # Get current dimensions.
        niterations = self.ncfile.variables['energies'].shape[0]
        nstates = self.ncfile.variables['energies'].shape[1]
        natoms = self.ncfile.variables['energies'].shape[2]

        # Extract energies.
        energies = ncfile.variables['energies']
        u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
        for n in range(niterations):
            u_kln_replica[:,:,n] = energies[n,:,:]

        # Deconvolute replicas
        u_kln = np.zeros([nstates, nstates, niterations], np.float64)
        for iteration in range(niterations):
            state_indices = ncfile.variables['states'][iteration,:]
            u_kln[state_indices,:,iteration] = energies[iteration,:,:]

        # Compute total negative log probability over all iterations.
        u_n = np.zeros([niterations], np.float64)
        for iteration in range(niterations):
            u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

        return u_n

    def _analysis(self):
        """
        Perform online analysis each iteration.

        Every iteration, this will update the estimate of the state relative free energy differences and statistical uncertainties.
        We can additionally request further analysis.

        """

        # Only root node can perform analysis.
        if self.mpicomm and (self.mpicomm.rank != 0): return

        # Determine how many iterations there are data available for.
        replica_states = self.ncfile.variables['states'][:,:]
        u_nkl_replica = self.ncfile.variables['energies'][:,:,:]

        # Determine number of iterations completed.
        number_of_iterations_completed = replica_states.shape[0]
        nstates = replica_states.shape[1]

        # Online analysis can only be performed after a sufficient quantity of data has been collected.
        if (number_of_iterations_completed < self.online_analysis_min_iterations):
            logger.debug("Online analysis will be performed after %d iterations have elapsed." % self.online_analysis_min_iterations)
            self.analysis = None
            return

        # Deconvolute replicas and compute total simulation effective self-energy timeseries.
        u_kln = np.zeros([nstates, nstates, number_of_iterations_completed], np.float32)
        u_n = np.zeros([number_of_iterations_completed], np.float64)
        for iteration in range(number_of_iterations_completed):
            state_indices = replica_states[iteration,:]
            u_n[iteration] = 0.0
            for replica_index in range(nstates):
                state_index = state_indices[replica_index]
                u_n[iteration] += u_nkl_replica[iteration,replica_index,state_index]
                u_kln[state_index,:,iteration] = u_nkl_replica[iteration,replica_index,:]

        # Determine optimal equilibration time, statistical inefficiency, and effectively uncorrelated sample indices.
        from pymbar import timeseries
        [t0, g, Neff_max] = timeseries.detectEquilibration(u_n)
        indices = t0 + timeseries.subsampleCorrelatedData(u_n[t0:], g=g)
        N_k = indices.size * np.ones([nstates], np.int32)

        # Next, analyze with pymbar, initializing with last estimate of free energies.
        from pymbar import MBAR
        if hasattr(self, 'f_k'):
            mbar = MBAR(u_kln[:,:,indices], N_k, initial_f_k=self.f_k)
        else:
            mbar = MBAR(u_kln[:,:,indices], N_k)

        # Cache current free energy estimate to save time in future MBAR solutions.
        self.f_k = mbar.f_k

        # Compute entropy and enthalpy.
        [Delta_f_ij, dDelta_f_ij, Delta_u_ij, dDelta_u_ij, Delta_s_ij, dDelta_s_ij] = mbar.computeEntropyAndEnthalpy()

        # Store analysis summary.
        # TODO: Convert this to an object?
        analysis = dict()
        analysis['equilibration_end'] = t0
        analysis['g'] = g
        analysis['indices'] = indices
        analysis['Delta_f_ij'] = Delta_f_ij
        analysis['dDelta_f_ij'] = dDelta_f_ij
        analysis['Delta_u_ij'] = Delta_u_ij
        analysis['dDelta_u_ij'] = dDelta_u_ij
        analysis['Delta_s_ij'] = Delta_s_ij
        analysis['dDelta_s_ij'] = dDelta_s_ij

        def matrix2str(x):
            """
            Return a print-ready string version of a matrix of numbers.

            Parameters
            ----------
            x : numpy.array of nrows x ncols matrix
               Matrix of numbers to print.

            TODO
            ----
            * Automatically determine optimal spacing

            """
            [nrows, ncols] = x.shape
            str_row = ""
            for i in range(nrows):
                for j in range(ncols):
                    str_row += "%8.3f" % x[i, j]
                str_row += "\n"
            return str_row

        # Print estimate
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("================================================================================")
            logger.debug("Online analysis estimate of free energies:")
            logger.debug("  equilibration end: %d iterations" % t0)
            logger.debug("  statistical inefficiency: %.1f iterations" % g)
            logger.debug("  effective number of uncorrelated samples: %.1f" % Neff_max)
            logger.debug("Reduced free energy (f), enthalpy (u), and entropy (s) differences among thermodynamic states:")
            logger.debug("Delta_f_ij")
            logger.debug(matrix2str(Delta_f_ij))
            logger.debug("dDelta_f_ij")
            logger.debug(matrix2str(dDelta_f_ij))
            logger.debug("Delta_u_ij")
            logger.debug(matrix2str(Delta_u_ij))
            logger.debug("dDelta_u_ij")
            logger.debug(matrix2str(dDelta_u_ij))
            logger.debug("Delta_s_ij")
            logger.debug(matrix2str(Delta_s_ij))
            logger.debug("dDelta_s_ij")
            logger.debug(matrix2str(dDelta_s_ij))
            logger.debug("================================================================================")

        self.analysis = analysis

        return

    def analyze(self):
        """
        Analyze the current simulation and return estimated free energies.

        Returns
        -------
        analysis : dict
           Analysis object containing end of equilibrated region, statistical inefficiency, and free energy differences:

        Keys
        ----
        equilibration_end : int
           The last iteration in the discarded equilibrated region
        g : float
           Estimated statistical inefficiency of production region
        indices : list of int
           Equilibrated, effectively uncorrelated iteration indices used in analysis
        Delta_f_ij : numpy array of nstates x nstates
           Delta_f_ij[i,j] is the free energy difference f_j - f_i in units of kT
        dDelta_f_ij : numpy array of nstates x nstates
           dDelta_f_ij[i,j] is estimated standard error of Delta_f_ij[i,j]
        Delta_u_ij
           Delta_u_ij[i,j] is the reduced enthalpy difference u_j - u_i in units of kT
        dDelta_u_ij
           dDelta_u_ij[i,j] is estimated standard error of Delta_u_ij[i,j]
        Delta_s_ij
           Delta_s_ij[i,j] is the reduced entropic contribution to the free energy difference s_j - s_i in units of kT
        dDelta_s_ij
           dDelta_s_ij[i,j] is estimated standard error of Delta_s_ij[i,j]

        """
        if not self._initialized:
            self._initialize_resume()

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
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideImplicit()
    >>> [system, positions] = [testsystem.system, testsystem.positions]
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * unit.kelvin
    >>> Tmax = 600.0 * unit.kelvin
    >>> nreplicas = 3
    >>> simulation = ParallelTempering(store_filename)
    >>> simulation.create(system, positions, Tmin=Tmin, Tmax=Tmax, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * unit.femtoseconds # set the timestep for integration
    >>> simulation.minimize = False
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    Parallel tempering of alanine dipeptide in explicit solvent at 1 atm.

    >>> # Create alanine dipeptide system
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideExplicit()
    >>> [system, positions] = [testsystem.system, testsystem.positions]
    >>> # Add Monte Carlo barsostat to system (must be same pressure as simulation).
    >>> import simtk.openmm as openmm
    >>> pressure = 1.0 * unit.atmosphere
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * unit.kelvin
    >>> Tmax = 600.0 * unit.kelvin
    >>> nreplicas = 3
    >>> simulation = ParallelTempering(store_filename)
    >>> simulation.create(system, positions, Tmin=Tmin, Tmax=Tmax, pressure=pressure, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * unit.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False # don't minimize first
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    def create(self, system, positions, options=None, Tmin=None, Tmax=None, ntemps=None, temperatures=None, pressure=None, metadata=None):
        """
        Initialize a parallel tempering simulation object.

        Parameters
        ----------
        system : simtk.openmm.System
           the system to simulate
        positions : simtk.unit.Quantity of np natoms x 3 array of units length, or list
           coordinate set(s) for one or more replicas, assigned in a round-robin fashion
        Tmin : simtk.unit.Quantity with units compatible with kelvin, optional, default=None
           min temperature
        Tmax : simtk.unit.Quantity with units compatible with kelvin, optional, default=None
           max temperature
        ntemps : int, optional, default=None
           number of exponentially-spaced temperatures between Tmin and Tmax
        temperatures : list of simtk.unit.Quantity with units compatible with kelvin, optional, default=None
           if specified, this list of temperatures will be used instead of (Tmin, Tmax, ntemps)
        pressure : simtk.unit.Quantity with units compatible with atmospheres, optional, default=None
           if specified, a MonteCarloBarostat will be added (or modified) to perform NPT simulations
        options : dict, optional, default=None
           Options to use for specifying simulation protocol.  Provided keywords will be matched to object variables to replace defaults.

        Notes
        -----
        Either (Tmin, Tmax, ntempts) must all be specified or the list of 'temperatures' must be specified.

        """
        # Create thermodynamic states from temperatures.
        if temperatures is not None:
            logger.info("Using provided temperatures")
            self.temperatures = temperatures
        elif (Tmin is not None) and (Tmax is not None) and (ntemps is not None):
            self.temperatures = [ Tmin + (Tmax - Tmin) * (math.exp(float(i) / float(ntemps-1)) - 1.0) / (math.e - 1.0) for i in range(ntemps) ]
        else:
            raise ValueError("Either 'temperatures' or 'Tmin', 'Tmax', and 'ntemps' must be provided.")

        states = [ ThermodynamicState(system=system, temperature=self.temperatures[i], pressure=pressure) for i in range(ntemps) ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.create(self, states, positions, options=options, metadata=metadata)

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
        logger.debug("Computing energies...")

        if self.mpicomm:
            # MPI implementation

            # Create an integrator and context.
            state = self.states[0]
            integrator = self.mm.VerletIntegrator(self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)

            for replica_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                # Set positions.
                context.setPositions(self.replica_positions[replica_index])
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

            # Create an integrator and context.
            state = self.states[0]
            integrator = self.mm.VerletIntegrator(self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)

            # Compute reduced potentials for all configurations in all states.
            for replica_index in range(self.nstates):
                # Set positions.
                context.setPositions(self.replica_positions[replica_index])
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
        logger.debug("Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy))

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
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideImplicit()
    >>> [reference_system, positions] = [testsystem.system, testsystem.positions]
    >>> # Copy reference system.
    >>> systems = [reference_system for index in range(10)]
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Create reference state.
    >>> reference_state = ThermodynamicState(reference_system, temperature=298.0*unit.kelvin)
    >>> # Create simulation.
    >>> simulation = HamiltonianExchange(store_filename)
    >>> simulation.create(reference_state, systems, positions)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * unit.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False
    >>> # Run simulation.
    >>> simulation.run() #doctest: +ELLIPSIS
    ...

    """

    def create(self, reference_state, systems, positions, options=None, metadata=None):
        """
        Initialize a Hamiltonian exchange simulation object.

        Parameters
        ----------
        reference_state : ThermodynamicState
           reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems : list of simtk.openmm.System
           list of systems to simulate (one per replica)
        positions : simtk.unit.Quantity of np natoms x 3 with units compatible with nanometers
           positions (or a list of positions objects) for initial assignment of replicas (will be used in round-robin assignment)
        options : dict, optional, default=None
           Optional dict to use for specifying simulation protocol. Provided keywords will be matched to object variables to replace defaults.
        metadata : dict, optional, default=None
           metadata to store in a 'metadata' group in store file

        """

        if systems is None:
            states = None
        else:
            # Create thermodynamic states from systems.
            states = [ ThermodynamicState(system=system, temperature=reference_state.temperature, pressure=reference_state.pressure) for system in systems ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.create(self, states, positions, options=options, metadata=metadata)

        # Override title.
        self.title = 'Hamiltonian exchange simulation created using HamiltonianExchange class of repex.py on %s' % time.asctime(time.localtime())

        return

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
