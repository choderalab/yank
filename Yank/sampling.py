#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Sampling algorithms for YANK.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import sys
import copy
import time
import logging
logger = logging.getLogger(__name__)

import numpy as np

from simtk import openmm, unit

from repex import ThermodynamicState
from repex import ReplicaExchange
from repex import MAX_SEED

from alchemy import AbsoluteAlchemicalFactory, AlchemicalState

#=============================================================================================
# Alchemical Modified Hamiltonian exchange class.
#=============================================================================================

class ModifiedHamiltonianExchange(ReplicaExchange):
    """
    A Hamiltonian exchange facility that uses a modified dynamics to introduce Monte Carlo moves to augment Langevin dynamics
    and manages a single System where alchemical states differ only by Context parameters for efficiency.

    DESCRIPTION

    This class provides an implementation of a Hamiltonian exchange simulation based on the HamiltonianExchange facility.
    It modifies the way HamiltonianExchange samples each replica using dynamics by adding Monte Carlo rotation/displacement
    trials to augment sampling by Langevin dynamics.

    EXAMPLES

    >>> # Create reference system.
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideImplicit()
    >>> [reference_system, positions] = [testsystem.system, testsystem.positions]
    >>> ligand_atoms = range(reference_system.getNumParticles())
    >>> # Alchemically modify system.
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Create reference state.
    >>> from repex import ThermodynamicState
    >>> reference_state = ThermodynamicState(reference_system, temperature=298.0*unit.kelvin)
    >>> reference_state.system = factory.alchemically_modified_system
    >>> displacement_sigma = 1.0 * unit.nanometer
    >>> mc_atoms = range(0, reference_system.getNumParticles())
    >>> simulation = ModifiedHamiltonianExchange(store_filename)
    >>> alchemical_states = AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()
    >>> simulation.create(reference_state, alchemical_states, positions, displacement_sigma=displacement_sigma, mc_atoms=mc_atoms)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * unit.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False # don't minimize prior to production
    >>> simulation.number_of_equilibration_iterations = 0 # don't equilibrate prior to production
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    # Options to store.
    options_to_store = ReplicaExchange.options_to_store + ['mc_atoms', 'mc_displacement', 'mc_rotation', 'displacement_sigma', 'displacement_trials_accepted', 'rotation_trials_accepted']

    def create(self, reference_state, alchemical_states, positions, displacement_sigma=None, mc_atoms=None, options=None, metadata=None):
        """
        Initialize a modified Hamiltonian exchange simulation object.

        Parameters
        ----------
        reference_state : ThermodynamicState
           reference state containing all thermodynamic parameters and reference System object'
        alchemical_states : list of AlchemicalState
           list of alchemical states (one per replica)
        positions : simtk.unit.Quantity of numpy natoms x 3 with units length
           positions (or a list of positions objects) for initial assignment of replicas (will be used in round-robin assignment)
        displacement_sigma : simtk.unit.Quantity with units distance
           size of displacement trial for Monte Carlo displacements, if specified (default: 1 nm)
        ligand_atoms : list of int, optional, default=None
           atoms to use for trial displacements for translational and orientational Monte Carlo trials, if specified (all atoms if None)
        options : dict, optional, default=None
           Optional dict to use for specifying simulation options. Provided keywords will be matched to object variables to replace defaults.
        metadata : dict, optional, default=None
           metadata to store in a 'metadata' group in store file

        """

        # If an empty set is specified for mc_atoms, set this to None.
        if mc_atoms is not None:
            if len(mc_atoms) == 0:
                mc_atoms = None

        # Store trial displacement magnitude and atoms to rotate in MC move.
        self.displacement_sigma = 1.0 * unit.nanometer
        if mc_atoms is not None:
            self.mc_atoms = np.array(mc_atoms)
            self.mc_displacement = True
            self.mc_rotation = True
        else:
            self.mc_atoms = None
            self.mc_displacement = False
            self.mc_rotation = False

        self.displacement_trials_accepted = 0 # number of MC displacement trials accepted
        self.rotation_trials_accepted = 0 # number of MC displacement trials accepted

        # Store reference system.
        self.reference_system = copy.deepcopy(reference_state.system)

        # TODO: Form metadata dict.

        # Initialize replica-exchange simlulation.
        states = list()
        for alchemical_state in alchemical_states:
            state = ThermodynamicState(system=self.reference_system, temperature=reference_state.temperature, pressure=reference_state.pressure)
            setattr(state, 'alchemical_state', copy.deepcopy(alchemical_state)) # attach alchemical state
            states.append(state)

        # Initialize replica-exchange simlulation.
        ReplicaExchange.create(self, states, positions, options=options, metadata=metadata)

        # Override title.
        self.title = 'Alchemical Hamiltonian exchange simulation created using HamiltonianExchange class of repex.py on %s' % time.asctime(time.localtime())

        return

    def resume(self, options=None):
        """
        """
        ReplicaExchange.resume(self, options=options)

        #
        # Cache Context and integrator.
        #

        # Use first state as reference state.
        state = self.states[0]

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

        # Alchemical states.
        ncgrp = ncfile.createGroup('alchemical_states')
        alchemical_parameters = self.states[0].alchemical_state.keys()
        for alchemical_parameter in alchemical_parameters:
            ncvar = ncgrp.createVariable(alchemical_parameter, 'f', ('replica',))
            for state_index in range(self.nstates):
                ncvar[state_index] = self.states[state_index].alchemical_state[alchemical_parameter]

        # Systems.
        logger.debug("Serializing system...")
        if 'scalar' not in ncfile.dimensions:
            ncfile.createDimension('scalar', 1) # scalar dimension
        ncvar_serialized_reference_system = ncgrp_stateinfo.createVariable('reference_system', str, ('scalar',), zlib=True)
        setattr(ncvar_serialized_reference_system, 'long_name', "reference is the serialized OpenMM System corresponding to the reference System object")
        ncvar_serialized_reference_system[0] = self.reference_system.__getstate__() # serialize reference system.

        # Report timing information.
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

        # Read thermodynamic state information.
        self.states = list()
        # Read reference system
        self.reference_system = self.mm.System()
        self.reference_system.__setstate__(str(ncgrp_stateinfo.variables['reference_system'][0]))
        # Read other parameters.
        for state_index in range(self.nstates):
            # Populate a new ThermodynamicState object.
            state = ThermodynamicState()
            # Read temperature.
            state.temperature = float(ncgrp_stateinfo.variables['temperatures'][state_index]) * unit.kelvin
            # Read pressure, if present.
            if 'pressures' in ncgrp_stateinfo.variables:
                state.pressure = float(ncgrp_stateinfo.variables['pressures'][state_index]) * unit.atmospheres
            # Read alchemical states.
            state.alchemical_state = AlchemicalState()
            for key in ncfile.groups['alchemical_states'].variables.keys():
                state.alchemical_state[key] = float(ncfile.groups['alchemical_states'].variables[key][state_index])
            # Set System object (which points to reference system).
            state.system = self.reference_system
            # Store state.
            self.states.append(state)

        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("Restoring thermodynamic states from NetCDF file took %.3f s." % elapsed_time)

        return True

    def _cache_context(self):
        """
        Create and cache OpenMM Context and Integrator.

        """

        # Create Context and integrator.
        initial_time = time.time()
        logger.debug("Creating and caching Context and Integrator.")
        state = self.states[0]
        self._integrator = openmm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)
        self._integrator.setRandomNumberSeed(int(np.random.randint(0, MAX_SEED)))
        if self.platform:
            self._context = openmm.Context(state.system, self._integrator, self.platform)
        else:
            self._context = openmm.Context(state.system, self._integrator)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("Context creation took %.3f s." % elapsed_time)

        return

    def _finalize(self):
        """
        Do anything necessary to finish run except close files.

        """
        ReplicaExchange._finalize(self)

        # Clean up cached context and integrator.
        if hasattr(self, 'context'):
            del self._context, self._integrator

        return

    @classmethod
    def _rotation_matrix_from_quaternion(cls, q):
        """
        Compute a 3x3 rotation matrix from a given quaternion (4-vector).

        ARGUMENTS

        q (numpy 4-vector) - quaterion (need not be normalized, zero norm OK)

        RETURNS

        Rq (numpy 3x3 matrix) - orthogonal rotation matrix corresponding to quaternion q

        EXAMPLES

        >>> q = np.array([0.1, 0.2, 0.3, -0.4])
        >>> Rq = ModifiedHamiltonianExchange._rotation_matrix_from_quaternion(q)

        REFERENCES

        [1] http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

        """

        [w,x,y,z] = q
        Nq = (q**2).sum()
        if (Nq > 0.0):
            s = 2.0/Nq
        else:
            s = 0.0
        X = x*s; Y = y*s; Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        Rq = np.matrix([[ 1.0-(yY+zZ),       xY-wZ,        xZ+wY  ],
                           [      xY+wZ,   1.0-(xX+zZ),       yZ-wX  ],
                           [      xZ-wY,        yZ+wX,   1.0-(xX+yY) ]])

        return Rq

    @classmethod
    def _generate_uniform_quaternion(cls):
        """
        Generate a uniform normalized quaternion 4-vector.

        REFERENCES

        [1] K. Shoemake. Uniform random rotations. In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.
        [2] Described briefly here: http://planning.cs.uiuc.edu/node198.html

        TODO: This package might be useful in simplifying: http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

        EXAMPLES

        >>> q = ModifiedHamiltonianExchange._generate_uniform_quaternion()

        """
        u = np.random.rand(3)
        q = np.array([np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),
                         np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
                         np.sqrt(u[0])*np.sin(2*np.pi*u[2]),
                         np.sqrt(u[0])*np.cos(2*np.pi*u[2])]) # uniform quaternion
        return q

    @classmethod
    def propose_displacement(cls, displacement_sigma, original_positions, mc_atoms):
        """
        Make symmetric Gaussian trial displacement of ligand.

        EXAMPLES

        >>> from openmmtools import testsystems
        >>> complex = testsystems.LysozymeImplicit()
        >>> [system, positions] = [complex.system, complex.positions]
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> displacement_sigma = 5.0 * unit.angstroms
        >>> perturbed_positions = ModifiedHamiltonianExchange.propose_displacement(displacement_sigma, positions, ligand_atoms)

        """
        positions_unit = original_positions.unit
        displacement_vector = unit.Quantity(np.random.randn(3) * (displacement_sigma / positions_unit), positions_unit)
        perturbed_positions = copy.deepcopy(original_positions)
        for atom_index in mc_atoms:
            perturbed_positions[atom_index,:] += displacement_vector

        return perturbed_positions

    @classmethod
    def propose_rotation(cls, original_positions, mc_atoms):
        """
        Make a uniform rotation.

        EXAMPLES

        >>> from openmmtools import testsystems
        >>> complex = testsystems.LysozymeImplicit()
        >>> [system, positions] = [complex.system, complex.positions]
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> perturbed_positions = ModifiedHamiltonianExchange.propose_rotation(positions, ligand_atoms)

        """
        positions_unit = original_positions.unit
        xold = original_positions[mc_atoms,:] / positions_unit
        x0 = xold.mean(0) # compute center of geometry of atoms to rotate
        # Generate a random quaterionion (uniform element of of SO(3)) using algorithm from:
        q = cls._generate_uniform_quaternion()
        # Create rotation matrix based on this quaterion.
        Rq = cls._rotation_matrix_from_quaternion(q)
        # Apply rotation.
        xnew = (Rq * np.matrix(xold - x0).T).T + x0
        perturbed_positions = copy.deepcopy(original_positions)
        perturbed_positions[mc_atoms,:] = unit.Quantity(xnew, positions_unit)

        return perturbed_positions

    @classmethod
    def randomize_ligand_position(cls, positions, receptor_atom_indices, ligand_atom_indices, sigma, close_cutoff):
        """
        Draw a new ligand position with minimal overlap.

        EXAMPLES

        >>> from openmmtools import testsystems
        >>> complex = testsystems.LysozymeImplicit()
        >>> [system, positions] = [complex.system, complex.positions]
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> sigma = 30.0 * unit.angstroms
        >>> close_cutoff = 3.0 * unit.angstroms
        >>> perturbed_positions = ModifiedHamiltonianExchange.randomize_ligand_position(positions, receptor_atoms, ligand_atoms, sigma, close_cutoff)

        """

        # Convert to dimensionless positions.
        positions_unit = positions.unit
        x = positions / positions_unit

        # Compute ligand center of geometry.
        x0 = x[ligand_atom_indices,:].mean(0)

        # Try until we have a non-overlapping ligand conformation.
        success = False
        nattempts = 0
        while (not success):
            # Choose a receptor atom to center ligand on.
            receptor_atom_index = receptor_atom_indices[np.random.randint(0, len(receptor_atom_indices))]

            # Randomize orientation of ligand.
            q = cls._generate_uniform_quaternion()
            Rq = cls._rotation_matrix_from_quaternion(q)
            x[ligand_atom_indices,:] = (Rq * np.matrix(x[ligand_atom_indices,:] - x0).T).T + x0

            # Choose a random displacement vector.
            xdisp = (sigma / positions_unit) * np.random.randn(3)

            # Translate ligand center to receptor atom plus displacement vector.
            for atom_index in ligand_atom_indices:
                x[atom_index,:] += xdisp[:] + (x[receptor_atom_index,:] - x0[:])

            # Compute min distance from ligand atoms to receptor atoms.
            y = x[receptor_atom_indices,:]
            mindist = 999.0
            success = True
            for ligand_atom_index in ligand_atom_indices:
                z = x[ligand_atom_index,:]
                distances = np.sqrt(((y - np.tile(z, (y.shape[0], 1)))**2).sum(1)) # distances[i] is the distance from the centroid to particle i
                mindist = distances.min()
                if (mindist < (close_cutoff/positions_unit)):
                    success = False
            nattempts += 1

        positions = unit.Quantity(x, positions_unit)
        return positions

    def _minimize_replica(self, replica_index):
        """
        Minimize the specified replica.

        """
        # Create and cache Integrator and Context if needed.
        if not hasattr(self, '_context'):
            self._cache_context()

        context = self._context

        # Retrieve thermodynamic state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state

        # Set alchemical state.
        AbsoluteAlchemicalFactory.perturbContext(context, state.alchemical_state)

        # Set box vectors.
        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        # Set positions.
        positions = self.replica_positions[replica_index]
        context.setPositions(positions)

        logger.debug("Replica %5d/%5d: initial energy %8.3f kT", replica_index, self.nstates, state.reduced_potential(positions, box_vectors=box_vectors, context=context))

        # Minimize energy.
        self.mm.LocalEnergyMinimizer.minimize(context, self.minimize_tolerance, self.minimize_max_iterations)

        # Store final positions
        positions = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
        self.replica_positions[replica_index] = positions

        logger.debug("Replica %5d/%5d: final   energy %8.3f kT", replica_index, self.nstates, state.reduced_potential(positions, box_vectors=box_vectors, context=context))

        return

    def _propagate_replica(self, replica_index):
        """
        Attempt a Monte Carlo rotation/translation move followed by dynamics.

        """

        # Create and cache Integrator and Context if needed.
        if not hasattr(self, '_context'):
            self._cache_context()

        # Retrieve state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state

        # Retrieve cached integrator and context.
        integrator = self._integrator
        context = self._context

        # Set thermodynamic parameters for this state.
        integrator.setTemperature(state.temperature)
        integrator.setRandomNumberSeed(int(np.random.randint(0, MAX_SEED)))
        if state.temperature and state.pressure:
            forces = { state.system.getForce(index).__class__.__name__ : state.system.getForce(index) for index in range(state.system.getNumForces()) }

            if 'MonteCarloAnisotropicBarostat' in forces:
                raise Exception('MonteCarloAnisotropicBarostat is unsupported.')

            if 'MonteCarloBarostat' in forces:
                barostat = forces['MonteCarloBarostat']
                # Set temperature and pressure.
                barostat.setTemperature(state.temperature)
                barostat.setDefaultPressure(state.pressure)
                context.setParameter(barostat.Pressure(), state.pressure) # must be set in context
                barostat.setRandomNumberSeed(int(np.random.randint(0, MAX_SEED)))

        # Set alchemical state.
        AbsoluteAlchemicalFactory.perturbContext(context, state.alchemical_state)

        #
        # Attempt a Monte Carlo rotation/translation move.
        #

        # Attempt gaussian trial displacement with stddev 'self.displacement_sigma'.
        # TODO: Can combine these displacements and/or use cached potential energies to speed up this phase.
        # TODO: Break MC displacement and rotation into member functions and write separate unit tests.
        if self.mc_displacement and (self.mc_atoms is not None):
            initial_time = time.time()
            # Store original positions and energy.
            original_positions = self.replica_positions[replica_index]
            u_old = state.reduced_potential(original_positions, context=context)
            # Make symmetric Gaussian trial displacement of ligand.
            perturbed_positions = self.propose_displacement(self.displacement_sigma, original_positions, self.mc_atoms)
            u_new = state.reduced_potential(perturbed_positions, context=context)
            # Accept or reject with Metropolis criteria.
            du = u_new - u_old
            if (du <= 0.0) or (np.random.rand() < np.exp(-du)):
                self.displacement_trials_accepted += 1
                self.replica_positions[replica_index] = perturbed_positions
            #print "translation du = %f (%d)" % (du, self.displacement_trials_accepted)
            # Print timing information.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            self.displacement_trial_time += elapsed_time

        # Attempt random rotation of ligand.
        if self.mc_rotation and (self.mc_atoms is not None):
            initial_time = time.time()
            # Store original positions and energy.
            original_positions = self.replica_positions[replica_index]
            u_old = state.reduced_potential(original_positions, context=context)
            # Compute new potential.
            perturbed_positions = self.propose_rotation(original_positions, self.mc_atoms)
            u_new = state.reduced_potential(perturbed_positions, context=context)
            du = u_new - u_old
            if (du <= 0.0) or (np.random.rand() < np.exp(-du)):
                self.rotation_trials_accepted += 1
                self.replica_positions[replica_index] = perturbed_positions
            #print "rotation du = %f (%d)" % (du, self.rotation_trials_accepted)
            # Accumulate timing information.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            self.rotation_trial_time += elapsed_time

        #
        # Propagate with dynamics.
        #

        start_time = time.time()

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

        # Compute timing.
        end_time = time.time()
        elapsed_time = end_time - start_time
        positions_elapsed_time = setpositions_end_time - start_time
        velocities_elapsed_time = setvelocities_end_time - setpositions_end_time
        integrator_elapsed_time = integrator_end_time - setvelocities_end_time
        getstate_elapsed_time = getstate_end_time - integrator_end_time
        logger.debug("Replica %d/%d: integrator elapsed time %.3f s (positions %.3f s | velocities %.3f s | integrate+getstate %.3f s)." % (replica_index, self.nreplicas, elapsed_time, positions_elapsed_time, velocities_elapsed_time, integrator_elapsed_time+getstate_elapsed_time))

        return elapsed_time

    def _propagate_replicas(self):
        # Reset statistics for MC trial times.
        self.displacement_trial_time = 0.0
        self.rotation_trial_time = 0.0
        self.displacement_trials_accepted = 0
        self.rotation_trials_accepted = 0

        # Propagate replicas.
        ReplicaExchange._propagate_replicas(self)

        # Print summary statistics.
        # TODO: Streamline this idiom.
        if self.mpicomm:
            # MPI
            from mpi4py import MPI
            if self.mc_displacement and (self.mc_atoms is not None):
                self.displacement_trials_accepted = self.mpicomm.reduce(self.displacement_trials_accepted, op=MPI.SUM)
                self.displacement_trial_time = self.mpicomm.reduce(self.displacement_trial_time, op=MPI.SUM)
                if self.mpicomm.rank == 0:
                    logger.debug("Displacement MC trial times consumed %.3f s aggregate (%d accepted)" % (self.displacement_trial_time, self.displacement_trials_accepted))

            if self.mc_rotation and (self.mc_atoms is not None):
                self.rotation_trials_accepted = self.mpicomm.reduce(self.rotation_trials_accepted, op=MPI.SUM)
                self.rotation_trial_time = self.mpicomm.reduce(self.rotation_trial_time, op=MPI.SUM)
                if self.mpicomm.rank == 0:
                    logger.debug("Rotation MC trial times consumed %.3f s aggregate (%d accepted)" % (self.rotation_trial_time, self.rotation_trials_accepted))
        else:
            # SERIAL
            if self.mc_displacement and (self.mc_atoms is not None):
                logger.debug("Displacement MC trial times consumed %.3f s aggregate (%d accepted)" % (self.displacement_trial_time, self.displacement_trials_accepted))
            if self.mc_rotation and (self.mc_atoms is not None):
                logger.debug("Rotation MC trial times consumed %.3f s aggregate (%d accepted)" % (self.rotation_trial_time, self.rotation_trials_accepted))

        return

    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        TODO

        * We have to re-order Context initialization if we have variable box volume
        * Parallel implementation

        """

        # Create and cache Integrator and Context if needed.
        if not hasattr(self, '_context'):
            self._cache_context()

        logger.debug("Computing energies...")
        start_time = time.time()

        # Retrieve context.
        context = self._context

        if self.mpicomm:
            # MPI version.

            # Compute energies for this node's share of states.
            for state_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                # Set alchemical state.
                AbsoluteAlchemicalFactory.perturbContext(context, self.states[state_index].alchemical_state)
                for replica_index in range(self.nstates):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_positions[replica_index], box_vectors=self.replica_box_vectors[replica_index], context=context)

            # Send final energies to all nodes.
            energies_gather = self.mpicomm.allgather(self.u_kl[:,self.mpicomm.rank:self.nstates:self.mpicomm.size])
            for state_index in range(self.nstates):
                source = state_index % self.mpicomm.size # node with trajectory data
                index = state_index // self.mpicomm.size # index within trajectory batch
                self.u_kl[:,state_index] = energies_gather[source][:,index]

        else:
            # Serial version.
            for state_index in range(self.nstates):
                # Set alchemical state.
                AbsoluteAlchemicalFactory.perturbContext(context, self.states[state_index].alchemical_state)
                for replica_index in range(self.nstates):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_positions[replica_index], box_vectors=self.replica_box_vectors[replica_index], context=context)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy= elapsed_time / float(self.nstates)**2
        logger.debug("Time to compute all energies %.3f s (%.3f per energy calculation)." % (elapsed_time, time_per_energy))

        return

    def _display_citations(self):
        ReplicaExchange._display_citations(self)

        yank_citations = """\
        Chodera JD, Shirts MR, Wang K, Friedrichs MS, Eastman P, Pande VS, and Branson K. YANK: An extensible platform for GPU-accelerated free energy calculations. In preparation."""

        print yank_citations
        print ""

        return
