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

import numpy
import numpy.random

import simtk.unit as units
import simtk.openmm as openmm

from repex import ThermodynamicState
from repex import HamiltonianExchange, ReplicaExchange

#=============================================================================================
# Modified Hamiltonian exchange class.
#=============================================================================================

class ModifiedHamiltonianExchange(HamiltonianExchange):
    """
    A Hamiltonian exchange facility that uses a modified dynamics to introduce Monte Carlo moves to augment Langevin dynamics.

    DESCRIPTION

    This class provides an implementation of a Hamiltonian exchange simulation based on the HamiltonianExchange facility.
    It modifies the way HamiltonianExchange samples each replica using dynamics by adding Monte Carlo rotation/displacement
    trials to augment sampling by Langevin dynamics.

    EXAMPLES

    >>> # Create reference system.
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
    >>> from repex import ThermodynamicState
    >>> reference_state = ThermodynamicState(reference_system, temperature=298.0*units.kelvin)
    >>> displacement_sigma = 1.0 * units.nanometer
    >>> mc_atoms = range(0, reference_system.getNumParticles())
    >>> simulation = ModifiedHamiltonianExchange(store_filename)
    >>> simulation.create(reference_state, systems, positions, displacement_sigma=displacement_sigma, mc_atoms=mc_atoms)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False # don't minimize prior to production
    >>> simulation.number_of_equilibration_iterations = 0 # don't equilibrate prior to production
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    # Options to store.
    options_to_store = HamiltonianExchange.options_to_store + ['mc_displacement', 'mc_rotation', 'displacement_sigma'] # TODO: Add mc_atoms

    def create(self, reference_state, systems, positions, displacement_sigma=None, mc_atoms=None, options=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize a modified Hamiltonian exchange simulation object.

        Parameters
        ----------
        reference_state : ThermodynamicState
           reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems : list of simtk.openmm.System
           list of systems to simulate (one per replica)
        positions : simtk.unit.Quantity of numpy natoms x 3 with units length
           positions (or a list of positions objects) for initial assignment of replicas (will be used in round-robin assignment)
        displacement_sigma : simtk.unit.Quantity with units distance
           size of displacement trial for Monte Carlo displacements, if specified (default: 1 nm)
        ligand_atoms : list of int, optional, default=None
           atoms to use for trial displacements for translational and orientational Monte Carlo trials, if specified (all atoms if None)
        options : dict, optional, default=None
           Optional dict to use for specifying simulation options. Provided keywords will be matched to object variables to replace defaults.

        """

        # If an empty set is specified for mc_atoms, set this to None.
        if mc_atoms is not None:
            if len(mc_atoms) == 0:
                mc_atoms = None

        # Store trial displacement magnitude and atoms to rotate in MC move.
        self.displacement_sigma = 1.0 * units.nanometer
        if mc_atoms is not None:
            self.mc_atoms = numpy.array(mc_atoms)
            self.mc_displacement = True
            self.mc_rotation = True
        else:
            self.mc_atoms = None
            self.mc_displacement = False
            self.mc_rotation = False

        self.displacement_trials_accepted = 0 # number of MC displacement trials accepted
        self.rotation_trials_accepted = 0 # number of MC displacement trials accepted

        # Initialize replica-exchange simlulation.
        HamiltonianExchange.create(self, reference_state, systems, positions, options=options, metadata=metadata)

        # Override title.
        self.title = 'Modified Hamiltonian exchange simulation created using HamiltonianExchange class of repex.py on %s' % time.asctime(time.localtime())

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

        >>> q = numpy.array([0.1, 0.2, 0.3, -0.4])
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
        Rq = numpy.matrix([[ 1.0-(yY+zZ),       xY-wZ,        xZ+wY  ],
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
        u = numpy.random.rand(3)
        q = numpy.array([numpy.sqrt(1-u[0])*numpy.sin(2*numpy.pi*u[1]),
                         numpy.sqrt(1-u[0])*numpy.cos(2*numpy.pi*u[1]),
                         numpy.sqrt(u[0])*numpy.sin(2*numpy.pi*u[2]),
                         numpy.sqrt(u[0])*numpy.cos(2*numpy.pi*u[2])]) # uniform quaternion
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
        >>> displacement_sigma = 5.0 * units.angstroms
        >>> perturbed_positions = ModifiedHamiltonianExchange.propose_displacement(displacement_sigma, positions, ligand_atoms)

        """
        positions_unit = original_positions.unit
        displacement_vector = units.Quantity(numpy.random.randn(3) * (displacement_sigma / positions_unit), positions_unit)
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
        xnew = (Rq * numpy.matrix(xold - x0).T).T + x0
        perturbed_positions = copy.deepcopy(original_positions)
        perturbed_positions[mc_atoms,:] = units.Quantity(xnew, positions_unit)

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
        >>> sigma = 30.0 * units.angstroms
        >>> close_cutoff = 3.0 * units.angstroms
        >>> perturbed_positions = ModifiedHamiltonianExchange.randomize_ligand_position(positions, receptor_atoms, ligand_atoms, sigma, close_cutoff)

        """

        # Convert to dimensionless positions.
        unit = positions.unit
        x = positions / unit

        # Compute ligand center of geometry.
        x0 = x[ligand_atom_indices,:].mean(0)

        import numpy.random

        # Try until we have a non-overlapping ligand conformation.
        success = False
        nattempts = 0
        while (not success):
            # Choose a receptor atom to center ligand on.
            receptor_atom_index = receptor_atom_indices[numpy.random.randint(0, len(receptor_atom_indices))]

            # Randomize orientation of ligand.
            q = cls._generate_uniform_quaternion()
            Rq = cls._rotation_matrix_from_quaternion(q)
            x[ligand_atom_indices,:] = (Rq * numpy.matrix(x[ligand_atom_indices,:] - x0).T).T + x0

            # Choose a random displacement vector.
            xdisp = (sigma / unit) * numpy.random.randn(3)

            # Translate ligand center to receptor atom plus displacement vector.
            for atom_index in ligand_atom_indices:
                x[atom_index,:] += xdisp[:] + (x[receptor_atom_index,:] - x0[:])

            # Compute min distance from ligand atoms to receptor atoms.
            y = x[receptor_atom_indices,:]
            mindist = 999.0
            success = True
            for ligand_atom_index in ligand_atom_indices:
                z = x[ligand_atom_index,:]
                distances = numpy.sqrt(((y - numpy.tile(z, (y.shape[0], 1)))**2).sum(1)) # distances[i] is the distance from the centroid to particle i
                mindist = distances.min()
                if (mindist < (close_cutoff/unit)):
                    success = False
            nattempts += 1

        positions = units.Quantity(x, unit)
        return positions

    def _propagate_replica(self, replica_index):
        """
        Attempt a Monte Carlo rotation/translation move.

        """

        # Attempt a Monte Carlo rotation/translation move.
        import numpy.random

        # Retrieve state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state

        # Retrieve integrator and context from thermodynamic state.
        integrator = state._integrator
        context = state._context

        # Attempt gaussian trial displacement with stddev 'self.displacement_sigma'.
        # TODO: Can combine these displacements and/or use cached potential energies to speed up this phase.
        # TODO: Break MC displacement and rotation into member functions and write separate unit tests.
        if self.mc_displacement and (self.mc_atoms is not None):
            initial_time = time.time()
            # Store original positions and energy.
            original_positions = self.replica_positions[replica_index]
            u_old = state.reduced_potential(original_positions)
            # Make symmetric Gaussian trial displacement of ligand.
            perturbed_positions = self.propose_displacement(self.displacement_sigma, original_positions, self.mc_atoms)
            u_new = state.reduced_potential(perturbed_positions)
            # Accept or reject with Metropolis criteria.
            du = u_new - u_old
            if (du <= 0.0) or (numpy.random.rand() < numpy.exp(-du)):
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
            u_old = state.reduced_potential(original_positions)
            # Compute new potential.
            perturbed_positions = self.propose_rotation(original_positions, self.mc_atoms)
            u_new = state.reduced_potential(perturbed_positions)
            du = u_new - u_old
            if (du <= 0.0) or (numpy.random.rand() < numpy.exp(-du)):
                self.rotation_trials_accepted += 1
                self.replica_positions[replica_index] = perturbed_positions
            #print "rotation du = %f (%d)" % (du, self.rotation_trials_accepted)
            # Accumulate timing information.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            self.rotation_trial_time += elapsed_time

        # Propagate with Langevin dynamics as usual.
        HamiltonianExchange._propagate_replica(self, replica_index)

        return

    def _propagate_replicas(self):
        # Reset statistics for MC trial times.
        self.displacement_trial_time = 0.0
        self.rotation_trial_time = 0.0
        self.displacement_trials_accepted = 0
        self.rotation_trials_accepted = 0

        # Propagate replicas.
        HamiltonianExchange._propagate_replicas(self)

        # Print summary statistics.
        if (self.mc_displacement or self.mc_rotation):
            if self.mpicomm:
                from mpi4py import MPI
                self.displacement_trials_accepted = self.mpicomm.reduce(self.displacement_trials_accepted, op=MPI.SUM)
                self.rotation_trials_accepted = self.mpicomm.reduce(self.rotation_trials_accepted, op=MPI.SUM)
            if self.verbose:
                total_mc_time = self.displacement_trial_time + self.rotation_trial_time
                print "Rotation and displacement MC trial times consumed %.3f s (%d translation | %d rotation accepted)" % (total_mc_time, self.displacement_trials_accepted, self.rotation_trials_accepted)

        return

    def _display_citations(self):
        HamiltonianExchange._display_citations(self)

        yank_citations = """\
        Chodera JD, Shirts MR, Wang K, Friedrichs MS, Eastman P, Pande VS, and Branson K. YANK: An extensible platform for GPU-accelerated free energy calculations. In preparation."""

        print yank_citations
        print ""

        return
