#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
YANK

A toolkit for GPU-accelerated ligand binding alchemical free energy calculations.

Portions of this code copyright (c) 2009-2011 University of California, Berkeley, 
Vertex Pharmaceuticals, Stanford University, University of Virginia, and the Authors.

@author John D. Chodera <jchodera@gmail.com>
@author Kim Branson <kim.branson@gmail.com>
@author Imran Haque <ihaque@gmail.com>
@author Michael Shirts <mrshirts@gmail.com>

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

* Make vacuum calculation optional.
* Handle complex_coordinates argument in Yank more intelligently if different kinds of input are provided.
  Currently crashes if a Quantity is provided rather than a list of coordinate sets.
* Add offline analysis facility for easy post-simulation analysis.
* Store Yank object information in NetCDF file?
* If resuming, skip creation of alchemical intermediates (since they are now read from NetCDF file).
* Have ligand in vacuum and solvent run simultaneously for MPI run if there are sufficient CPUs.
* Remove replication in run() and run_mpi()
* Add support for compressed NetCDF4 files.
* Move imports to happen just before needed, to speed up quick return of error messages early on in file reading.
* Add an 'automatic' mode for automated analysis and adjustment of iterations to give precision target.
* Add automated ligand and receptor parameterization support.
* Add automatic solvation (implicit and explicit) support.
* Add support for softening some receptor atoms.
* Add support for imposing restraints on the noninteracting system.

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

from alchemy import AbsoluteAlchemicalFactory
from thermodynamics import ThermodynamicState
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
    >>> import testsystems
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
    >>> displacement_sigma = 1.0 * units.nanometer
    >>> mc_atoms = range(0, reference_system.getNumParticles())
    >>> simulation = ModifiedHamiltonianExchange(reference_state, systems, coordinates, store_filename, displacement_sigma=displacement_sigma, mc_atoms=mc_atoms)
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

    def __init__(self, reference_state, systems, coordinates, store_filename, displacement_sigma=None, mc_atoms=None, protocol=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize a modified Hamiltonian exchange simulation object.

        ARGUMENTS

        reference_state (ThermodynamicState) - reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems (list of simtk.openmm.System) - list of systems to simulate (one per replica)
        coordinates (simtk.unit.Quantity of numpy natoms x 3 with units length) -  coordinates (or a list of coordinates objects) for initial assignment of replicas (will be used in round-robin assignment)
        store_filename (string) - name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        displacement_sigma (simtk.unit.Quantity with units distance) - size of displacement trial for Monte Carlo displacements, if specified (default: 1 nm)
        ligand_atoms (list of int) - atoms to use for trial displacements for translational and orientational Monte Carlo trials, if specified (default: all atoms)
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.
        mm (simtk.openmm implementation) - Implementation of OpenMM API to use.
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)        

        """

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
        HamiltonianExchange.__init__(self, reference_state, systems, coordinates, store_filename, protocol=protocol, mm=mm, mpicomm=mpicomm, metadata=metadata)

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
        >>> numpy.linalg.norm(Rq, 2)
        1.0

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

        """
        u = numpy.random.rand(3)
        q = numpy.array([numpy.sqrt(1-u[0])*numpy.sin(2*numpy.pi*u[1]),
                         numpy.sqrt(1-u[0])*numpy.cos(2*numpy.pi*u[1]),
                         numpy.sqrt(u[0])*numpy.sin(2*numpy.pi*u[2]),
                         numpy.sqrt(u[0])*numpy.cos(2*numpy.pi*u[2])]) # uniform quaternion
        return q

    @classmethod
    def propose_displacement(cls, displacement_sigma, original_coordinates, mc_atoms):
        """
        # Make symmetric Gaussian trial displacement of ligand.

        """
        coordinates_unit = original_coordinates.unit
        displacement_vector = units.Quantity(numpy.random.randn(3) * (displacement_sigma / coordinates_unit), coordinates_unit)
        perturbed_coordinates = copy.deepcopy(original_coordinates)
        for atom_index in mc_atoms:
            perturbed_coordinates[atom_index,:] += displacement_vector
        
        return perturbed_coordinates

    @classmethod
    def propose_rotation(cls, original_coordinates, mc_atoms):
        """
        Make a uniform rotation.

        """
        coordinates_unit = original_coordinates.unit
        xold = original_coordinates[mc_atoms,:] / coordinates_unit
        x0 = xold.mean(0) # compute center of geometry of atoms to rotate
        # Generate a random quaterionion (uniform element of of SO(3)) using algorithm from:
        q = cls._generate_uniform_quaternion()
        # Create rotation matrix based on this quaterion.
        Rq = cls._rotation_matrix_from_quaternion(q)
        # Apply rotation.
        xnew = (Rq * numpy.matrix(xold - x0).T).T + x0
        perturbed_coordinates = copy.deepcopy(original_coordinates)
        perturbed_coordinates[mc_atoms,:] = units.Quantity(xnew, coordinates_unit)
        
        return perturbed_coordinates

    @classmethod
    def randomize_ligand_position(cls, coordinates, receptor_atom_indices, ligand_atom_indices, sigma, close_cutoff):
        """
        Draw a new ligand position with minimal overlap.

        """

        # Convert to dimensionless coordinates.
        unit = coordinates.unit
        x = coordinates / unit
        
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
            
        coordinates = units.Quantity(x, unit)            
        return coordinates
        
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
            # Store original coordinates and energy.
            original_coordinates = self.replica_coordinates[replica_index]
            u_old = state.reduced_potential(original_coordinates)
            # Make symmetric Gaussian trial displacement of ligand.
            perturbed_coordinates = self.propose_displacement(self.displacement_sigma, original_coordinates, self.mc_atoms)
            u_new = state.reduced_potential(perturbed_coordinates)
            # Accept or reject with Metropolis criteria.
            du = u_new - u_old
            if (du <= 0.0) or (numpy.random.rand() < numpy.exp(-du)):
                self.displacement_trials_accepted += 1
                self.replica_coordinates[replica_index] = perturbed_coordinates
            #print "translation du = %f (%d)" % (du, self.displacement_trials_accepted)
            # Print timing information.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            self.displacement_trial_time += elapsed_time            
            
        # Attempt random rotation of ligand.
        if self.mc_rotation and (self.mc_atoms is not None):
            initial_time = time.time()
            # Store original coordinates and energy.
            original_coordinates = self.replica_coordinates[replica_index]
            u_old = state.reduced_potential(original_coordinates)
            # Compute new potential.
            perturbed_coordinates = self.propose_rotation(original_coordinates, self.mc_atoms)
            u_new = state.reduced_potential(perturbed_coordinates)
            du = u_new - u_old
            if (du <= 0.0) or (numpy.random.rand() < numpy.exp(-du)):
                self.rotation_trials_accepted += 1
                self.replica_coordinates[replica_index] = perturbed_coordinates
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
        
#=============================================================================================
# YANK class
#=============================================================================================

class Yank(object):
    """
    A class for computing receptor-ligand binding free energies through alchemical transformations.

    Options that can be set after initializaton but before run() has been called:

    data_directory (string) - destination for datafiles (default: .)
    online_analysis (boolean) - if True, will analyze data as simulations proceed (default: False)
    temperature (simtk.unit.Quantity compatible with kelvin) - temperature of simulation (default: 298 K)
    pressure (simtk.unit.Quantity compatible with atmosphere) - pressure of simulation if explicit solvent (default: 1 atm)
    niterations (integer) - number of iterations to run (default: 10)
    solvent_protocol (list of AlchemicalState) - protocol to use for turning off ligand in solvent
    complex_protocol (list of AlchemicalState) - protocol to use for turning off ligand in complex

    """

    def __init__(self, receptor=None, ligand=None, complex=None, complex_coordinates=None, output_directory=None, verbose=False):
        """
        Create a YANK binding free energy calculation object.

        ARGUMENTS
        
        receptor (simtk.openmm.System) - the receptor OpenMM system (receptor with implicit solvent forces)
        ligand (simtk.openmm.System) - the ligand OpenMM system (ligand with implicit solvent forces)
        complex_coordinates (simtk.unit.Quantity of coordinates, or list thereof) - coordinates for the complex to initialize replicas with, either a single snapshot or a list of snapshots
        output_directory (String) - the output directory to write files (default: current directory)

        OPTIONAL ARGUMENTS
        
        verbose (boolean) - if True, will give verbose output
        complex (simtk.openmm.System) - specified System will be used instead of concatenating receptor + ligand (default: None)

        NOTES

        * Explicit solvent is not yet supported.
        
        TODO

        * Automatically use a temporary directory, or prepend a unique string, for each set of output files?

        """

        # Check arguments.
        if (receptor is None) or (ligand is None):
            raise Exception("Yank must be initialized with receptor and ligand System objects.")
        if complex_coordinates is None:
            raise Exception("Yank must be initialized with at least one set of complex coordinates.")

        # Mark as not yet initialized.
        self._initialized = False

        # Set defaults for free energy calculation and protocol.
        self.verbose = False # Don't show verbose output
        self.online_analysis = False # if True, simulation will be analyzed each iteration
        self.temperature = 298.0 * units.kelvin # simulation temperature
        self.pressure = 1.0 * units.atmosphere # simulation pressure (for explicit solvent)
        self.niterations = 2000 # number of production iterations
        self.perform_sanity_checks = True # perform some sanity checks to ensure correct results
        self.platform = None # don't specify a platform

        # Store deep copies of receptor and ligand.
        self.receptor = copy.deepcopy(receptor) 
        self.ligand = copy.deepcopy(ligand)

        # Don't randomize ligand by default.
        self.randomize_ligand = False

        # Create complex and store atom indices.
        self.complex = copy.deepcopy(complex)

        # Set up output directory.
        if output_directory is None:
            output_directory = os.getcwd()
        self.output_directory = output_directory

        # DEBUG
        #if self.verbose: print "receptor has %d atoms; ligand has %d atoms" % (self.receptor.getNumParticles(), self.ligand.getNumParticles())

        # TODO: Determine whether system is periodic.
        self.is_periodic = False
        
        # Select default protocols for alchemical transformation.
        self.vacuum_protocol = AbsoluteAlchemicalFactory.defaultVacuumProtocol()
        if self.is_periodic:
            self.solvent_protocol = AbsoluteAlchemicalFactory.defaultSolventProtocolExplicit()
            self.complex_protocol = AbsoluteAlchemicalFactory.defaultComplexProtocolExplicit() 
        else:
            self.solvent_protocol = AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()
            self.complex_protocol = AbsoluteAlchemicalFactory.defaultComplexProtocolImplicit() 
        
        # Determine atom indices in complex.
        self.receptor_atoms = range(0, self.receptor.getNumParticles()) # list of receptor atoms
        self.ligand_atoms = range(self.receptor.getNumParticles(), self.complex.getNumParticles()) # list of ligand atoms

        # Monte Carlo displacement standard deviation for encouraging rapid decorrelation of ligand in the annihilated/decoupled state.
        # TODO: Replace this by a more sophisticated MCMC move set that includes dynamics for replica propagation.
        self.displacement_sigma = 1.0 * units.nanometers # attempt to displace ligand by this stddev will be made each iteration

        # Store complex coordinates.
        # TODO: Make sure each coordinate set is packaged as a numpy array.  For now, we require the user to pass in a list of Quantity objects that contain numpy arrays.
        # TODO: Switch to a special Coordinate object (with box size) to support explicit solvent simulations?
        self.complex_coordinates = copy.deepcopy(complex_coordinates)

        # Type of restraints requested.
        #self.restraint_type = 'flat-bottom' # default to a flat-bottom restraint between the ligand and receptor
        self.restraint_type = 'harmonic' # default to a single harmonic restraint between the ligand and receptor

        return

    def _initialize(self):
        """
        """
        self._initialized = True

        # TODO: Run some sanity checks on arguments to see if we can initialize a valid simulation.

        # Turn off pressure if we aren't simulating a periodic system.
        if not self.is_periodic:
            self.pressure = None

        # Extract ligand coordinates.
        self.ligand_coordinates = [ coordinates[self.ligand_atoms,:] for coordinates in self.complex_coordinates ]

        # TODO: Pack up a 'protocol' for modified Hamiltonian exchange simulations. Use this instead in setting up simulations.
        self.protocol = dict()
        self.protocol['number_of_equilibration_iterations'] = 1
        self.protocol['number_of_iterations'] = self.niterations
        self.protocol['verbose'] = self.verbose
        self.protocol['timestep'] = 2.0 * units.femtoseconds
        self.protocol['collision_rate'] = 5.0 / units.picoseconds
        self.protocol['minimize'] = True
        self.protocol['show_mixing_statistics'] = False # this causes slowdown with iteration and should not be used for production

        # DEBUG
        self.protocol['number_of_equilibration_iterations'] = 0
        self.protocol['minimize'] = False
        self.protocol['minimize_maxIterations'] = 5
        self.protocol['show_mixing_statistics'] = False

        return

    def run(self):
        """
        Run a free energy calculation.
        
        TODO: Have CPUs run ligand in vacuum and implicit solvent, while GPUs run ligand in explicit solvent and complex.

        TODO: Add support for explicit solvent.  Would ligand be solvated here?  Or would ligand already be in solvent?
        
        """

        # Initialize if we haven't yet done so.
        if not self._initialized:
            self._initialize()
                    
        # Create reference thermodynamic state corresponding to experimental conditions.
        reference_state = ThermodynamicState(temperature=self.temperature, pressure=self.pressure)

        #
        # Set up ligand in vacuum simulation.
        #

        # TODO: Remove any implicit solvation forces, if present.
        #vacuum_ligand = pyopenmm.System(self.ligand)
        #for force in vacuum_ligand.forces:
        #    if type(force) in pyopenmm.IMPLICIT_SOLVATION_FORCES:
        #        vacuum_ligand.removeForce(force)
        #vacuum_ligand = vacuum_ligand.asSwig()

        #if self.verbose: print "Running vacuum simulation..."
        #factory = AbsoluteAlchemicalFactory(vacuum_ligand, ligand_atoms=range(self.ligand.getNumParticles()))
        #systems = factory.createPerturbedSystems(self.vacuum_protocol)
        #store_filename = os.path.join(self.output_directory, 'vacuum.nc')
        #vacuum_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_coordinates, store_filename, protocol=self.protocol)
        #if self.platform:
        #    if self.verbose: print "Using platform '%s'" % self.platform.getName()
        #    vacuum_simulation.platform = self.platform
        #else:
        #    vacuum_simulation.platform = openmm.Platform.getPlatformByName('Reference')
        #vacuum_simulation.nsteps_per_iteration = 500
        #vacuum_simulation.run() # DEBUG
        
        # 
        # Set up ligand in solvent simulation.
        #

        if self.verbose: print "Running solvent simulation..."
        factory = AbsoluteAlchemicalFactory(self.ligand, ligand_atoms=range(self.ligand.getNumParticles()))
        systems = factory.createPerturbedSystems(self.solvent_protocol)
        store_filename = os.path.join(self.output_directory, 'solvent.nc')
        solvent_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_coordinates, store_filename, protocol=self.protocol)
        if self.platform:
            if self.verbose: print "Using platform '%s'" % self.platform.getName()
            solvent_simulation.platform = self.platform
        solvent_simulation.nsteps_per_iteration = 500
        solvent_simulation.run() # DEBUG
        
        #
        # Set up ligand in complex simulation.
        # 

        if self.verbose: print "Setting up complex simulation..."

        if not self.is_periodic:
            # Impose restraints to keep the ligand from drifting too far from the protein.
            import restraints
            reference_coordinates = self.complex_coordinates[0]
            if self.restraint_type == 'harmonic':
                restraints = restraints.ReceptorLigandRestraint(reference_state, self.complex, reference_coordinates, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'flat-bottom':
                restraints = restraints.FlatBottomReceptorLigandRestraint(reference_state, self.complex, reference_coordinates, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'none':
                restraints = None
            else:
                raise Exception("restraint_type of '%s' is not supported." % self.restraint_type)
            if restraints:
                force = restraints.getRestraintForce() # Get Force object incorporating restraints
                self.complex.addForce(force)
                self.standard_state_correction = restraints.getStandardStateCorrection() # standard state correction in kT
            else:
                # TODO: We need to include a standard state correction for going from simulation box volume to standard state volume.
                # TODO: Alternatively, we need a scheme for specifying restraints with only protein molecules, not solvent.
                self.standard_state_correction = 0.0 
        
        factory = AbsoluteAlchemicalFactory(self.complex, ligand_atoms=self.ligand_atoms)
        systems = factory.createPerturbedSystems(self.complex_protocol, verbose=self.verbose)
        store_filename = os.path.join(self.output_directory, 'complex.nc')

        metadata = dict()
        metadata['standard_state_correction'] = self.standard_state_correction

        if self.randomize_ligand:
            print "Randomizing ligand positions and excluding overlapping configurations..."
            randomized_coordinates = list()
            sigma = 20.0 * units.angstrom
            close_cutoff = 3.0 * units.angstrom
            nstates = len(systems)
            for state_index in range(nstates):
                coordinates = self.complex_coordinates[numpy.random.randint(0, len(self.complex_coordinates))]
                new_coordinates = ModifiedHamiltonianExchange.randomize_ligand_position(coordinates, self.receptor_atoms, self.ligand_atoms, sigma, close_cutoff)
                randomized_coordinates.append(new_coordinates)
            self.complex_coordinates = randomized_coordinates

        complex_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.complex_coordinates, store_filename, displacement_sigma=self.displacement_sigma, mc_atoms=self.ligand_atoms, protocol=self.protocol, metadata=metadata)
        complex_simulation.nsteps_per_iteration = 500
        if self.platform:
            if self.verbose: print "Using platform '%s'" % self.platform.getName()
            complex_simulation.platform = self.platform

        # Run the simulation.
        if self.verbose: print "Running complex simulation..."
        complex_simulation.run()        
        
        return

    def run_mpi(self, mpi_comm_world, cpuid_gpuid_mapping=None, ncpus_per_node=None):
        """
        Run a free energy calculation using MPI.
        
        ARGUMENTS
        
        mpi_comm_world - MPI 'world' communicator
        
        """

        # TODO: Make a configuration file for CPU:GPU id mapping, or determine it automatically?

        # TODO: Reduce code duplication by combining run_mpi() and run() or making more use of class methods.

        # Turn off output from non-root nodes:
        if not (mpi_comm_world.rank==0):
            verbose = False

        # Make sure random number generators have unique seeds.
        seed = numpy.random.randint(sys.maxint - MPI.COMM_WORLD.size) + MPI.COMM_WORLD.rank
        numpy.random.seed(seed)

        # Specify which CPUs should be attached to specific GPUs for maximum performance.
        cpu_platform_name = 'Reference'
        #gpu_platform_name = 'CPU' 
        #gpu_platform_name = 'CUDA' 
        gpu_platform_name = 'OpenCL'
        
        if not cpuid_gpuid_mapping:
            # TODO: Determine number of GPUs and set up simple mapping.
            cpuid_gpuid_mapping = { 0:0, 1:1, 2:2, 3:3 }

        # If number of CPUs per node not specified, set equal to total number of MPI processes.
        # TODO: Automatically determine number of CPUs per node.
        if ncpus_per_node is None:
            ncpus_per_node = MPI.COMM_WORLD.size

        # Choose appropriate platform for each device.
        # TODO: Print more compact report of MPI node, rank, and device.
        cpuid = MPI.COMM_WORLD.rank % ncpus_per_node # use default rank as CPUID (TODO: Improve this)
        #print "node '%s' MPI_WORLD rank %d/%d" % (hostname, MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)
        if cpuid in cpuid_gpuid_mapping.keys():
            platform = openmm.Platform.getPlatformByName(gpu_platform_name)
            deviceid = cpuid_gpuid_mapping[cpuid]
            platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
            platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index
            print "node '%s' MPI_WORLD rank %d/%d cpuid %d platform %s deviceid %d" % (hostname, MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, cpuid, gpu_platform_name, deviceid)
        else:
            platform = openmm.Platform.getPlatformByName(cpu_platform_name)
            print "node '%s' MPI_WORLD rank %d/%d running on CPU" % (hostname, MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)

        # Set up CPU and GPU communicators.
        gpu_process_list = filter(lambda x : x < MPI.COMM_WORLD.size, cpuid_gpuid_mapping.keys())
        if cpuid in gpu_process_list:
            this_is_gpu_process = 1 # running on a GPU
        else:
            this_is_gpu_process = 0 # running on a CPU
        comm = MPI.COMM_WORLD.Split(color=this_is_gpu_process)

        # Initialize if we haven't yet done so.
        if not self._initialized:
            self._initialize()
                    
        # Create reference thermodynamic state corresponding to experimental conditions.
        reference_state = ThermodynamicState(temperature=self.temperature, pressure=self.pressure)

        # All processes assist in creating alchemically-modified complex.
        
        # Run ligand in complex simulation on GPUs.
        #self.protocol['verbose'] = False # DEBUG: Suppress terminal output from ligand in solvent and vacuum simulations.

        self.standard_state_correction = 0.0 
        if not self.is_periodic: 
            # Impose restraints to keep the ligand from drifting too far from the protein.
            import restraints
            reference_coordinates = self.complex_coordinates[0]
            if self.restraint_type == 'harmonic':
                restraints = restraints.ReceptorLigandRestraint(reference_state, self.complex, reference_coordinates, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'flat-bottom':
                restraints = restraints.FlatBottomReceptorLigandRestraint(reference_state, self.complex, reference_coordinates, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'none':
                restraints = None
            else:
                raise Exception("restraint_type of '%s' is not supported." % self.restraint_type)
            if restraints:
                force = restraints.getRestraintForce() # Get Force object incorporating restraints
                self.complex.addForce(force)
                self.standard_state_correction = restraints.getStandardStateCorrection() # standard state correction in kT
            else:
                # TODO: We need to include a standard state correction for going from simulation box volume to standard state volume.
                # TODO: Alternatively, we need a scheme for specifying restraints with only protein molecules, not solvent.
                self.standard_state_correction = 0.0 

        # Create alchemically perturbed systems if not resuming run.
        try:
            # We can attempt to restore if serialized states exist.
            # TODO: We probably don't need to check this.  
            import time
            initial_time = time.time()
            store_filename = os.path.join(self.output_directory, 'complex.nc')
            import netCDF4 as netcdf
            ncfile = netcdf.Dataset(store_filename, 'r') 
            ncvar_systems = ncfile.groups['thermodynamic_states'].variables['systems']
            nsystems = ncvar_systems.shape[0]
            ncfile.close()
            systems = None
            resume = True
        except Exception as e:        
            # Create states using alchemical factory.
            factory = AbsoluteAlchemicalFactory(self.complex, ligand_atoms=self.ligand_atoms)
            systems = factory.createPerturbedSystems(self.complex_protocol, verbose=self.verbose, mpicomm=MPI.COMM_WORLD)
            resume = False

        if this_is_gpu_process:
            # Only GPU processes continue with complex simulation.

            store_filename = os.path.join(self.output_directory, 'complex.nc')

            metadata = dict()
            metadata['standard_state_correction'] = self.standard_state_correction

            if self.randomize_ligand and not resume:
                randomized_coordinates = list()
                sigma = 20.0 * units.angstrom
                close_cutoff = 1.5 * units.angstrom
                nstates = len(systems)
                for state_index in range(nstates):
                    coordinates = self.complex_coordinates[numpy.random.randint(0, len(self.complex_coordinates))]
                    new_coordinates = ModifiedHamiltonianExchange.randomize_ligand_position(coordinates, self.receptor_atoms, self.ligand_atoms, sigma, close_cutoff)
                    randomized_coordinates.append(new_coordinates)
                self.complex_coordinates = randomized_coordinates
                
            # Set up Hamiltonian exchange simulation.
            complex_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.complex_coordinates, store_filename, displacement_sigma=self.displacement_sigma, mc_atoms=self.ligand_atoms, protocol=self.protocol, mpicomm=comm, metadata=metadata)
            complex_simulation.platform = platform
            complex_simulation.nsteps_per_iteration = 500
            complex_simulation.run()        

        else:
            print "Running on cpu (node %s, rank %d / %d)" % (hostname, MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)

            self.protocol['verbose'] = False # DEBUG: Suppress terminal output from ligand in solvent and vacuum simulations.

            # Run ligand in solvent simulation on CPUs.
            # TODO: Have this run on GPUs if explicit solvent.

            factory = AbsoluteAlchemicalFactory(self.ligand, ligand_atoms=range(self.ligand.getNumParticles()))
            systems = factory.createPerturbedSystems(self.solvent_protocol)
            store_filename = os.path.join(self.output_directory, 'solvent.nc')
            solvent_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_coordinates, store_filename, protocol=self.protocol, mpicomm=comm)
            solvent_simulation.platform = openmm.Platform.getPlatformByName(cpu_platform_name)
            solvent_simulation.nsteps_per_iteration = 500
            solvent_simulation.run() 

            # Run ligand in vacuum simulation on CPUs.                        
            # Remove any implicit solvation forces, if present.
            #vacuum_ligand = pyopenmm.System(self.ligand)
            #for force in vacuum_ligand.forces:
            #    if type(force) in pyopenmm.IMPLICIT_SOLVATION_FORCES:
            #        vacuum_ligand.removeForce(force)
            #vacuum_ligand = vacuum_ligand.asSwig()
            #    
            #factory = AbsoluteAlchemicalFactory(vacuum_ligand, ligand_atoms=range(self.ligand.getNumParticles()))
            #systems = factory.createPerturbedSystems(self.vacuum_protocol)
            #store_filename = os.path.join(self.output_directory, 'vacuum.nc')
            #vacuum_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_coordinates, store_filename, protocol=self.protocol, mpicomm=comm)
            #vacuum_simulation.platform = openmm.Platform.getPlatformByName(cpu_platform_name)
            #vacuum_simulation.nsteps_per_iteration = 500
            #vacuum_simulation.run() # DEBUG
        

        # Wait for all nodes to finish.
        MPI.COMM_WORLD.barrier()
       
        return

    @classmethod
    def _extract_u_n(cls, ncfile):
        """
        Extract timeseries of u_n = - log q(x_n)               

        """

        # Get current dimensions.
        niterations = ncfile.variables['energies'].shape[0]
        nstates = ncfile.variables['energies'].shape[1]
        natoms = ncfile.variables['energies'].shape[2]

        # Extract energies.
        energies = ncfile.variables['energies']
        u_kln_replica = numpy.zeros([nstates, nstates, niterations], numpy.float64)
        for n in range(niterations):
            u_kln_replica[:,:,n] = energies[n,:,:]

        # Deconvolve replicas
        u_kln = numpy.zeros([nstates, nstates, niterations], numpy.float64)
        for iteration in range(niterations):
            state_indices = ncfile.variables['states'][iteration,:]
            u_kln[state_indices,:,iteration] = energies[iteration,:,:]

        # Compute total negative log probability over all iterations.
        u_n = numpy.zeros([niterations], numpy.float64)
        for iteration in range(niterations):
            u_n[iteration] = numpy.sum(numpy.diagonal(u_kln[:,:,iteration]))

        return u_n

    def analyze(self, verbose=False):
        """
        Analyze the results of a YANK free energy calculation.

        OPTIONAL ARGUMENTS

        verbose (bool) - if True, will print verbose progress information (default: False)

        """

        import analyze
        import pymbar
        import timeseries
        import netCDF4 as netcdf

        # Storage for results.
        results = dict()

        if verbose: print "Analyzing simulation data..."

        # Process each netcdf file in output directory.
        source_directory = self.output_directory
        #phases = ['vacuum', 'solvent', 'complex'] # DEBUG
        phases = ['solvent', 'complex']
        for phase in phases:
            # Construct full path to NetCDF file.
            fullpath = os.path.join(source_directory, phase + '.nc')

            # Skip if the file doesn't exist.
            if (not os.path.exists(fullpath)): continue

            # Open NetCDF file for reading.
            print "Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars()
            ncfile = netcdf.Dataset(fullpath, 'r')

            # Read dimensions.
            niterations = ncfile.variables['positions'].shape[0]
            nstates = ncfile.variables['positions'].shape[1]
            natoms = ncfile.variables['positions'].shape[2]
            if verbose: print "Read %(niterations)d iterations, %(nstates)d states" % vars()

            # Choose number of samples to discard to equilibration.
            u_n = self._extract_u_n(ncfile)
            [nequil, g_t, Neff_max] = timeseries.detectEquilibration(u_n)
            if verbose: print [nequil, Neff_max]

            # Examine mixing statistics.
            analyze.show_mixing_statistics(ncfile, cutoff=0.05, nequil=nequil)

            # Estimate free energies.
            (Deltaf_ij, dDeltaf_ij) = analyze.estimate_free_energies(ncfile, ndiscard=nequil)
    
            # Estimate average enthalpies
            (DeltaH_i, dDeltaH_i) = analyze.estimate_enthalpies(ncfile, ndiscard=nequil)
    
            # Accumulate free energy differences
            entry = dict()
            entry['DeltaF'] = Deltaf_ij[0,nstates-1] 
            entry['dDeltaF'] = dDeltaf_ij[0,nstates-1]
            entry['DeltaH'] = DeltaH_i[nstates-1] - DeltaH_i[0]
            entry['dDeltaH'] = numpy.sqrt(dDeltaH_i[0]**2 + dDeltaH_i[nstates-1]**2)
            results[phase] = entry

            # Get temperatures.
            ncvar = ncfile.groups['thermodynamic_states'].variables['temperatures']
            temperature = ncvar[0] * units.kelvin
            kT = analyze.kB * temperature

            # Close input NetCDF file.
            ncfile.close()
        
        return results

#=============================================================================================
# Command-line driver
#=============================================================================================

def read_amber_crd(filename, natoms_expected, verbose=False):
    """
    Read AMBER coordinate file.

    ARGUMENTS

    filename (string) - AMBER crd file to read
    natoms_expected (int) - number of atoms expected

    RETURNS

    coordinates (numpy-wrapped simtk.unit.Quantity with units of distance) - a single read coordinate set

    TODO

    * Automatically handle box vectors for systems in explicit solvent
    * Merge this function back into main?

    """

    if verbose: print "Reading cooordinate sets from '%s'..." % filename
    
    # Read coordinates.
    import simtk.openmm.app as app
    inpcrd = app.AmberInpcrdFile(filename)
    coordinates = inpcrd.getPositions(asNumpy=True)   

    # Check to make sure number of atoms match expectation.
    natoms = coordinates.shape[0]
    if natoms != natoms_expected:
        raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))

    return coordinates

def read_openeye_crd(filename, natoms_expected, verbose=False):
    """
    Read one or more coordinate sets from a file that OpenEye supports.

    ARGUMENTS
    
    filename (string) - the coordinate filename to be read
    natoms_expected (int) - number of atoms expected

    RETURNS
    
    coordinates_list (list of numpy array of simtk.unit.Quantity) - list of coordinate sets read

    """

    if verbose: print "Reading cooordinate sets from '%s'..." % filename

    import openeye.oechem as oe
    imolstream = oe.oemolistream()
    imolstream.open(filename)
    coordinates_list = list()
    for molecule in imolstream.GetOEGraphMols():
        oecoords = molecule.GetCoords() # oecoords[atom_index] is tuple of atom coordinates, in angstroms
        natoms = len(oecoords) # number of atoms
        if natoms != natoms_expected:
            raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))
        coordinates = units.Quantity(numpy.zeros([natoms,3], numpy.float32), units.angstroms) # coordinates[atom_index,dim_index] is coordinates of dim_index dimension of atom atom_index
        for atom_index in range(natoms):
            coordinates[atom_index,:] = units.Quantity(numpy.array(oecoords[atom_index]), units.angstroms)
        coordinates_list.append(coordinates)

    if verbose: print "%d coordinate sets read." % len(coordinates_list)
    
    return coordinates_list

def read_pdb_crd(filename, natoms_expected, verbose=False):
    """
    Read one or more coordinate sets from a PDB file.
    Multiple coordinate sets (in the form of multiple MODELs) can be read.

    ARGUMENTS

    filename (string) - name of the file to be read
    natoms_expected (int) - number of atoms expected

    RETURNS

    coordinates_list (list of numpy array of simtk.unit.Quantity) - list of coordinate sets read

    """
    
    # Open PDB file.
    pdbfile = open(filename, 'r')

    # Storage for sets of coordinates.
    coordinates_list = list()
    coordinates = units.Quantity(numpy.zeros([natoms_expected,3], numpy.float32), units.angstroms) # coordinates[atom_index,dim_index] is coordinates of dim_index dimension of atom atom_index

    # Extract the ATOM entries.
    # Format described here: http://bmerc-www.bu.edu/needle-doc/latest/atom-format.html
    atom_index = 0
    atoms = list()
    for line in pdbfile:
        if line[0:6] == "MODEL ":
            # Reset atom counter.
            atom_index = 0
            atoms = list()
        elif (line[0:6] == "ENDMDL") or (line[0:6] == "END   "):
            # Store model.
            coordinates_list.append(copy.deepcopy(coordinates))
            coordinates *= 0
            atom_index = 0
        elif line[0:6] == "ATOM  ":
            # Parse line into fields.
            atom = dict()
            atom["serial"] = line[6:11]
            atom["atom"] = line[12:16]
            atom["altLoc"] = line[16:17]
            atom["resName"] = line[17:20]
            atom["chainID"] = line[21:22]
            atom["Seqno"] = line[22:26]
            atom["iCode"] = line[26:27]
            atom["x"] = line[30:38]
            atom["y"] = line[38:46]
            atom["z"] = line[46:54]
            atom["occupancy"] = line[54:60]
            atom["tempFactor"] = line[60:66]
            atoms.append(atom)
            coordinates[atom_index,:] = units.Quantity(numpy.array([float(atom["x"]), float(atom["y"]), float(atom["z"])]), units.angstroms)
            atom_index += 1

    # Close PDB file.
    pdbfile.close()

    # Append if we haven't dumped coordinates yet.
    if (atom_index == natoms_expected):
        coordinates_list.append(copy.deepcopy(coordinates))

    # Return coordinates.
    return coordinates_list

if __name__ == '__main__':    
    # Initialize command-line argument parser.

    """
    USAGE

    %prog --ligand_prmtop PRMTOP --receptor_prmtop PRMTOP { {--ligand_crd CRD | --ligand_mol2 MOL2} {--receptor_crd CRD | --receptor_pdb PDB} | {--complex_crd CRD | --complex_pdb PDB} } [-v | --verbose] [-i | --iterations ITERATIONS] [-o | --online] [-m | --mpi] [--restraints restraint-type] [--doctests] [--randomize_ligand]

    EXAMPLES

    # Specify AMBER prmtop/crd files for ligand and receptor.
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --ligand_crd ligand.crd --receptor_crd receptor.crd --iterations 1000

    # Specify (potentially multi-conformer) mol2 file for ligand and (potentially multi-model) PDB file for receptor.
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --ligand_mol2 ligand.mol2 --receptor_pdb receptor.pdb --iterations 1000

    # Specify (potentially multi-model) PDB file for complex, along with flat-bottom restraints (instead of harmonic).
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_pdb complex.pdb --iterations 1000 --restraints flat-bottom

    # Specify (potentially multi-model) PDB file for complex, along with flat-bottom restraints (instead of harmonic); randomize ligand positions/orientations at start.
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_pdb complex.pdb --iterations 1000 --restraints flat-bottom --randomize_ligand

    NOTES

    In atom ordering, receptor comes before ligand atoms.

    """

    # Parse command-line arguments.
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--ligand_prmtop", dest="ligand_prmtop_filename", default=None, help="ligand Amber parameter file", metavar="LIGAND_PRMTOP")
    parser.add_option("--receptor_prmtop", dest="receptor_prmtop_filename", default=None, help="receptor Amber parameter file", metavar="RECEPTOR_PRMTOP")    
    parser.add_option("--ligand_crd", dest="ligand_crd_filename", default=None, help="ligand Amber crd file", metavar="LIGAND_CRD")
    parser.add_option("--receptor_crd", dest="receptor_crd_filename", default=None, help="receptor Amber crd file", metavar="RECEPTOR_CRD")
    parser.add_option("--ligand_mol2", dest="ligand_mol2_filename", default=None, help="ligand mol2 file (can contain multiple conformations)", metavar="LIGAND_MOL2")
    parser.add_option("--receptor_pdb", dest="receptor_pdb_filename", default=None, help="receptor PDB file (can contain multiple MODELs)", metavar="RECEPTOR_PDB")
    parser.add_option("--complex_prmtop", dest="complex_prmtop_filename", default=None, help="complex Amber parameter file", metavar="COMPLEX_PRMTOP")
    parser.add_option("--complex_crd", dest="complex_crd_filename", default=None, help="complex Amber crd file", metavar="COMPLEX_CRD")
    parser.add_option("--complex_pdb", dest="complex_pdb_filename", default=None, help="complex PDB file (can contain multiple MODELs)", metavar="COMPLEX_PDB")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="verbosity flag")
    parser.add_option("-i", "--iterations", dest="niterations", default=None, help="number of iterations", metavar="ITERATIONS")
    parser.add_option("-o", "--online", dest="online_analysis", default=False, help="perform online analysis")
    parser.add_option("-m", "--mpi", action="store_true", dest="mpi", default=False, help="use mpi if possible")
    parser.add_option("--restraints", dest="restraint_type", default=None, help="specify ligand restraint type: 'harmonic' or 'flat-bottom' (default: 'harmonic')")
    parser.add_option("--output", dest="output_directory", default=None, help="specify output directory---must be unique for each calculation (default: current directory)")
    parser.add_option("--doctests", action="store_true", dest="doctests", default=False, help="run doctests first (default: False)")
    parser.add_option("--randomize_ligand", action="store_true", dest="randomize_ligand", default=False, help="randomize ligand positions and orientations (default: False)")
    parser.add_option("--ignore_signal", action="append", dest="ignore_signals", default=[], help="signals to trap and ignore (default: None)")

    # Parse command-line arguments.
    (options, args) = parser.parse_args()
    
    if options.doctests:
        print "Running doctests..."
        import doctest
        (failure_count, test_count) = doctest.testmod(verbose=options.verbose)
        if failure_count == 0:
            print "All doctests pass."
            sys.exit(0)
        else:
            print "WARNING: There were %d doctest failures." % failure_count
            sys.exit(1)

    # Check arguments for validity.
    if not (options.ligand_prmtop_filename and options.receptor_prmtop_filename):
        parser.error("ligand and receptor prmtop files must be specified")        
    if not (bool(options.ligand_mol2_filename) ^ bool(options.ligand_crd_filename) ^ bool(options.complex_pdb_filename) ^ bool(options.complex_crd_filename)):
        parser.error("Ligand coordinates must be specified through only one of --ligand_crd, --ligand_mol2, --complex_crd, or --complex_pdb.")
    if not (bool(options.receptor_pdb_filename) ^ bool(options.receptor_crd_filename) ^ bool(options.complex_pdb_filename) ^ bool(options.complex_crd_filename)):
        parser.error("Receptor coordinates must be specified through only one of --receptor_crd, --receptor_pdb, --complex_crd, or --complex_pdb.")    

    # DEBUG: Require complex prmtop files to be specified while JDC debugs automatic combination of systems.
    if not (options.complex_prmtop_filename):
        parser.error("Please specify --complex_prmtop [complex_prmtop_filename] argument.  JDC is still debugging automatic generation of complex topologies from receptor+ligand.")

    # Ignore requested signals.
    if len(options.ignore_signals) > 0:
        import signal
        # Create a dummy signal handler.
        def signal_handler(signal, frame):
            print 'Signal %s received and ignored.' % str(signal)
        # Register the dummy signal handler.
        for signal_name in options.ignore_signals:
            print "Will ignore signal %s" % signal_name
            signal.signal(getattr(signal, signal_name), signal_handler)

    # Initialize MPI if requested.
    if options.mpi:
        # Initialize MPI. 
        try:
            from mpi4py import MPI # MPI wrapper
            hostname = os.uname()[1]
            options.mpi = MPI.COMM_WORLD
            if not MPI.COMM_WORLD.rank == 0: 
                options.verbose = False
            MPI.COMM_WORLD.barrier()
            if MPI.COMM_WORLD.rank == 0: print "Initialized MPI on %d processes." % (MPI.COMM_WORLD.size)
        except Exception as e:
            print e
            parser.error("Could not initialize MPI.")

    # Select simulation parameters.
    # TODO: Simulation parameters will be different for explicit solvent.
    import simtk.openmm.app as app
    nonbondedMethod = app.NoCutoff
    implicitSolvent = app.OBC2
    constraints = app.HBonds
    removeCMMotion = False

    # Create System objects for ligand and receptor.
    ligand_system = app.AmberPrmtopFile(options.ligand_prmtop_filename).createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)
    receptor_system = app.AmberPrmtopFile(options.receptor_prmtop_filename).createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)
    
    # Load complex prmtop if specified.
    complex_system = None
    if options.complex_prmtop_filename:
        complex_system = app.AmberPrmtopFile(options.complex_prmtop_filename).createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)

    # Determine number of atoms for each system.
    natoms_receptor = receptor_system.getNumParticles()
    natoms_ligand = ligand_system.getNumParticles()
    natoms_complex = natoms_receptor + natoms_ligand

    # Read ligand and receptor coordinates.
    ligand_coordinates = list()
    receptor_coordinates = list()
    complex_coordinates = list()
    if (options.complex_crd_filename or options.complex_pdb_filename):
        # Read coordinates for whole complex.
        if options.complex_crd_filename:
            coordinates = read_amber_crd(options.complex_crd_filename, natoms_complex, options.verbose)
            complex_coordinates.append(coordinates)
        else:
            try:
                coordinates_list = read_openeye_crd(options.complex_pdb_filename, natoms_complex, options.verbose)
            except:
                coordinates_list = read_pdb_crd(options.complex_pdb_filename, natoms_complex, options.verbose)
            complex_coordinates += coordinates_list
    elif options.ligand_crd_filename:
        coordinates = read_amber_crd(options.ligand_crd_filename, natoms_ligand, options.verbose)
        coordinates = units.Quantity(numpy.array(coordinates / coordinates.unit), coordinates.unit)
        ligand_coordinates.append(coordinates)
    elif options.ligand_mol2_filename:
        coordinates_list = read_openeye_crd(options.ligand_mol2_filename, natoms_ligand, options.verbose)
        ligand_coordinates += coordinates_list
    elif options.receptor_crd_filename:
        coordinates = read_amber_crd(options.receptor_crd_filename, natoms_receptor, options.verbose)
        coordinates = units.Quantity(numpy.array(coordinates / coordinates.unit), coordinates.unit)
        receptor_coordinates.append(coordinates)
    elif options.receptor_pdb_filename:
        try:
            coordinates_list = read_openeye_crd(options.receptor_pdb_filename, natoms_receptor, options.verbose)
        except:
            coordinates_list = read_pdb_crd(options.receptor_pdb_filename, natoms_receptor, options.verbose)
        receptor_coordinates += coordinates_list

    # Assemble complex coordinates if we haven't read any.
    if len(complex_coordinates)==0:
        for x in receptor_coordinates:
            for y in ligand_coordinates:
                z = units.Quantity(numpy.zeros([natoms_complex,3]), units.angstroms)
                z[0:natoms_receptor,:] = x[:,:]
                z[natoms_receptor:natoms_complex,:] = y[:,:]
                complex_coordinates.append(z)

    # Initialize YANK object.
    yank = Yank(receptor=receptor_system, ligand=ligand_system, complex=complex_system, complex_coordinates=complex_coordinates, output_directory=options.output_directory, verbose=options.verbose)

    # Configure YANK object with command-line parameter overrides.
    if options.niterations is not None:
        yank.niterations = int(options.niterations)
    if options.verbose:
        yank.verbose = options.verbose
    if options.online_analysis:
        yank.online_analysis = options.online_analysis
    if options.restraint_type is not None:
        yank.restraint_type = options.restraint_type
    if options.randomize_ligand:
        yank.randomize_ligand = True 

    # cpuid:gpuid for NCSA Forge # TODO: Make a configuration file or command-line argument for this, or have it autodetect.
    #cpuid_gpuid_mapping = { 0:0, 1:1, 8:2, 9:3, 10:4, 11:5 } 

    # cpuid:gpuid for Blue Waters
    #cpuid_gpuid_mapping = { 0:0, 8:0, } 
    #ncpus_per_node = 16

    # cpuid:gpuid for Exxact 4xGPU nodes
    cpuid_gpuid_mapping = { 0:0, 1:1, 2:2, 3:3 } 
    ncpus_per_node = None

    # Run calculation.
    if options.mpi:
        # Run MPI version.
        yank.run_mpi(options.mpi, cpuid_gpuid_mapping=cpuid_gpuid_mapping, ncpus_per_node=ncpus_per_node)
    else:
        # Run serial version.
        yank.platform = openmm.Platform.getPlatformByName("CPU") # DEBUG: Use CPU platform for serial version
        yank.run()

    # Run analysis.
    #results = yank.analyze()

    # TODO: Print/write results.
    #print results

    

