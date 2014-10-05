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
* Handle complex_positions argument in Yank more intelligently if different kinds of input are provided.
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

from . import sampling

from alchemy import AbsoluteAlchemicalFactory
from oldrepex import ThermodynamicState
from oldrepex import HamiltonianExchange, ReplicaExchange

#=============================================================================================
# YANK class
#=============================================================================================

class Yank(object):
    """
    A class for computing receptor-ligand binding free energies through alchemical transformations.

    Attributes that can be set after initializaton but before initialization by run():

    data_directory (string) - destination for datafiles (default: .)
    online_analysis (boolean) - if True, will analyze data as simulations proceed (default: False)
    temperature (simtk.unit.Quantity compatible with kelvin) - temperature of simulation (default: 298 K)
    pressure (simtk.unit.Quantity compatible with atmosphere) - pressure of simulation if explicit solvent (default: 1 atm)
    niterations (integer) - number of iterations to run (default: 10)
    solvent_protocol (list of AlchemicalState) - protocol to use for turning off ligand in solvent
    complex_protocol (list of AlchemicalState) - protocol to use for turning off ligand in complex

    """

    def __init__(self, store_directory):
        """
        Initialize YANK object with default parameters.

        Parameters
        ----------
        store_directory : str
           The storage directory in which output NetCDF files are read or written.

        """

        # Mark as not yet initialized.
        self._initialized = False

        # Set defaults for free energy calculation and protocol.
        self.verbose = False # Don't show verbose output
        self.online_analysis = False # if True, simulation will be analyzed each iteration
        self.temperature = 298.0 * units.kelvin # simulation temperature
        self.pressure = 1.0 * units.atmosphere # simulation pressure (for explicit solvent)
        self.niterations = 2000 # number of production iterations
        self.platform = None # don't specify a platform

        # Set internal variables.
        self._phases = list()
        self._store_filenames = dict()

        return

    def resume(self, phases=None, options=None, verbose=False):
        """
        Set up YANK to resume from an existing simulation.

        Parameters
        ----------
        phases : list of str, optional, default=None
           The list of calculation phases (e.g. ['solvent', 'complex']) to resume.
           If not specified, all NetCDF files ('*.nc') in the store_directory will be resumed.
        options : dict of str, optional, defauilt=None
           If specified, some parameters can be overridden after restoring them from NetCDF file.
        verbose : bool, optional, default=True
           If True, will give verbose output

        """
        # If no phases specified, construct a list of phases from the filename prefixes in the store directory.
        if phases is None:
            phases = list()
            fullpaths = glob.glob(os.path.join(store_directory, '*.nc'))
            for fullpath in fullpaths:
                [filepath, filename] = os.path.split(fullpath)
                [shortname, extension] = os.path.splitext(filename)
                phases.append(shortname)
        self._phases = phases

        # Construct store filenames.
        for phase in self._phases:
            self._store_filenames[phase] = os.path.join(store_directory, phase + '.nc')

        # Ensure we can resume from each store file, processing override options.
        for phase in self._phases:
            self._store_filenames[phase] = os.path.join(store_directory, phase + '.nc')

        # TODO: Resume from files on disk.

        # TODO: Process options overrides.



        return

    def create(self, phases, systems, positions, alchemical_indices, thermodynamic_state, verbose=False):
        """
        Create a YANK binding free energy calculation object.

        Parameters
        ----------
        store_directory : str
           The storage directory in which output NetCDF files are read or written.
        phases : list of str, optional, default=None
           The list of calculation phases (e.g. ['solvent', 'complex']) to run.
           If resuming, will resume from all NetCDF files ('*.nc') in the store_directory unless specific phases are given.
        systems : dict of simtk.openmm.System, optional, default=None
           A dict of System objects for each phase, e.g. systems['solvent'] is for solvent phase.
        positions : dict of simtk.unit.Quantity arrays (numpy or Python) with units compatible with nanometers, or dict of lists, optional, default=None
           A dict of positions corresponding to each phase, e.g. positions['solvent'] is a single set of positions or list of positions to seed replicas.
           Shape must be natoms x 3, with natoms matching number of particles in corresponding system.
        alchemical_indices : dict of list of int, optional, default=None
           A dict of atom index lists corresponding to each phase, e.g. alchemical_indices['solvent'] is list of atoms to be alchemically eliminated in 'solvent' phase.
        thermodynamic_state : ThermodynamicState (System need not be defined), optional, default=None
           Thermodynamic state at which calculations are to be carried out
        verbose : bool, optional, default=True
           If True, will give verbose output

        """

        # Abort if there are files there already but initialization was requested.
        for phase in phases:
            store_filename = os.path.join(store_directory, phase + '.nc')
            if os.path.exists(store_filename):
                raise Exception("Store filename %s already exists." % store_filename)

        # Set defaults for free energy calculation and protocol.
        self.verbose = False # Don't show verbose output
        self.online_analysis = False # if True, simulation will be analyzed each iteration
        self.temperature = 298.0 * units.kelvin # simulation temperature
        self.pressure = 1.0 * units.atmosphere # simulation pressure (for explicit solvent)
        self.niterations = 2000 # number of production iterations
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
        if self.verbose: print "receptor has %d atoms; ligand has %d atoms" % (self.receptor.getNumParticles(), self.ligand.getNumParticles())

        # TODO: Use more general approach to determine whether system is periodic.
        self.is_periodic = False
        forces = { self.complex.getForce(index).__class__.__name__ : self.complex.getForce(index) for index in range(self.complex.getNumForces()) }
        if forces['NonbondedForce'].getNonbondedMethod in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
            self.is_periodic = True

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

        # Store complex positions.
        # TODO: Make sure each coordinate set is packaged as a numpy array.  For now, we require the user to pass in a list of Quantity objects that contain numpy arrays.
        # TODO: Switch to a special Coordinate object (with box size) to support explicit solvent simulations?
        self.complex_positions = copy.deepcopy(complex_positions)

        # Type of restraints requested.
        #self.restraint_type = 'flat-bottom' # default to a flat-bottom restraint between the ligand and receptor
        self.restraint_type = 'harmonic' # default to a single harmonic restraint between the ligand and receptor

        # DEBUG
        if self.verbose: print "Yank object created."

        return

    def _initialize(self):
        """
        """
        self._initialized = True

        # TODO: Run some sanity checks on arguments to see if we can initialize a valid simulation.

        # Turn off pressure if we aren't simulating a periodic system.
        if not self.is_periodic:
            self.pressure = None

        # Extract ligand positions from complex.
        self.ligand_positions = [ positions[self.ligand_atoms,:] for positions in self.complex_positions ]

        # TODO: Pack up a 'protocol' for modified Hamiltonian exchange simulations. Use this instead in setting up simulations.
        self.protocol = dict()
        self.protocol['number_of_equilibration_iterations'] = 1
        self.protocol['number_of_iterations'] = self.niterations
        self.protocol['verbose'] = self.verbose
        self.protocol['timestep'] = 2.0 * units.femtoseconds
        self.protocol['collision_rate'] = 5.0 / units.picoseconds
        self.protocol['minimize'] = True
        self.protocol['show_mixing_statistics'] = True # this causes slowdown with iteration and should not be used for production

        return

    def run(self, mpicomm=None):
        """
        Run a free energy calculation.

        Parameters
        ----------
        mpicomm : MPI communicator, optional, default=None
           If an MPI communicator is passed, an MPI simulation will be attempted.

        """
        # Make sure we've been properly initialized first.
        if not self._initialized:
            raise Exception("Yank must first be initialized by either resume() or create().")

        # Run all phases.
        from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
        for phase in self._phases:
            store_filename = self._store_filenames[phase]
            simulation = ModifiedHamiltonianExchange(store_filename=store_filename, mpicomm=mpicomm)
            simulation.run()


    def _run_serial(self):
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
        #vacuum_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_positions, store_filename, protocol=self.protocol)
        #if self.platform:
        #    if self.verbose: print "Using platform '%s'" % self.platform.getName()
        #    vacuum_simulation.platform = self.platform
        #else:
        #    vacuum_simulation.platform = openmm.Platform.getPlatformByName('Reference')
        #vacuum_simulation.nsteps_per_iteration = 5000
        #vacuum_simulation.run() # DEBUG

        #
        # Set up ligand in solvent simulation.
        #

        if self.verbose: print "Running solvent simulation..."
        factory = AbsoluteAlchemicalFactory(self.ligand, ligand_atoms=range(self.ligand.getNumParticles()))
        systems = factory.createPerturbedSystems(self.solvent_protocol)
        store_filename = os.path.join(self.output_directory, 'solvent.nc')
        from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
        solvent_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_positions, store_filename, protocol=self.protocol)
        if self.platform:
            if self.verbose: print "Using platform '%s'" % self.platform.getName()
            solvent_simulation.platform = self.platform
        solvent_simulation.nsteps_per_iteration = 5000
        solvent_simulation.run()

        #
        # Set up ligand in complex simulation.
        #

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
            systems = factory.createPerturbedSystems(self.complex_protocol)
            resume = False

        #if self.verbose: print "Setting up complex simulation..."
        if self.verbose: print "Creating receptor-ligand restraints..."
        if not self.is_periodic:
            # Impose restraints to keep the ligand from drifting too far from the protein.
            import restraints
            reference_positions = self.complex_positions[0]
            if self.restraint_type == 'harmonic':
                complex_restraints = restraints.HarmonicReceptorLigandRestraint(reference_state, self.complex, reference_positions, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'flat-bottom':
                complex_restraints = restraints.FlatBottomReceptorLigandRestraint(reference_state, self.complex, reference_positions, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'none':
                complex_restraints = None
            else:
                raise Exception("restraint_type of '%s' is not supported." % self.restraint_type)
            if restraints:
                force = complex_restraints.getRestraintForce() # Get Force object incorporating restraints
                self.complex.addForce(force)
                self.standard_state_correction = complex_restraints.getStandardStateCorrection() # standard state correction in kT
            else:
                # TODO: We need to include a standard state correction for going from simulation box volume to standard state volume.
                # TODO: Alternatively, we need a scheme for specifying restraints with only protein molecules, not solvent.
                self.standard_state_correction = 0.0

        #if self.verbose: print "Creating alchemical intermediates..."
        #factory = AbsoluteAlchemicalFactory(self.complex, ligand_atoms=self.ligand_atoms)
        #systems = factory.createPerturbedSystems(self.complex_protocol)
        #store_filename = os.path.join(self.output_directory, 'complex.nc')

        metadata = dict()
        metadata['standard_state_correction'] = self.standard_state_correction

        if self.randomize_ligand and not resume:
            if self.verbose: print "Randomizing ligand positions and excluding overlapping configurations..."
            randomized_positions = list()
            sigma = 2*complex_restraints.getReceptorRadiusOfGyration()
            close_cutoff = 1.5 * units.angstrom # TODO: Allow this to be specified by user.
            nstates = len(systems)
            for state_index in range(nstates):
                positions = self.complex_positions[numpy.random.randint(0, len(self.complex_positions))]
                from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
                new_positions = ModifiedHamiltonianExchange.randomize_ligand_position(positions, self.receptor_atoms, self.ligand_atoms, sigma, close_cutoff)
                randomized_positions.append(new_positions)
            self.complex_positions = randomized_positions

        if self.verbose: print "Setting up replica exchange simulation..."
        from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
        complex_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.complex_positions, store_filename, displacement_sigma=self.displacement_sigma, mc_atoms=self.ligand_atoms, protocol=self.protocol, metadata=metadata)
        complex_simulation.nsteps_per_iteration = 5000
        if self.platform:
            if self.verbose: print "Using platform '%s'" % self.platform.getName()
            complex_simulation.platform = self.platform

        # Run the simulation.
        if self.verbose: print "Running complex simulation..."
        complex_simulation.run()

        return

    def _run_mpi(self, mpicomm, gpus_per_node):
        """
        Run a free energy calculation using MPI.

        Parameters
        ----------
        mpicomm :  MPI 'world' communicator
            The MPI communicator to use.
        gpus_per_node : int
            The number of GPUs per node.

        Note
        ----
        * After run() or run_mpi() is called for the first time, the simulation parameters should no longer be changed.

        TODO
        ----
        * Add intelligent way to determine how many threads per node to use for CPUs and GPUs if left unspecified
        * Restore ability to run CPU solvent and GPU complex threads concurrently.

        """

        # TODO: Reduce code duplication by combining run_mpi() and run() or making more use of class methods.

        # Turn off output from non-root nodes:
        if not (mpicomm.rank==0):
            verbose = False

        # Make sure each thread's random number generators have unique seeds.
        # TODO: Also store seed in repex object.
        seed = numpy.random.randint(sys.maxint - mpicomm.size) + mpicomm.rank
        numpy.random.seed(seed)

        # Choose appropriate platform for each device.
        # Set GPU ID or number of threads.
        deviceid = mpicomm.rank % gpus_per_node
        platform_name = self.platform.getName()
        if platform_name == 'CUDA':
            self.platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index
        elif platform_name == 'OpenCL':
            self.platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
        elif platform_name == 'CPU':
            self.platform.setPropertyDefaultValue('CpuThreads', '1') # set number of CPU threads

        hostname = os.uname()[1]
        print "node '%s' MPI_WORLD rank %d/%d running on %s" % (hostname, mpicomm.rank, mpicomm.size, platform_name)

        # Initialize if we haven't yet done so.
        if not self._initialized:
            self._initialize()

        # Create reference thermodynamic state corresponding to experimental conditions.
        reference_state = ThermodynamicState(temperature=self.temperature, pressure=self.pressure)

        # SOLVENT simulation

        factory = AbsoluteAlchemicalFactory(self.ligand, ligand_atoms=range(self.ligand.getNumParticles()))
        systems = factory.createPerturbedSystems(self.solvent_protocol)
        store_filename = os.path.join(self.output_directory, 'solvent.nc')
        from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
        solvent_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_positions, store_filename, protocol=self.protocol, mpicomm=mpicomm)
        solvent_simulation.nsteps_per_iteration = 5000
        solvent_simulation.platform = self.platform
        solvent_simulation.run()
        mpicomm.barrier()

        # Run ligand in complex simulation on GPUs.
        #self.protocol['verbose'] = False # DEBUG: Suppress terminal output from ligand in solvent and vacuum simulations.
        if self.verbose: print "Creating receptor-ligand restraints..."
        self.standard_state_correction = 0.0
        if not self.is_periodic:
            # Impose restraints to keep the ligand from drifting too far from the protein.
            import restraints
            reference_positions = self.complex_positions[0]
            if self.restraint_type == 'harmonic':
                complex_restraints = restraints.HarmonicReceptorLigandRestraint(reference_state, self.complex, reference_positions, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'flat-bottom':
                complex_restraints = restraints.FlatBottomReceptorLigandRestraint(reference_state, self.complex, reference_positions, self.receptor_atoms, self.ligand_atoms)
            elif self.restraint_type == 'none':
                complex_restraints = None
            else:
                raise Exception("restraint_type of '%s' is not supported." % self.restraint_type)
            if complex_restraints:
                force = complex_restraints.getRestraintForce() # Get Force object incorporating restraints
                self.complex.addForce(force)
                self.standard_state_correction = complex_restraints.getStandardStateCorrection() # standard state correction in kT
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
            if self.verbose: print "Creating alchemical states..."
            factory = AbsoluteAlchemicalFactory(self.complex, ligand_atoms=self.ligand_atoms)
            systems = factory.createPerturbedSystems(self.complex_protocol)
            resume = False

        # COMPLEX simulation
        store_filename = os.path.join(self.output_directory, 'complex.nc')

        metadata = dict()
        metadata['standard_state_correction'] = self.standard_state_correction

        # Randomize ligand positions.
        if self.randomize_ligand and not resume:
            if self.verbose: print "Randomizing ligand positions..."
            randomized_positions = list()
            sigma = 2*complex_restraints.getReceptorRadiusOfGyration()
            close_cutoff = 1.5 * units.angstrom # TODO: Allow this to be specified by user.
            nstates = len(systems)
            for state_index in range(nstates):
                positions = self.complex_positions[numpy.random.randint(0, len(self.complex_positions))]
                from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
                new_positions = ModifiedHamiltonianExchange.randomize_ligand_position(positions, self.receptor_atoms, self.ligand_atoms, sigma, close_cutoff)
                randomized_positions.append(new_positions)
            self.complex_positions = randomized_positions

        # Set up Hamiltonian exchange simulation.
        if self.verbose: print "Setting up complex simulation..."
        from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
        complex_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.complex_positions, store_filename, displacement_sigma=self.displacement_sigma, mc_atoms=self.ligand_atoms, protocol=self.protocol, mpicomm=mpicomm, metadata=metadata)
        complex_simulation.nsteps_per_iteration = 5000
        complex_simulation.platform = self.platform
        complex_simulation.run()
        mpicomm.barrier()

        # VACUUM simulation

        # TODO: Create vacuum version of ligand.
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
        #vacuum_simulation = ModifiedHamiltonianExchange(reference_state, systems, self.ligand_positions, store_filename, protocol=self.protocol, mpicomm=mpicomm)
        #vacuum_simulation.nsteps_per_iteration = 5000
        #vacuum_simulation.run() # DEBUG
        #mpicomm.barrier()

        return

    # TODO: Move this to analyze?
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
        from pymbar import pymbar, timeseries
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

