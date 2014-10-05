#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Main interface for automated free energy calculations using OpenMM.

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
from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
from restraints import HarmonicReceptorLigandRestraint, FlatBottomReceptorLigandRestraint

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
        self.niterations = 1000 # number of production iterations
        self.platform_name = None # don't specify a platform

        # Store output directory.
        self._store_directory = store_directory

        # Set internal variables.
        self._phases = list()
        self._store_filenames = dict()

        # Monte Carlo displacement standard deviation for encouraging rapid decorrelation of ligand in the annihilated/decoupled state.
        self.displacement_sigma = 1.0 * units.nanometers # attempt to displace ligand by this stddev will be made each iteration

        # Type of restraints requested.
        #self.restraint_type = 'flat-bottom' # default to a flat-bottom restraint between the ligand and receptor
        self.restraint_type = 'harmonic' # default to a single harmonic restraint between the ligand and receptor

        return

    def _find_phases_in_store_directory(self):
        """
        Build a list of phases in the store directory.

        Parameters
        ----------
        store_directory : str
           The directory to examine for stored phase datafiles.

        Returns
        -------
        phases : list of str
           The names of phases found.

        """
        phases = list()
        fullpaths = glob.glob(os.path.join(self._store_directory, '*.nc'))
        for fullpath in fullpaths:
            [filepath, filename] = os.path.split(fullpath)
            [shortname, extension] = os.path.splitext(filename)
            phases.append(shortname)

        if len(phases) == 0:
            raise Exception("Could not find any valid YANK store (*.nc) files in store directory: %s" % self._store_directory)

        return phases

    # TODO: Get rid of this, or just use whether ReplicaExchange can resume?
    def _is_valid_store(self, store_filename):
        """
        Check if the specified store filename is valid.

        Parameters
        ----------
        store_filename : str
           The name of the store file to examine.

        Returns
        -------
        is_valid : bool
           True is returned if the store file is valid.
           An exception is raised if not.

        """
        ncfile = netcdf.Dataset(store_filename, 'r')
        ncvar_systems = ncfile.groups['thermodynamic_states'].variables['systems']
        nsystems = ncvar_systems.shape[0]
        ncfile.close()
        return True

    def resume(self, phases=None, verbose=False):
        """
        Set up YANK to resume from an existing simulation.

        Parameters
        ----------
        phases : list of str, optional, default=None
           The list of calculation phases (e.g. ['solvent', 'complex']) to resume.
           If not specified, all NetCDF files ('*.nc') in the store_directory will be resumed.
        verbose : bool, optional, default=True
           If True, will give verbose output

        """
        # If no phases specified, construct a list of phases from the filename prefixes in the store directory.
        if phases is None:
            phases = self._find_phases_in_store_directory()
        self._phases = phases

        # Construct store filenames.
        self._store_filenames = { phase : os.path.join(store_directory, phase + '.nc') for phase in self._phases }

        # Ensure we can resume from each store file, processing override options.
        for phase in self._phases:
            # TODO: Check the store file is OK.
            store_filename = self._store_filenames[phase]
            pass

        return

    def create(self, phases, systems, positions, atom_indices, thermodynamic_state, protocols=None, verbose=False):
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
        atom_indices : dict of dict list of int, optional, default=None
           atom_indices[phase][component] is a list of atom indices, for each component in ['ligand', 'receptor', 'complex', 'solvent']
        thermodynamic_state : ThermodynamicState (System need not be defined), optional, default=None
           Thermodynamic state at which calculations are to be carried out
        protocols : dict of list of AlchemicalState, optional, default=None
           If specified, the alchemical protocol protocols[phase] will be used for phase 'phase' instead of the default.
        verbose : bool, optional, default=True
           If True, will give verbose output

        """

        # Abort if there are files there already but initialization was requested.
        for phase in phases:
            store_filename = os.path.join(store_directory, phase + '.nc')
            if os.path.exists(store_filename):
                raise Exception("Store filename %s already exists." % store_filename)

        for phase in phases:
            self._initialize_phase(self, phase, systems[phase], positions[phase], atom_indices[phase], thermodynamic_states[phase], protocols=None):

        # DEBUG
        if self.verbose: print "Yank object created."

        return

    def _initialize(self):
        """
        Initialize the Yank calculation.


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

    def _initialize_phase(self, phase, reference_system, positions, atom_indices, thermodynamic_state, protocols=None):
        """
        Initialize a specific phase.


        """

        # Create metadata storage.
        metadata = dict()

        # Make a deep copy of the reference system so we don't accidentally modify it.
        reference_system = copy.deepcopy(reference_system)

        # TODO: Use more general approach to determine whether system is periodic.
        is_periodic = False
        forces = { reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index) for index in range(reference_system.getNumForces()) }
        if forces['NonbondedForce'].getNonbondedMethod in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
            is_periodic = True

        # Compute standard state corrections for complex phase.
        metadata['standard_state_correction'] = 0.0
        # TODO: Do we need to include a standard state correction for other phases in periodic boxes?
        if (phase == 'complex'):
            if (not is_periodic):
                # Impose restraints for complex system in implicit solvent to keep ligand from drifting too far away from receptor.
                if self.verbose: print "Creating receptor-ligand restraints..."
                reference_positions = positions[0]
                if self.restraint_type == 'harmonic':
                    restraints = HarmonicReceptorLigandRestraint(reference_state, self.complex, reference_positions, self.receptor_atoms, self.ligand_atoms)
                elif self.restraint_type == 'flat-bottom':
                    restraints = FlatBottomReceptorLigandRestraint(reference_state, self.complex, reference_positions, self.receptor_atoms, self.ligand_atoms)
                else:
                    raise Exception("restraint_type of '%s' is not supported." % self.restraint_type)

                force = restraints.getRestraintForce() # Get Force object incorporating restraints
                reference_system.addForce(force)
                metadata['standard_state_correction'] = complex_restraints.getStandardStateCorrection() # standard state correction in kT
            else:
                # For periodic systems, we do not use a restraint, but must add a standard state correction for the box volume.
                # TODO: What if the box volume fluctuates during the simulation?
                box_vectors = reference_system.getDefaultPeriodicBoxVectors()
                box_volume = thermodynamic_state._volume(box_vectors)
                STANDARD_STATE_VOLUME = 1660.53928 * angstroms**3
                metadata['standard_state_correction'] = numpy.log(STANDARD_STATE_CORRECTION / box_volume) # TODO: Check sign.

        if not protocols:
            # Select default protocols for alchemical transformation.
            # TODO: Allow protocols to be intelligently selected based on the system details?
            protocols = dict()
            protocols['vacuum'] = AbsoluteAlchemicalFactory.defaultVacuumProtocol()
            if is_periodic:
                protocols['solvent'] = AbsoluteAlchemicalFactory.defaultSolventProtocolExplicit()
                protocols['complex'] = AbsoluteAlchemicalFactory.defaultComplexProtocolExplicit()
            else:
                protocols['solvent'] = AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()
                protocols['complex'] = AbsoluteAlchemicalFactory.defaultComplexProtocolImplicit()

        # Create alchemically-modified states using alchemical factory.
        factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=atom_indices['ligand'], receptor_atoms=atom_indices['receptor'])
        systems = factory.createPerturbedSystems(self.protocols[phase])

        # Randomize ligand position if requested, but only for implicit solvent systems.
        if self.randomize_ligand and (not is_periodic):
            if self.verbose: print "Randomizing ligand positions and excluding overlapping configurations..."
            randomized_positions = list()
            sigma = 2*complex_restraints.getReceptorRadiusOfGyration()
            close_cutoff = 1.5 * units.angstrom # TODO: Allow this to be specified by user.
            nstates = len(systems)
            for state_index in range(nstates):
                positions = positions[numpy.random.randint(0, len(positions))]
                new_positions = ModifiedHamiltonianExchange.randomize_ligand_position(positions, atom_indices['receptor'], atom_indices['ligand'], sigma, close_cutoff)
                randomized_positions.append(new_positions)
            positions = randomized_positions

        # Construct


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
        context = openmm.Context(system, integrator)
        platform = context.getPlatform()
        del context, integrator
        return platform

    def run(self, niterations=None, mpicomm=None, options=None):
        """
        Run a free energy calculation.

        Parameters
        ----------
        niterations : int, optional, default=None
           If specified, only this many iterations will be run for each phase.
           This is useful for running simulation incrementally, but may incur a good deal of overhead.
        mpicomm : MPI communicator, optional, default=None
           If an MPI communicator is passed, an MPI simulation will be attempted.
        options : dict of str, optional, default=None
           If specified, these options will override any other options.

        """

        # Make sure we've been properly initialized first.
        if not self._initialized:
            raise Exception("Yank must first be initialized by either resume() or create().")

        # Handle some logistics necessary for MPI.
        if mpicomm:
            # Turn off output from non-root nodes:
            if not (mpicomm.rank==0):
                verbose = False

            # Make sure each thread's random number generators have unique seeds.
            # TODO: Do we need to store seed in repex object?
            seed = numpy.random.randint(sys.maxint - mpicomm.size) + mpicomm.rank
            numpy.random.seed(seed)

        # Run all phases sequentially.
        # TODO: Divide up MPI resources among the phases?
        for phase in self._phases:
            store_filename = self._store_filenames[phase]
            # Resume simulation from store file.
            simulation = ModifiedHamiltonianExchange(store_filename=store_filename, mpicomm=mpicomm, protocol=options)
            # TODO: We may need to manually update run options here if protocol=options above does not behave as expected.
            simulation.run(niterations=niterations)
            # Clean up to ensure we close files, contexts, etc.
            del simulation

        return

    def status(self):
        """
        Determine status of all phases of Yank calculation.

        Returns
        -------
        status : dict
           status[phase] contains a dict with information about progress in the calculation for 'phase'
           If Yank has not been initialized, the status is False.

        """
        if not self._initialized: return False

        status = dict()
        for phase in self._phases:
            status[phase] = ModifiedReplicaExchange.status_from_store(self._store_filenames[phase])

        return status

    def analyze(self, verbose=False):
        """
        Programmatic interface to retrieve the results of a YANK free energy calculation.

        Parameters
        ----------
        verbose : bool, optional, default=False
           If True, will print verbose progress information (default: False)

        Returns
        -------
        results : dict
           results[phase][component] is the estimate of 'component' of thermodynamic leg 'phase'
           'component' can be one of ['DeltaF', 'dDeltaF', 'DeltaH', 'dDeltaH']
           DeltaF is the estimated free energy difference
           dDeltaF is the statistical uncertainty in DeltaF (one standard error)
           DeltaH is the estimated enthalpy difference
           dDeltaH is the statistical uncertainty in DeltaH (one standard error)
           all quantites are reported in units are kT

        """

        # TODO: Can we simplify this code by pushing more into analyze.py or repex.py?

        import analyze
        from pymbar import pymbar, timeseries
        import netCDF4 as netcdf

        # Storage for results.
        results = dict()

        if verbose: print "Analyzing simulation data..."

        # Process each netcdf file in output directory.
        for phase in self._phases:
            fullpath = self._store_filenames[phase]

            # Skip if the file doesn't exist.
            if (not os.path.exists(fullpath)): continue

            # Analyze this leg.
            simulation = ModifiedHamiltonianExchange(store_filename=store_filename, mpicomm=mpicomm, protocol=options)
            analysis = simulation.analyze()
            del simulation

            # Store results.
            results[phase] = analysis

        # TODO: Analyze binding or hydration, depending on what phases are present.
        # TODO: Include effects of analytical contributions.
        phases_available = results.keys()

        if set('solvent', 'vacuum').issubset(phases_available):
            # SOLVATION FREE ENERGY
            results['solvation'] = dict()

            results['solvation']['Delta_f'] = results['solvent']['Delta_f'] + results['vacuum']['Delta_f']
            results['solvation']['dDelta_f'] = numpy.sqrt(results['solvent']['dDelta_f']**2 + results['vacuum']['Delta_f']**2)

        if set('ligand', 'complex').issubset(phases_available):
            # BINDING FREE ENERGY
            results['binding'] = dict()

            # Read standard state correction free energy.
            Delta_f_restraints = 0.0
            phase = 'complex'
            fullpath = os.path.join(source_directory, phase + '.nc')
            ncfile = netcdf.Dataset(fullpath, 'r')
            Delta_f_restraints = ncfile.groups['metadata'].variables['standard_state_correction'][0]
            ncfile.close()
            results['binding']['standard_state_correction'] = Delta_f_restraints

            # Compute binding free energy.
            results['binding']['Delta_f'] = results['solvent']['Delta_f'] - Delta_f_restraints - results['complex']['Delta_f']
            results['binding']['dDelta_f'] = numpy.sqrt(results['solvent']['dDelta_f']**2 + results['complex']['dDelta_f']**2)

        return results

