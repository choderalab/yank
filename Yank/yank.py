#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Yank
====

Interface for automated free energy calculations.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import copy
import glob
import inspect
import logging
logger = logging.getLogger(__name__)

import numpy as np

import simtk.unit as unit
import simtk.openmm as openmm

from alchemy import AbsoluteAlchemicalFactory
from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
from restraints import HarmonicReceptorLigandRestraint, FlatBottomReceptorLigandRestraint

#=============================================================================================
# YANK class to manage multiple thermodynamic transformations
#=============================================================================================

class Yank(object):
    """
    A class for managing alchemical replica-exchange free energy calculations.

    """

    default_parameters = {
        'restraint_type': 'flat-bottom',
        'randomize_ligand': False,
        'randomize_ligand_sigma_multiplier': 2.0,
        'randomize_ligand_close_cutoff': 1.5 * unit.angstrom,
        'mc_displacement_sigma': 10.0 * unit.angstroms
    }

    def __init__(self, store_directory, mpicomm=None, **kwargs):
        """
        Initialize YANK object with default parameters.

        Parameters
        ----------
        store_directory : str
           The storage directory in which output NetCDF files are read or written.
        mpicomm : MPI communicator, optional
           If an MPI communicator is passed, an MPI simulation will be attempted.
        restraint_type : str, optional
           Restraint type to add between protein and ligand. Supported types are
           'flat-bottom' and 'harmonic'. The second one is available only in
           implicit solvent (default: 'flat-bottom').
        randomize_ligand : bool, optional
           Randomize ligand position when True. Not available in explicit solvent
           (default: False).
        randomize_ligand_close_cutoff : simtk.unit.Quantity (units: length), optional
           Cutoff for ligand position randomization (default: 1.5*unit.angstrom).
        randomize_ligand_sigma_multiplier : float, optional
           Multiplier for ligand position randomization displacement (default: 2.0).
        mc_displacement_sigma : simtk.unit.Quantity (units: length), optional
           Maximum displacement for Monte Carlo moves that augment Langevin dynamics
           (default: 10.0*unit.angstrom).

        Other Parameters
        ----------------
        **kwargs
           More options to pass to the ReplicaExchange or AlchemicalFactory classes
           on initialization.

        See Also
        --------
        ReplicaExchange.default_parameters : extra parameters accepted.

        """

        # Copy kwargs to avoid modifications
        parameters = copy.deepcopy(kwargs)

        # Record that we are not yet initialized.
        self._initialized = False

        # Store output directory.
        self._store_directory = store_directory

        # Save MPI communicator
        self._mpicomm = mpicomm

        # Set internal variables.
        self._phases = list()
        self._store_filenames = dict()

        # Default alchemical protocols.
        self.default_protocols = dict()
        self.default_protocols['vacuum'] = AbsoluteAlchemicalFactory.defaultVacuumProtocol()
        self.default_protocols['solvent-implicit'] = AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()
        self.default_protocols['complex-implicit'] = AbsoluteAlchemicalFactory.defaultComplexProtocolImplicit()
        self.default_protocols['solvent-explicit'] = AbsoluteAlchemicalFactory.defaultSolventProtocolExplicit()
        self.default_protocols['complex-explicit'] = AbsoluteAlchemicalFactory.defaultComplexProtocolExplicit()

        # Store Yank parameters
        for option_name, default_value in self.default_parameters.items():
            setattr(self, '_' + option_name, parameters.pop(option_name, default_value))

        # Store repex parameters
        self._repex_parameters = {par: parameters.pop(par) for par in
                                  ModifiedHamiltonianExchange.default_parameters
                                  if par in parameters}

        # Store AlchemicalFactory parameters
        self._alchemy_parameters = {par: parameters.pop(par) for par in
                                    inspect.getargspec(AbsoluteAlchemicalFactory.__init__).args
                                    if par in parameters}

        # Check for unknown parameters
        if len(parameters) > 0:
            raise TypeError('got an unexpected keyword arguments {}'.format(
                ', '.join(parameters.keys())))

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

    def resume(self, phases=None):
        """
        Resume an existing set of alchemical free energy calculations found in current store directory.

        Parameters
        ----------
        phases : list of str, optional, default=None
           The list of calculation phases (e.g. ['solvent', 'complex']) to resume.
           If not specified, all simulations in the store_directory will be resumed.

        """
        # If no phases specified, construct a list of phases from the filename prefixes in the store directory.
        if phases is None:
            phases = self._find_phases_in_store_directory()
        self._phases = phases

        # Construct store filenames.
        self._store_filenames = { phase : os.path.join(self._store_directory, phase + '.nc') for phase in self._phases }

        # Ensure we can resume from each store file, processing override options.
        for phase in self._phases:
            # TODO: Use resume capabaility of repex to resume and modify any parameters we can change.
            store_filename = self._store_filenames[phase]
            pass

        # Record we are now initialized.
        self._initialized = True

        return

    def create(self, phases, systems, positions, atom_indices, thermodynamic_state, protocols=None):
        """
        Set up a new set of alchemical free energy calculations for the specified phases.

        Parameters
        ----------
        store_directory : str
           The storage directory in which output NetCDF files are read or written.
        phases : list of str, optional, default=None
           The list of calculation phases (e.g. ['solvent', 'complex']) to run.
           If resuming, will resume from all NetCDF files ('*.nc') in the store_directory unless specific phases are given.
        systems : dict of simtk.openmm.System, optional, default=None
           A dict of System objects for each phase, e.g. systems['solvent'] is for solvent phase.
        positions : dict of simtk.unit.Quantity arrays (np or Python) with units compatible with nanometers, or dict of lists, optional, default=None
           A dict of positions corresponding to each phase, e.g. positions['solvent'] is a single set of positions or list of positions to seed replicas.
           Shape must be natoms x 3, with natoms matching number of particles in corresponding system.
        atom_indices : dict of dict list of int, optional, default=None
           atom_indices[phase][component] is a list of atom indices, for each component in ['ligand', 'receptor',
           'complex', 'solvent', 'ligand_counterions']
        thermodynamic_state : ThermodynamicState (System need not be defined), optional, default=None
           Thermodynamic state at which calculations are to be carried out
        protocols : dict of list of AlchemicalState, optional, default=None
           If specified, the alchemical protocol protocols[phase] will be used for phase 'phase' instead of the default.

        """

        logger.debug("phases: %s"  % phases)
        logger.debug("systems: %s" % systems.keys())
        logger.debug("positions: %s" % positions.keys())
        logger.debug("atom_indices: %s" % atom_indices.keys())
        logger.debug("thermodynamic_state: %s" % thermodynamic_state)

        # Abort if there are files there already but initialization was requested.
        for phase in phases:
            store_filename = os.path.join(self._store_directory, phase + '.nc')
            if os.path.exists(store_filename):
                raise Exception("Store filename %s already exists." % store_filename)

        # Create new repex simulations.
        for phase in phases:
            self._create_phase(phase, systems[phase], positions[phase], atom_indices[phase], thermodynamic_state, protocols=protocols)

        # Record that we are now initialized.
        self._initialized = True

        return

    def _is_periodic(self, system):
        """
        Report whether a given system is periodic or not.

        Parameters
        ----------
        system : simtk.openmm.System
           The system object to examine for periodic forces.

        Returns
        -------
        is_periodic : bool
           True is returned if a NonbondedForce object is present with getNonBondedMethod() reporting one of [CutoffPeriodic, Ewald, PME]

        """

        is_periodic = False
        forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
        if forces['NonbondedForce'].getNonbondedMethod in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
            is_periodic = True
        return is_periodic

    def _create_phase(self, phase, reference_system, positions, atom_indices, thermodynamic_state, protocols=None):
        """
        Create a repex object for a specified phase.

        Parameters
        ----------
        phase : str
           The phase being initialized (one of ['complex', 'solvent', 'vacuum'])
        reference_system : simtk.openmm.System
           The reference system object from which alchemical intermediates are to be construcfted.
        positions : list of simtk.unit.Qunatity objects containing (natoms x 3) positions (as np or lists)
           The list of positions to be used to seed replicas in a round-robin way.
        atom_indices : dict
           atom_indices[phase][component] is the set of atom indices associated with component, where component
           is ['ligand', 'receptor', 'complex', 'solvent', 'ligand_counterions']
        thermodynamic_state : ThermodynamicState
           Thermodynamic state from which reference temperature and pressure are to be taken.
        protocols : dict of list of AlchemicalState, optional, default=None
           If specified, the alchemical protocol protocols[phase] will be used for phase 'phase' instead of the default.

        """

        # We add default repex options only on creation, on resume repex will pick them from the store file
        repex_parameters = {
            'number_of_equilibration_iterations': 0,
            'number_of_iterations': 100,
            'timestep': 2.0 * unit.femtoseconds,
            'collision_rate': 5.0 / unit.picoseconds,
            'minimize': False,
            'show_mixing_statistics': True,  # this causes slowdown with iteration and should not be used for production
            'displacement_sigma': 1.0 * unit.nanometers  # attempt to displace ligand by this stddev will be made each iteration
        }
        repex_parameters.update(self._repex_parameters)

        # Make sure positions argument is a list of coordinate snapshots.
        if hasattr(positions, 'unit'):
            # Wrap in list.
            positions = [positions]

        # Check the dimensions of positions.
        for index in range(len(positions)):
            # Make sure it is recast as a np array.
            positions[index] = unit.Quantity(np.array(positions[index] / positions[index].unit), positions[index].unit)

            [natoms, ndim] = (positions[index] / positions[index].unit).shape
            if natoms != reference_system.getNumParticles():
                raise Exception("positions argument must be a list of simtk.unit.Quantity of (natoms,3) lists or np array with units compatible with nanometers.")

        # Create metadata storage.
        metadata = dict()

        # Make a deep copy of the reference system so we don't accidentally modify it.
        reference_system = copy.deepcopy(reference_system)

        # TODO: Use more general approach to determine whether system is periodic.
        is_periodic = self._is_periodic(reference_system)

        # Make sure pressure is None if not periodic.
        if not is_periodic: thermodynamic_state.pressure = None

        # Compute standard state corrections for complex phase.
        metadata['standard_state_correction'] = 0.0
        # TODO: Do we need to include a standard state correction for other phases in periodic boxes?
        if phase == 'complex-implicit':
            # Impose restraints for complex system in implicit solvent to keep ligand from drifting too far away from receptor.
            logger.debug("Creating receptor-ligand restraints...")
            reference_positions = positions[0]
            if self._restraint_type == 'harmonic':
                restraints = HarmonicReceptorLigandRestraint(thermodynamic_state, reference_system, reference_positions, atom_indices['receptor'], atom_indices['ligand'])
            elif self._restraint_type == 'flat-bottom':
                restraints = FlatBottomReceptorLigandRestraint(thermodynamic_state, reference_system, reference_positions, atom_indices['receptor'], atom_indices['ligand'])
            else:
                raise Exception("restraint_type of '%s' is not supported." % self._restraint_type)

            force = restraints.getRestraintForce() # Get Force object incorporating restraints
            reference_system.addForce(force)
            metadata['standard_state_correction'] = restraints.getStandardStateCorrection() # standard state correction in kT
        elif phase == 'complex-explicit':
            # For periodic systems, we do not use a restraint, but must add a standard state correction for the box volume.
            # TODO: What if the box volume fluctuates during the simulation?
            box_vectors = reference_system.getDefaultPeriodicBoxVectors()
            box_volume = thermodynamic_state._volume(box_vectors)
            STANDARD_STATE_VOLUME = 1660.53928 * unit.angstrom**3
            metadata['standard_state_correction'] = - np.log(STANDARD_STATE_VOLUME / box_volume)

        # Use default alchemical protocols if not specified.
        if not protocols:
            protocols = self.default_protocols

        # Create alchemically-modified states using alchemical factory.
        logger.debug("Creating alchemically-modified states...")
        try:
            alchemical_indices = atom_indices['ligand_counterions'] + atom_indices['ligand']
        except KeyError:
            alchemical_indices = atom_indices['ligand']
        factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=alchemical_indices,
                                            **self._alchemy_parameters)
        alchemical_states = protocols[phase]
        alchemical_system = factory.alchemically_modified_system
        thermodynamic_state.system = alchemical_system

        # Check systems for finite energies.
        # TODO: Refactor this into another function.
        finite_energy_check = False
        if finite_energy_check:
            logger.debug("Checking energies are finite for all alchemical systems.")
            integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
            context = openmm.Context(alchemical_system, integrator)
            context.setPositions(positions[0])
            for alchemical_state in alchemical_states:
                AbsoluteAlchemicalFactory.perturbContext(context, alchemical_state)
                potential = context.getState(getEnergy=True).getPotentialEnergy()
                if np.isnan(potential / unit.kilocalories_per_mole):
                    raise Exception("Energy for system %d is NaN." % index)
            del context, integrator
            logger.debug("All energies are finite.")

        # Randomize ligand position if requested, but only for implicit solvent systems.
        if self._randomize_ligand and (phase == 'complex-implicit'):
            logger.debug("Randomizing ligand positions and excluding overlapping configurations...")
            randomized_positions = list()
            nstates = len(alchemical_states)
            for state_index in range(nstates):
                positions_index = np.random.randint(0, len(positions))
                current_positions = positions[positions_index]
                new_positions = ModifiedHamiltonianExchange.randomize_ligand_position(current_positions,
                                                                                      atom_indices['receptor'], atom_indices['ligand'],
                                                                                      self._randomize_ligand_sigma_multiplier * restraints.getReceptorRadiusOfGyration(),
                                                                                      self._randomize_ligand_close_cutoff)
                randomized_positions.append(new_positions)
            positions = randomized_positions
        if self._randomize_ligand and (phase == 'complex-explicit'):
            logger.warning("Ligand randomization requested, but will not be performed for explicit solvent simulations.")

        # Identify whether any atoms will be displaced via MC, unless option is turned off.
        mc_atoms = None
        if self._mc_displacement_sigma:
            mc_atoms = list()
            if 'ligand' in atom_indices:
                mc_atoms = atom_indices['ligand']

        # Set up simulation.
        # TODO: Support MPI initialization?
        logger.debug("Creating replica exchange object...")
        store_filename = os.path.join(self._store_directory, phase + '.nc')
        self._store_filenames[phase] = store_filename
        simulation = ModifiedHamiltonianExchange(store_filename)
        simulation.create(thermodynamic_state, alchemical_states, positions,
                          displacement_sigma=self._mc_displacement_sigma, mc_atoms=mc_atoms,
                          options=repex_parameters, metadata=metadata)

        # Initialize simulation.
        # TODO: Use the right scheme for initializing the simulation without running.
        #logger.debug("Initializing simulation...")
        #simulation.run(0)

        # Clean up simulation.
        del simulation

        # Add to list of phases that have been set up.
        self._phases.append(phase)

        return

    def run(self, niterations_to_run=None):
        """
        Run a free energy calculation.

        Parameters
        ----------
        niterations_to_run : int, optional, default=None
           If specified, only this many iterations will be run for each phase.
           This is useful for running simulation incrementally, but may incur a good deal of overhead.

        """

        # Make sure we've been properly initialized first.
        if not self._initialized:
            raise Exception("Yank must first be initialized by either resume() or create().")

        # Handle some logistics necessary for MPI.
        if self._mpicomm is not None:
            logger.debug("yank.run starting for MPI...")
            # Make sure each thread's random number generators have unique seeds.
            # TODO: Do we need to store seed in repex object?
            seed = np.random.randint(4294967295 - self._mpicomm.size) + self._mpicomm.rank
            np.random.seed(seed)

        # Run all phases sequentially.
        # TODO: Divide up MPI resources among the phases so they can run simultaneously?
        for phase in self._phases:
            store_filename = self._store_filenames[phase]
            # Resume simulation from store file.
            simulation = ModifiedHamiltonianExchange(store_filename=store_filename, mpicomm=self._mpicomm)
            simulation.resume(options=self._repex_parameters)
            # TODO: We may need to manually update run options here if options=options above does not behave as expected.
            simulation.run(niterations_to_run=niterations_to_run)
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
           If simulation has not been initialized by a call to resume() or create(), None is returned.

        """
        if not self._initialized: return None

        status = dict()
        for phase in self._phases:
            status[phase] = ModifiedReplicaExchange.status_from_store(self._store_filenames[phase])

        return status

    def analyze(self):
        """
        Programmatic interface to retrieve the results of a YANK free energy calculation.

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
           If simulation has not been initialized by a call to resume() or create(), None is returned.

        """
        if not self._initialized: return None

        # TODO: Can we simplify this code by pushing more into analyze.py or repex.py?

        import analyze
        from pymbar import MBAR, timeseries
        import netCDF4 as netcdf

        # Storage for results.
        results = dict()

        logger.debug("Analyzing simulation data...")

        # Process each netcdf file in output directory.
        for phase in self._phases:
            fullpath = self._store_filenames[phase]

            # Skip if the file doesn't exist.
            if (not os.path.exists(fullpath)): continue

            # Read this phase.
            simulation = ModifiedHamiltonianExchange(fullpath)
            simulation.resume()

            # Analyze this phase.
            analysis = simulation.analyze()

            # Retrieve standard state correction.
            analysis['standard_state_correction'] = simulation.metadata['standard_state_correction']

            # Store results.
            results[phase] = analysis

            # Clean up.
            del simulation

        # TODO: Analyze binding or hydration, depending on what phases are present.
        # TODO: Include effects of analytical contributions.
        phases_available = results.keys()

        if set(['solvent', 'vacuum']).issubset(phases_available):
            # SOLVATION FREE ENERGY
            results['solvation'] = dict()

            results['solvation']['Delta_f'] = results['solvent']['Delta_f'] + results['vacuum']['Delta_f']
            # TODO: Correct in different ways depending on what reference conditions are desired.
            results['solvation']['Delta_f'] += results['solvent']['standard_state_correction'] + results['vacuum']['standard_state_correction']

            results['solvation']['dDelta_f'] = np.sqrt(results['solvent']['dDelta_f']**2 + results['vacuum']['Delta_f']**2)

        if set(['ligand', 'complex']).issubset(phases_available):
            # BINDING FREE ENERGY
            results['binding'] = dict()

            # Compute binding free energy.
            results['binding']['Delta_f'] = (results['solvent']['Delta_f'] + results['solvent']['standard_state_correction']) - (results['complex']['Delta_f'] + results['complex']['standard_state_correction'])
            results['binding']['dDelta_f'] = np.sqrt(results['solvent']['dDelta_f']**2 + results['complex']['dDelta_f']**2)

        return results
