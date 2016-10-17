#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Yank
====

Interface for automated free energy calculations.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import os.path
import copy
import inspect
import logging
import numpy as np

import simtk.unit as unit
import simtk.openmm as openmm

from alchemy import AbsoluteAlchemicalFactory
from .sampling import ModifiedHamiltonianExchange
from .restraints import createRestraints

from . import utils

logger = logging.getLogger(__name__)

# ==============================================================================
# Class that define a single thermodynamic leg (phase) of the calculation
# ==============================================================================


class AlchemicalPhase(object):
    """A single thermodynamic leg (phase) of an alchemical free energy calculation.

    Attributes
    ----------
    name : str
        The name of the alchemical phase.
    reference_system : simtk.openmm.System
        The reference system object from which alchemical intermediates are
        to be constructed.
    reference_topology : simtk.openmm.app.Topology
        The topology object for the reference_system.
    positions : list of simtk.unit.Quantity natoms x 3 array with units of length
        Atom positions for the system used to seed replicas in a round-robin way.
    atom_indices : dict of list of int
        atom_indices[component] is the set of atom indices associated with
        component, where component is one of ('ligand', 'receptor', 'complex',
        'solvent', 'ligand_counterions').
    protocol : list of AlchemicalState
        The alchemical protocol used for the calculation.

    """
    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        # Make sure positions argument is a list of coordinate snapshots
        if hasattr(value, 'unit'):
            value = [value]  # Wrap in list

        self._positions = [0 for _ in range(len(value))]

        # Make sure that positions are recast as np arrays
        for i in range(len(value)):
            positions_unit = value[i].unit
            self._positions[i] = unit.Quantity(np.array(value[i] / positions_unit),
                                               positions_unit)

    def __init__(self, name, reference_system, reference_topology,
                 positions, atom_indices, protocol):
        """Constructor.

        Parameters
        ----------
        name : str
            The name of the phase being initialized.
        reference_system : simtk.openmm.System
            The reference system object from which alchemical intermediates are
            to be constructed.
        reference_topology : simtk.openmm.app.Topology
            The topology object for the reference_system.
        positions : (list of) simtk.unit.Quantity natoms x 3 array with units of length
            Initial atom positions for the system. If a single simtk.unit.Quantity
            natoms x 3 array, all replicas start from the same positions. If a
            list of simtk.unit.Quantity natoms x 3 arrays are found, they are
            used to seed replicas in a round-robin way.
        atom_indices : dict of list of int
            atom_indices[component] is the set of atom indices associated with
            component, where component is one of ('ligand', 'receptor', 'complex',
            'solvent', 'ligand_counterions').
        protocol : list of AlchemicalState
            The alchemical protocol used for the calculation.

        """
        self.name = name
        self.reference_system = reference_system
        self.reference_topology = reference_topology
        self.positions = positions
        self.atom_indices = atom_indices
        self.protocol = protocol


# ==============================================================================
# YANK class to manage multiple thermodynamic transformations
# ==============================================================================

class Yank(object):
    """
    A class for managing alchemical replica-exchange free energy calculations.

    """

    default_parameters = {
        'restraint_type': 'FlatBottom',
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
           Restraint type to add between protein and ligand.
           Supported types are 'FlatBottom' and 'Harmonic'
           The second one is available only in implicit solvent (default: 'FlatBottom').
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
            phases = utils.find_phases_in_store_directory(self._store_directory)
        self._phases = phases.keys()

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

    def create(self, thermodynamic_state, *alchemical_phases):
        """
        Set up a new set of alchemical free energy calculations for the specified phases.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState (System need not be defined)
            Thermodynamic state from which reference temperature and pressure are to be taken.
        *alchemical_phases :
            Variable list of AlchemicalPhase objects to create.

        """
        # Make a deep copy of thermodynamic state.
        thermodynamic_state = copy.deepcopy(thermodynamic_state)

        logger.debug("phases: {}".format([phase.name for phase in alchemical_phases]))
        logger.debug("thermodynamic_state: {}".format(thermodynamic_state))

        # Initialization checks
        for phase in alchemical_phases:
            # Abort if there are files there already but initialization was requested.
            store_filename = os.path.join(self._store_directory, phase.name + '.nc')
            if os.path.exists(store_filename):
                raise RuntimeError("Store filename %s already exists." % store_filename)

            # Abort if there are no atoms to alchemically modify
            if len(phase.atom_indices['ligand']) == 0:
                raise ValueError('Ligand atoms are not specified.')

        # Create store directory if needed
        if not os.path.isdir(self._store_directory):
            os.mkdir(self._store_directory)

        # Create new repex simulations.
        for phase in alchemical_phases:
            self._create_phase(thermodynamic_state, phase)

        # Record that we are now initialized.
        self._initialized = True

        return

    def _create_phase(self, thermodynamic_state, alchemical_phase):
        """
        Create a repex object for a specified phase.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState (System need not be defined)
            Thermodynamic state from which reference temperature and pressure are to be taken.
        alchemical_phase : AlchemicalPhase
           The alchemical phase to be created.

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

        # Convenience variables
        positions = alchemical_phase.positions
        reference_system = copy.deepcopy(alchemical_phase.reference_system)
        atom_indices = alchemical_phase.atom_indices
        alchemical_states = alchemical_phase.protocol

        # Check the dimensions of positions.
        for index in range(len(positions)):
            n_atoms, _ = (positions[index] / positions[index].unit).shape
            if n_atoms != reference_system.getNumParticles():
                err_msg = "Phase {}: number of atoms in positions {} and and " \
                          "reference system differ ({} and {} respectively)"
                err_msg.format(alchemical_phase.name, index, n_atoms,
                               reference_system.getNumParticles())
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        # Inizialize metadata storage.
        metadata = dict()

        # TODO: Use more general approach to determine whether system is periodic.
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        is_complex_explicit = len(atom_indices['receptor']) > 0 and is_periodic
        is_complex_implicit = len(atom_indices['receptor']) > 0 and not is_periodic

        # Make sure pressure is None if not periodic.
        if not is_periodic:
            thermodynamic_state.pressure = None
        # If temperature and pressure are specified, make sure MonteCarloBarostat is attached.
        elif thermodynamic_state.temperature and thermodynamic_state.pressure:
            forces = { reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index) for index in range(reference_system.getNumForces()) }

            if 'MonteCarloAnisotropicBarostat' in forces:
                raise Exception('MonteCarloAnisotropicBarostat is unsupported.')

            if 'MonteCarloBarostat' in forces:
                logger.debug('MonteCarloBarostat found: Setting default temperature and pressure.')
                barostat = forces['MonteCarloBarostat']
                # Set temperature and pressure.
                try:
                    barostat.setDefaultTemperature(thermodynamic_state.temperature)
                except AttributeError:  # versions previous to OpenMM7.1
                    barostat.setTemperature(thermodynamic_state.temperature)
                barostat.setDefaultPressure(thermodynamic_state.pressure)
            else:
                # Create barostat and add it to the system if it doesn't have one already.
                logger.debug('MonteCarloBarostat not found: Creating one.')
                barostat = openmm.MonteCarloBarostat(thermodynamic_state.pressure, thermodynamic_state.temperature)
                reference_system.addForce(barostat)

        # Store a serialized copy of the reference system.
        metadata['reference_system'] = openmm.XmlSerializer.serialize(reference_system)
        metadata['topology'] = utils.serialize_topology(alchemical_phase.reference_topology)

        # Create a copy of the system for which the fully-interacting energy is to be computed.
        # For explicit solvent calculations, an enlarged cutoff is used to account for the anisotropic dispersion correction.

        # fully_interacting_system = copy.deepcopy(reference_system)
        reference_system_LJ = copy.deepcopy(reference_system)
        forces_to_remove = []
        for forceIndex in range(reference_system_LJ.getNumForces()):
            force = reference_system_LJ.getForce(forceIndex)
            if isinstance(force, openmm.NonbondedForce):
                for particle in range(force.getNumParticles()):
                    q, sigma, epsilon = force.getParticleParameters(particle)
                    force.setParticleParameters(particle, 0, sigma, epsilon)
                for exception in range(force.getNumExceptions()):
                    particle1, particle2, chargeprod, epsilon, sigma = force.getExceptionParameters(exception)
                    force.setExceptionParameters(exception, particle1, particle2, 0, sigma, epsilon)
            else:
                # Queue force to remove if not a NB fore
                forces_to_remove.append(forceIndex)
        # Remove all but NonbondedForce
        # If done in preveious loop, nuber of forces change so indices change
        for forceIndex in forces_to_remove[::-1]:
            reference_system_LJ.removeForce(forceIndex)

        reference_system_LJ_expanded = copy.deepcopy(reference_system_LJ)
        if is_periodic:
            # Determine minimum box side dimension
            box_vectors = reference_system_LJ_expanded.getDefaultPeriodicBoxVectors()
            min_box_dimension = min([max(vector) for vector in box_vectors])

            # Expand cutoff to minimize artifact and verify that box is big enough.
            # If we use a barostat we leave more room for volume fluctuations or
            # we risk fatal errors. If we don't use a barostat, OpenMM will raise
            # the appropriate exception on context creation.
            max_allowed_cutoff = 16 * unit.angstroms
            max_switching_distance = max_allowed_cutoff - (1 * unit.angstrom)
            # TODO: Make max_allowed_cutoff an option
            if thermodynamic_state.pressure and min_box_dimension < 2.25 * max_allowed_cutoff:
                raise RuntimeError('Barostated box sides must be at least 36 Angstroms '
                                   'to correct for missing dispersion interactions')

            logger.debug('Setting cutoff for fully interacting system to maximum allowed (%s)' % str(max_allowed_cutoff))

            # Expanded cutoff LJ system if needed
            # We don't want to reduce the cutoff if its already large
            for force in reference_system_LJ_expanded.getForces():
                try:
                    if force.getCutoffDistance() < max_allowed_cutoff:
                        force.setCutoffDistance(max_allowed_cutoff)
                        # Set switch distance
                        # We don't need to check if we are using a switch since there is a setting for that.
                        force.setSwitchingDistance(max_switching_distance)
                except:
                    pass
                try:
                    if force.getCutoff() < max_allowed_cutoff:
                        force.setCutoff(max_allowed_cutoff)
                except:
                    pass

        # Construct thermodynamic states
        reference_state = copy.deepcopy(thermodynamic_state)
        reference_state.system = reference_system
        reference_LJ_state = copy.deepcopy(thermodynamic_state)
        reference_LJ_expanded_state = copy.deepcopy(thermodynamic_state)
        reference_LJ_state.system = reference_system_LJ
        reference_LJ_expanded_state.system = reference_system_LJ_expanded

        # Compute standard state corrections for complex phase.
        metadata['standard_state_correction'] = 0.0
        # TODO: Do we need to include a standard state correction for other phases in periodic boxes?
        if is_complex_implicit:
            # Impose restraints for complex system in implicit solvent to keep ligand from drifting too far away from receptor.
            logger.debug("Creating receptor-ligand restraints...")
            reference_positions = positions[0]
            restraints = createRestraints(self._restraint_type,
                alchemical_phase.reference_topology, thermodynamic_state, reference_system, reference_positions, atom_indices['receptor'], atom_indices['ligand'])
            force = restraints.getRestraintForce() # Get Force object incorporating restraints
            reference_system.addForce(force)
            metadata['standard_state_correction'] = restraints.getStandardStateCorrection() # standard state correction in kT
        elif is_complex_explicit:
            # For periodic systems, we do not use a restraint, but must add a standard state correction for the box volume.
            # TODO: What if the box volume fluctuates during the simulation?
            box_vectors = reference_system.getDefaultPeriodicBoxVectors()
            box_volume = thermodynamic_state._volume(box_vectors)
            STANDARD_STATE_VOLUME = 1660.53928 * unit.angstrom**3
            metadata['standard_state_correction'] = - np.log(STANDARD_STATE_VOLUME / box_volume)

        # Create alchemically-modified states using alchemical factory.
        logger.debug("Creating alchemically-modified states...")
        try:
            alchemical_indices = atom_indices['ligand_counterions'] + atom_indices['ligand']
        except KeyError:
            alchemical_indices = atom_indices['ligand']
        factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=alchemical_indices,
                                            **self._alchemy_parameters)
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
            for index, alchemical_state in enumerate(alchemical_states):
                AbsoluteAlchemicalFactory.perturbContext(context, alchemical_state)
                potential = context.getState(getEnergy=True).getPotentialEnergy()
                if np.isnan(potential / unit.kilocalories_per_mole):
                    raise Exception("Energy for system %d is NaN." % index)
            del context, integrator
            logger.debug("All energies are finite.")

        # Randomize ligand position if requested, but only for implicit solvent systems.
        if self._randomize_ligand and is_complex_implicit:
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
        if self._randomize_ligand and is_complex_explicit:
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
        store_filename = os.path.join(self._store_directory, alchemical_phase.name + '.nc')
        self._store_filenames[alchemical_phase.name] = store_filename
        simulation = ModifiedHamiltonianExchange(store_filename)
        simulation.create(thermodynamic_state, alchemical_states, positions,
                          displacement_sigma=self._mc_displacement_sigma, mc_atoms=mc_atoms,
                          options=repex_parameters, metadata=metadata,
                          reference_state = reference_state,
                          reference_LJ_state = reference_LJ_state,
                          reference_LJ_expanded_state = reference_LJ_expanded_state)

        # Initialize simulation.
        # TODO: Use the right scheme for initializing the simulation without running.
        #logger.debug("Initializing simulation...")
        #simulation.run(0)

        # Clean up simulation.
        del simulation

        # Add to list of phases that have been set up.
        self._phases.append(alchemical_phase.name)

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
            status[phase] = ModifiedHamiltonianExchange.status_from_store(self._store_filenames[phase])

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

        from . import analyze
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
