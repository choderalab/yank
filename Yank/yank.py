#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Yank
====

Interface for automated free energy calculations.

"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import os
import copy
import time
import inspect
import logging

import mdtraj
import pandas
import numpy as np
import openmmtools as mmtools
from simtk import unit, openmm

from yank import utils, pipeline
from yank.restraints import RestraintState, RestraintParameterError, V0

logger = logging.getLogger(__name__)


# ==============================================================================
# TOPOGRAPHY
# ==============================================================================

class Topography(object):
    """A class mapping and labelling the different components of a system.

    The object holds the topology of a system and offers convenience functions
    to identify its various parts such as solvent, receptor, ions and ligand
    atoms.

    A molecule should be labeled as a ligand, only if there is also a receptor.
    If there is only a single molecule its atom indices can be obtained from
    solute_atoms instead. In ligand-receptor system, solute_atoms provides the
    atom indices for both molecules.

    Parameters
    ----------
    topology : mdtraj.Topology or simtk.openmm.app.Topology
        The topology object specifying the system.
    ligand_atoms : iterable of int or str, optional
        The atom indices of the ligand. A string is interpreted as an mdtraj
        DSL specification of the ligand atoms.
    solvent_atoms : iterable of int or str, optional
        The atom indices of the solvent. A string is interpreted as an mdtraj
        DSL specification of the solvent atoms. If 'auto', a list of common
        solvent residue names will be used to automatically detect solvent
        atoms (default is 'auto').

    Attributes
    ----------
    ligand_atoms
    receptor_atoms
    solute_atoms
    solvent_atoms
    ions_atoms

    """
    def __init__(self, topology, ligand_atoms=None, solvent_atoms='auto'):
        # Determine if we need to convert the topology to mdtraj.
        if isinstance(topology, mdtraj.Topology):
            self._topology = topology
        else:
            self._topology = mdtraj.Topology.from_openmm(topology)

        # Handle default ligand atoms.
        if ligand_atoms is None:
            ligand_atoms = []

        # Once ligand and solvent atoms are defined, every other region is implied.
        self.solvent_atoms = solvent_atoms
        self.ligand_atoms = ligand_atoms

    @property
    def topology(self):
        """mdtraj.Topology: A copy of the topology (read-only)."""
        return copy.deepcopy(self._topology)

    @property
    def ligand_atoms(self):
        """The atom indices of the ligand.

        This can be empty if this Topography doesn't represent a receptor-ligand
        system. Use solute_atoms to obtain the atom indices of the molecule if
        this is the case.

        If assigned to a string, it will be interpreted as an mdtraj DSL specification
        of the atom indices.

        """
        return self._ligand_atoms

    @ligand_atoms.setter
    def ligand_atoms(self, value):
        self._ligand_atoms = self._resolve_atom_indices(value)

        # Safety check: with a ligand there should always be a receptor.
        if len(self._ligand_atoms) > 0 and len(self.receptor_atoms) == 0:
            raise ValueError('Specified ligand but cannot find '
                             'receptor atoms. Ligand: {}'.format(value))

    @property
    def receptor_atoms(self):
        """The atom indices of the receptor (read-only).

        This can be empty if this Topography doesn't represent a receptor-ligand
        system. Use solute_atoms to obtain the atom indices of the molecule if
        this is the case.

        """
        # If there's no ligand, there's no receptor.
        if len(self._ligand_atoms) == 0:
            return []

        # Create a set for fast searching.
        ligand_atomset = frozenset(self._ligand_atoms)
        # Receptor atoms are all solute atoms that are not ligand.
        return [i for i in self.solute_atoms if i not in ligand_atomset]

    @property
    def solute_atoms(self):
        """The atom indices of the non-solvent molecule(s) (read-only).

        Practically, this are all the indices of the atoms that are not considered
        solvent. In a receptor-ligand system, this includes the atom indices of
        both the receptor and the ligand.

        """
        # Create a set for fast searching.
        solvent_atomset = frozenset(self._solvent_atoms)
        # The solute is everything that is not solvent.
        return [i for i in range(self._topology.n_atoms) if i not in solvent_atomset]

    @property
    def solvent_atoms(self):
        """The atom indices of the solvent molecules.

        This includes eventual ions. If assigned to a string, it will be
        interpreted as an mdtraj DSL specification of the atom indices. If
        assigned to 'auto', a set of solvent auto indices is automatically
        built from common solvent residue names.

        """
        return self._solvent_atoms

    @solvent_atoms.setter
    def solvent_atoms(self, value):
        # If the user doesn't provide a solvent description,
        # we use a default set of resnames in mdtraj.
        if value == 'auto':
            solvent_resnames = mdtraj.core.residue_names._SOLVENT_TYPES
            self._solvent_atoms = [atom.index for atom in self._topology.atoms
                                   if atom.residue.name in solvent_resnames]
        else:
            self._solvent_atoms = self._resolve_atom_indices(value)

    @property
    def ions_atoms(self):
        """The indices of all ions atoms in the solvent (read-only)."""
        # Ions are all atoms of the solvent whose residue name show a charge.
        return [i for i in self._solvent_atoms
                if '-' in self._topology.atom(i).residue.name or
                '+' in self._topology.atom(i).residue.name]

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def __getstate__(self):
        # We serialize the MDTraj Topology through a pandas dataframe because
        # it doesn't implement __getstate__ and __setstate__ methods that
        # guarantee future-compatibility. This serialization protocol will be
        # compatible at least until the Topology API is broken.
        atoms, bonds = self._topology.to_dataframe()
        serialized_topology = {'atoms': atoms.to_json(orient='records'),
                               'bonds': bonds.tolist()}
        return dict(topology=serialized_topology,
                    ligand_atoms=self._ligand_atoms,
                    solvent_atoms=self._solvent_atoms)

    def __setstate__(self, serialization):
        topology_dict = serialization['topology']
        atoms = pandas.read_json(topology_dict['atoms'], orient='records')
        bonds = np.array(topology_dict['bonds'])
        self._topology = mdtraj.Topology.from_dataframe(atoms, bonds)
        self._ligand_atoms = serialization['ligand_atoms']
        self._solvent_atoms = serialization['solvent_atoms']

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _resolve_atom_indices(self, atoms_description):
        if isinstance(atoms_description, str):
            # Assume this is DSL selection. select() returns a numpy array
            # of int64 that we convert to python integers.
            atoms_description = self._topology.select(atoms_description).tolist()
        # Convert to a frozen set of indices.
        return atoms_description


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
    def __init__(self, sampler):
        self._sampler = sampler

    def create(self, thermodynamic_state, sampler_states, topography, protocol, storage,
               alchemical_regions=None, alchemical_factory=None, restraint=None,
               anisotropic_dispersion_cutoff=None, metadata=None):
        """
        Create a repex object for a specified phase.

        If `anisotropic_dispersion_cutoff` is different than `None`. The end states
        of the phase will be reweighted. The fully interacting state accounts for:
            1. The truncation of nonbonded interactions.
            2. The reciprocal space which is not modeled in alchemical states if
               an Ewald method is used for long-range interactions.
            3. The free energy of removing the restraint if there is a restraint
               and if the first step of the protocol has `lambda_restraints=1.0`.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState (System need not be defined)
            Thermodynamic state from which reference temperature and pressure are to be taken.
        alchemical_phase : AlchemicalPhase
           The alchemical phase to be created.
        restraint_type : str or None
           Restraint type to add between protein and ligand. Supported
           types are 'FlatBottom' and 'Harmonic'. The second one is
           available only in implicit solvent.

        """
        # Do not modify passed thermodynamic state.
        reference_thermodynamic_state = copy.deepcopy(thermodynamic_state)
        thermodynamic_state = copy.deepcopy(thermodynamic_state)
        reference_system = thermodynamic_state.system
        is_periodic = thermodynamic_state.is_periodic
        is_complex = len(topography.receptor_atoms) > 0

        # Make sure sampler_states is a list of SamplerStates.
        if isinstance(sampler_states, mmtools.states.SamplerState):
            sampler_states = [sampler_states]

        # Initialize metadata storage and handle default argument.
        # We'll use the sampler full name for resuming, the reference
        # thermodynamic state for minimization and the topography for
        # ligand randomization.
        if metadata is None:
            metadata = dict()
        sampler_full_name = utils.typename(self._sampler.__class__)
        metadata['sampler_full_name'] = sampler_full_name
        metadata['reference_state'] = mmtools.utils.serialize(thermodynamic_state)
        metadata['topography'] = mmtools.utils.serialize(topography)

        # Add default title if user hasn't specified.
        if 'title' not in metadata:
            default_title = ('Alchemical free energy calculation created '
                             'using yank.AlchemicalPhase and {} on {}')
            metadata['title'] = default_title.format(sampler_full_name,
                                                     time.asctime(time.localtime()))

        # Restraint and standard state correction.
        # ----------------------------------------

        # Add receptor-ligand restraint and compute standard state corrections.
        restraint_state = None
        metadata['standard_state_correction'] = 0.0
        if is_complex and restraint is not None:
            logger.debug("Creating receptor-ligand restraints...")

            try:
                restraint.restrain_state(thermodynamic_state)
            except RestraintParameterError:
                logger.debug('There are undefined restraint parameters. '
                             'Trying automatic parametrization.')
                restraint.determine_missing_parameters(thermodynamic_state,
                                                       sampler_states[0], topography)
                restraint.restrain_state(thermodynamic_state)
            correction = restraint.get_standard_state_correction(thermodynamic_state)  # in kT
            metadata['standard_state_correction'] = correction

            # Create restraint state that will be part of composable states.
            restraint_state = RestraintState(lambda_restraints=1.0)

        # Raise error if we can't find a ligand-receptor to apply the restraint.
        elif restraint is not None:
            raise RuntimeError("Cannot apply the restraint. No receptor-ligand "
                               "complex could be found. ")

        # For not-restrained ligand-receptor periodic systems, we must still
        # add a standard state correction for the box volume.
        elif is_complex and is_periodic:
            # TODO: What if the box volume fluctuates during the simulation?
            box_vectors = reference_system.getDefaultPeriodicBoxVectors()
            box_volume = mmtools.states._box_vectors_volume(box_vectors)
            metadata['standard_state_correction'] = - np.log(V0 / box_volume)

        # For implicit solvent/vacuum complex systems, we require a restraint
        # to keep the ligand from drifting too far away from receptor.
        elif is_complex and not is_periodic:
            raise ValueError('A receptor-ligand system in implicit solvent or '
                             'vacuum requires a restraint.')

        # Create alchemical states.
        # -------------------------

        # Handle default alchemical region.
        if alchemical_regions is None:
            alchemical_regions = self._build_default_alchemical_region(
                reference_system, topography, protocol)

        # Check that we have atoms to alchemically modify.
        if len(alchemical_regions.alchemical_atoms) == 0:
            raise ValueError("Couldn't find atoms to alchemically modify.")

        # Create alchemically-modified system using alchemical factory.
        logger.debug("Creating alchemically-modified states...")
        if alchemical_factory is None:
            factory = mmtools.alchemy.AlchemicalFactory()
        alchemical_system = factory.create_alchemical_system(thermodynamic_state.system,
                                                             alchemical_regions)

        # Create compound alchemically modified state to pass to sampler.
        thermodynamic_state.system = alchemical_system
        alchemical_state = mmtools.alchemy.AlchemicalState.from_system(alchemical_system)
        if restraint_state is not None:
            composable_states = [alchemical_state, restraint_state]
        else:
            composable_states = [alchemical_state]
        compound_state = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=thermodynamic_state, composable_states=composable_states)

        # Create all compound states to pass to sampler.create()
        # following the requested protocol.
        compound_states = []
        protocol_keys, protocol_values = zip(*protocol.items())
        for state_id, state_values in enumerate(zip(*protocol_values)):
            compound_states.append(copy.deepcopy(compound_state))
            for lambda_key, lambda_value in zip(protocol_keys, state_values):
                if hasattr(compound_state, lambda_key):
                    setattr(compound_states[state_id], lambda_key, lambda_value)
                else:
                    raise AttributeError('CompoundThermodynamicState object does not '
                                         'have protocol attribute {}'.format(lambda_key))

        # Expanded cutoff unsampled states.
        # ---------------------------------

        # TODO should we allow expanded states for non-periodic systems?
        expanded_cutoff_states = []
        if is_periodic and anisotropic_dispersion_cutoff is not None:
            # Create non-alchemically modified state with an expanded cutoff.
            # This must state must not have the restraint.
            reference_state_expanded = self._expand_state_cutoff(reference_thermodynamic_state,
                                                                 anisotropic_dispersion_cutoff)
            # Create the expanded cutoff decoupled state. The free energy
            # of removing the restraint will be taken into account with
            # the standard state correction.
            last_state_expanded = self._expand_state_cutoff(compound_states[-1],
                                                            anisotropic_dispersion_cutoff)
            expanded_cutoff_states = [reference_state_expanded, last_state_expanded]
        elif anisotropic_dispersion_cutoff is not None:
            logger.warning("The requested anisotropic dispersion correction "
                           "won't be computed for the non-periodic systems.")

        # Create simulation.
        # ------------------

        logger.debug("Creating sampler object...")
        self._sampler.create(compound_states, sampler_states, storage=storage,
                             unsampled_thermodynamic_states=expanded_cutoff_states,
                             metadata=metadata)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    @staticmethod
    def _expand_state_cutoff(thermodynamic_state, expanded_cutoff_distance):
        # Do not modify passed thermodynamic state.
        thermodynamic_state = copy.deepcopy(thermodynamic_state)
        system = thermodynamic_state.system

        # Determine minimum box side dimension. This assumes
        # that box vectors are aligned with the axes.
        box_vectors = system.getDefaultPeriodicBoxVectors()
        min_box_dimension = min([max(vector) for vector in box_vectors])

        # If we use a barostat we leave more room for volume fluctuations or
        # we risk fatal errors. If we don't use a barostat, OpenMM will raise
        # the appropriate exception on context creation.
        if (thermodynamic_state.pressure is not None and
                min_box_dimension < 2.25 * expanded_cutoff_distance):
            raise RuntimeError('Barostated box sides must be at least {} Angstroms '
                               'to correct for missing dispersion interactions'
                               ''.format(expanded_cutoff_distance/unit.angstrom * 2))

        logger.debug('Setting cutoff for fully interacting system to maximum '
                     'allowed {}'.format(expanded_cutoff_distance))

        # Expanded forces cutoff.
        for force in system.getForces():
            try:
                # We don't want to reduce the cutoff if it's already large.
                if force.getCutoffDistance() < expanded_cutoff_distance:
                    force.setCutoffDistance(expanded_cutoff_distance)

                    # Set switch distance. We don't need to check if we are
                    # using a switch since there is a setting for that.
                    switching_distance = expanded_cutoff_distance - 1.0*unit.angstrom
                    force.setSwitchingDistance(switching_distance)
            except AttributeError:
                pass

        # Return the new thermodynamic state with the expanded cutoff.
        thermodynamic_state.system = system
        return thermodynamic_state

    @staticmethod
    def _build_default_alchemical_region(system, topography, protocol):
        # TODO: we should probably have a second region that annihilate sterics of counterions.
        alchemical_region_kwargs = {}

        # Modify ligand if this is a receptor-ligand phase, or
        # solute if this is a transfer free energy calculation.
        if len(topography.ligand_atoms) > 0:
            alchemical_region_name = 'ligand_atoms'
        else:
            alchemical_region_name = 'solute_atoms'
        alchemical_atoms = getattr(topography, alchemical_region_name)

        # In periodic systems, we alchemically modify the ligand/solute
        # counterions to make sure that the solvation box is always neutral.
        if system.usesPeriodicBoundaryConditions():
            alchemical_counterions = pipeline.find_alchemical_counterions(
                system, topography, alchemical_region_name)
            alchemical_atoms += alchemical_counterions

            # Sort them by index for safety. We don't want to
            # accidentally exchange two atoms' positions.
            alchemical_atoms = sorted(alchemical_atoms)

        alchemical_region_kwargs['alchemical_atoms'] = alchemical_atoms

        # Check if we need to modify bonds/angles/torsions.
        for element_type in ['bonds', 'angles', 'torsions']:
            if 'lambda_' + element_type in protocol:
                modify_it = True
            else:
                modify_it = None
            alchemical_region_kwargs['alchemical_' + element_type] = modify_it

        # Create alchemical region.
        alchemical_region = mmtools.alchemy.AlchemicalRegion(**alchemical_region_kwargs)

        return alchemical_region


class AlchemicalPhaseOld(object):
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
        'randomize_ligand': False,
        'randomize_ligand_sigma_multiplier': 2.0,
        'randomize_ligand_close_cutoff': 1.5 * unit.angstrom,
        'mc_displacement_sigma': 10.0 * unit.angstroms,
        'anisotropic_dispersion_correction': True,
        'anisotropic_dispersion_cutoff': 16 * unit.angstroms

    }

    def __init__(self, store_directory, mpicomm=None, platform=None, **kwargs):
        """
        Initialize YANK object with default parameters.

        Parameters
        ----------
        store_directory : str
           The storage directory in which output NetCDF files are read or written.
        mpicomm : MPI communicator, optional
           If an MPI communicator is passed, an MPI simulation will be attempted.
        platform : simtk.openmm.Platform, optional
            Platform to use for execution. If None, the fastest available platform
            is used (default: None).
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
        anisotropic_dispersion_correction : bool, optional
           Correct for anisotropic dispersion effects by generating 2 additional
           states with expanded long-range cutoff distances to estimate energies
           at. These states are not simulated. (default: True)

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

        # Store platform to pas to replica exchange simulation object
        self._platform = platform

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
        self._phases = sorted(phases.keys())  # sorting ensures all MPI processes run the same phase

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

    def create(self, thermodynamic_state, *alchemical_phases, **kwargs):
        """
        Set up a new set of alchemical free energy calculations for the specified phases.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState (System need not be defined)
            Thermodynamic state from which reference temperature and pressure are to be taken.
        *alchemical_phases :
            Variable list of AlchemicalPhase objects to create.
        restraint_type : str, optional
           Restraint type to add between protein and ligand. Supported
           types are 'FlatBottom' and 'Harmonic'. The second one is
           available only in implicit solvent (default: None).

        """
        # Get kwargs
        restraint_type = kwargs.pop('restraint_type', None)
        if len(kwargs) != 0:
            raise TypeError('got unexpected keyword arguments {}'.format(kwargs))

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
            self._create_phase(thermodynamic_state, phase, restraint_type)

        # Record that we are now initialized.
        self._initialized = True

        return

    def _create_phase(self, thermodynamic_state, alchemical_phase, restraint_type):
        """
        Create a repex object for a specified phase.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState (System need not be defined)
            Thermodynamic state from which reference temperature and pressure are to be taken.
        alchemical_phase : AlchemicalPhase
           The alchemical phase to be created.
        restraint_type : str or None
           Restraint type to add between protein and ligand. Supported
           types are 'FlatBottom' and 'Harmonic'. The second one is
           available only in implicit solvent.

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
        is_complex = len(atom_indices['receptor']) > 0
        is_complex_explicit = is_complex and is_periodic
        is_complex_implicit = is_complex and not is_periodic

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

        # For explicit solvent calculations, we create a copy of the system for
        # which the fully-interacting energy is to be computed. An enlarged cutoff
        # is used to account for the anisotropic dispersion correction. This must
        # be done BEFORE adding the restraint to reference_system.
        # We dont care about restraint in decoupled state (although we will set to 0) since we will
        # account for that by hand.

        # Helper function for expanded cutoff
        def expand_cutoff(passed_system, expanded_cutoff_length):
            # Determine minimum box side dimension
            box_vectors = passed_system.getDefaultPeriodicBoxVectors()
            min_box_dimension = min([max(vector) for vector in box_vectors])
            # Expand cutoff to minimize artifact and verify that box is big enough.
            # If we use a barostat we leave more room for volume fluctuations or
            # we risk fatal errors. If we don't use a barostat, OpenMM will raise
            # the appropriate exception on context creation.
            expanded_switching_distance = expanded_cutoff_length - (1 * unit.angstrom)
            if thermodynamic_state.pressure and min_box_dimension < 2.25 * expanded_cutoff_length:
                raise RuntimeError('Barostated box sides must be at least {} Angstroms '
                                   'to correct for missing dispersion interactions'
                                   ''.format(expanded_cutoff_length/unit.angstrom * 2))

            logger.debug('Setting cutoff for fully interacting system to maximum '
                         'allowed {}'.format(str(expanded_cutoff_length)))

            # Expanded cutoff system if needed
            # We don't want to reduce the cutoff if its already large
            for force in passed_system.getForces():
                try:
                    if force.getCutoffDistance() < expanded_cutoff_length:
                        force.setCutoffDistance(expanded_cutoff_length)
                        # Set switch distance
                        # We don't need to check if we are using a switch since there is a setting for that.
                        force.setSwitchingDistance(expanded_switching_distance)
                except Exception:
                    pass
                try:
                    # Check for other types of forces
                    if force.getCutoff() < expanded_cutoff_length:
                        force.setCutoff(expanded_cutoff_length)
                except Exception:
                    pass

        # Set the fully-interacting expanded cutoff state here
        if not is_periodic or not self._anisotropic_dispersion_correction:
            fully_interacting_expanded_state = None
        else:
            # Create the fully interacting system
            fully_interacting_expanded_system = copy.deepcopy(reference_system)

            # Expand Cutoff
            expand_cutoff(fully_interacting_expanded_system, self._anisotropic_dispersion_cutoff)

            # Construct thermodynamic states
            fully_interacting_expanded_state = copy.deepcopy(thermodynamic_state)
            fully_interacting_expanded_state.system = fully_interacting_expanded_system

        # Compute standard state corrections for complex phase.
        metadata['standard_state_correction'] = 0.0
        # TODO: Do we need to include a standard state correction for other phases in periodic boxes?
        if is_complex and restraint_type is not None:
            logger.debug("Creating receptor-ligand restraints...")
            reference_positions = positions[0]
            restraints = create_restraint(restraint_type, alchemical_phase.reference_topology,
                                          thermodynamic_state, reference_system, reference_positions,
                                          atom_indices['receptor'], atom_indices['ligand'])
            force = restraints.get_restraint_force()  # Get Force object incorporating restraints.
            reference_system.addForce(force)
            metadata['standard_state_correction'] = restraints.get_standard_state_correction()  # in kT
        elif is_complex_explicit:
            # For periodic systems, we must still add a standard state
            # correction for the box volume.
            # TODO: What if the box volume fluctuates during the simulation?
            box_vectors = reference_system.getDefaultPeriodicBoxVectors()
            box_volume = thermodynamic_state._volume(box_vectors)
            metadata['standard_state_correction'] = - np.log(V0 / box_volume)
        elif is_complex_implicit:
            # For implicit solvent/vacuum complex systems, we require a restraint
            # to keep the ligand from drifting too far away from receptor.
            raise ValueError('A receptor-ligand system in implicit solvent or '
                             'vacuum requires a restraint.')

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

        # Create the expanded cutoff decoupled state
        if fully_interacting_expanded_state is None:
            noninteracting_expanded_state = None
        else:
            # Create the system for noninteracting
            expanded_factory = AbsoluteAlchemicalFactory(fully_interacting_expanded_state.system,
                                                         ligand_atoms=alchemical_indices,
                                                         **self._alchemy_parameters)
            noninteracting_expanded_system = expanded_factory.alchemically_modified_system
            # Set all USED alchemical interactions to the decoupled state
            alchemical_state = alchemical_states[-1]
            AbsoluteAlchemicalFactory.perturbSystem(noninteracting_expanded_system, alchemical_state)

            # Construct thermodynamic states
            noninteracting_expanded_state = copy.deepcopy(thermodynamic_state)
            noninteracting_expanded_state.system = noninteracting_expanded_system

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
        simulation = ModifiedHamiltonianExchange(store_filename, platform=self._platform)
        simulation.create(thermodynamic_state, alchemical_states, positions,
                          displacement_sigma=self._mc_displacement_sigma, mc_atoms=mc_atoms,
                          options=repex_parameters, metadata=metadata,
                          fully_interacting_expanded_state = fully_interacting_expanded_state,
                          noninteracting_expanded_state = noninteracting_expanded_state)

        # Initialize simulation.
        # TODO: Use the right scheme for initializing the simulation without running.
        #logger.debug("Initializing simulation...")
        #simulation.run(0)

        # Clean up simulation.
        del simulation

        # Add to list of phases that have been set up.
        self._phases.append(alchemical_phase.name)
        self._phases.sort()  # sorting ensures all MPI processes run the same phase

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
            simulation = ModifiedHamiltonianExchange(store_filename=store_filename, mpicomm=self._mpicomm,
                                                     platform=self._platform)
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
            simulation = ModifiedHamiltonianExchange(fullpath, platform=self._platform)
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
