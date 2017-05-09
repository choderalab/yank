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
import abc
import copy
import time
import logging
import importlib
import collections

import mdtraj
import pandas
import numpy as np
import openmmtools as mmtools
from simtk import unit, openmm

from . import utils, pipeline, repex, mpi
from .restraints import RestraintState, RestraintParameterError, V0

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

class IMultiStateSampler(mmtools.utils.SubhookedABCMeta):
    """A sampler for multiple thermodynamic states.

    This is the interface documents the properties and methods that
    need to be exposed by the sampler object to be compatible with
    the class `AlchemicalPhase`.

    """

    @property
    def number_of_iterations(self):
        """int: the total number of iterations to run."""
        pass

    @abc.abstractproperty
    def iteration(self):
        """int: the current iteration."""
        pass

    @abc.abstractproperty
    def metadata(self):
        """dict: a copy of the metadata dictionary passed on creation."""
        pass

    @abc.abstractproperty
    def sampler_states(self):
        """list of SamplerState: the sampler states at the current iteration."""
        pass

    @abc.abstractmethod
    def create(self, thermodynamic_state, sampler_states, storage,
               unsampled_thermodynamic_states, metadata):
        """Create new simulation and initialize the storage.

        Parameters
        ----------
        thermodynamic_state : list of openmmtools.states.ThermodynamicState
            The thermodynamic states for the simulation.
        sampler_states : openmmtools.states.SamplerState or list
            One or more sets of initial sampler states. If a list of SamplerStates,
            they will be assigned to thermodynamic states in a round-robin fashion.
        storage : str or Reporter
            If str: The path to the storage file. Reads defaults from the Reporter class
            If Reporter: Reads the reporter settings for files and options
            In the future this will be able to take a Storage class as well.
        unsampled_thermodynamic_states : list of openmmtools.states.ThermodynamicState
            These are ThermodynamicStates that are not propagated, but their
            reduced potential is computed at each iteration for each replica.
            These energy can be used as data for reweighting schemes.
        metadata : dict
           Simulation metadata to be stored in the file.

        """
        pass

    @abc.abstractmethod
    def minimize(self, tolerance, max_iterations):
        """Minimize all states.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity
            Minimization tolerance (units of energy/mole/length, default is
            1.0 * unit.kilojoules_per_mole / unit.nanometers).
        max_iterations : int
            Maximum number of iterations for minimization. If 0, minimization
            continues until converged.

        """
        pass

    @abc.abstractmethod
    def equilibrate(self, n_iterations, mcmc_moves=None):
        """Equilibrate all states.

        Parameters
        ----------
        n_iterations : int
            Number of equilibration iterations.
        mcmc_moves : MCMCMove or list of MCMCMove, optional
            Optionally, the MCMCMoves to use for equilibration can be
            different from the ones used in production (default is None).

        """
        pass

    @abc.abstractmethod
    def run(self, n_iterations=None):
        """Run the simulation.

        This runs at most `number_of_iterations` iterations. Use `extend()`
        to pass the limit.

        Parameters
        ----------
        n_iterations : int, optional
           If specified, only at most the specified number of iterations
           will be run (default is None).

        """
        pass

    @abc.abstractmethod
    def extend(self, n_iterations):
        """Extend the simulation by the given number of iterations.

        Contrarily to `run()`, this will extend the number of iterations past
        `number_of_iteration` if requested.

        Parameters
        ----------
        n_iterations : int
           The number of iterations to run.

        """
        pass


class AlchemicalPhase(object):
    """A single thermodynamic leg (phase) of an alchemical free energy calculation.

    This class wraps around a general MultiStateSampler and handle the creation
    of an alchemical free energy calculation.

    Parameters
    ----------
    sampler : MultiStateSampler
        The sampler instance implementing the IMultiStateSampler interface.

    """
    def __init__(self, sampler):
        self._sampler = sampler

    @staticmethod
    def from_storage(storage):
        """Static constructor from an existing storage file.

        Parameters
        ----------
        storage : str or Reporter
            If str: The path to the primary storage file. Default checkpointing options are stored in this case
            If Reporter: loads from the reporter class, including checkpointing information
            In the future this will be able to take a Storage class as well.

        Returns
        -------
        alchemical_phase : AlchemicalPhase
            A new instance of AlchemicalPhase in the same state of the
            last stored iteration.

        """
        # Check if netcdf file exists.
        if type(storage) is str:
            reporter = repex.Reporter(storage)
        else:
            reporter = storage
        if not reporter.storage_exists():
            reporter.close()
            raise RuntimeError('Storage file at {} does not exists; cannot resume.'.format(reporter.filename))

        # TODO: this should skip the Reporter and use the Storage to read storage.metadata.
        # Open Reporter for reading and read metadata.
        reporter.open(mode='r')
        metadata = reporter.read_dict('metadata')
        reporter.close()

        # Retrieve the sampler class.
        sampler_full_name = metadata['sampler_full_name']
        module_name, cls_name = sampler_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)

        # Resume sampler and return new AlchemicalPhase.
        sampler = cls.from_storage(reporter)
        return AlchemicalPhase(sampler)

    @property
    def iteration(self):
        """int: the current iteration (read-only)."""
        return self._sampler.iteration

    @property
    def number_of_iterations(self):
        """int: the total number of iterations to run."""
        return self._sampler.number_of_iterations

    @number_of_iterations.setter
    def number_of_iterations(self, value):
        self._sampler.number_of_iterations = value

    def create(self, thermodynamic_state, sampler_states, topography, protocol,
               storage, restraint=None, anisotropic_dispersion_cutoff=None,
               alchemical_regions=None, alchemical_factory=None, metadata=None):
        """Create a new AlchemicalPhase calculation for a specified protocol.

        If `anisotropic_dispersion_cutoff` is different than `None`. The
        end states of the phase will be reweighted. The fully interacting
        state accounts for:
            1. The truncation of nonbonded interactions.
            2. The reciprocal space which is not modeled in alchemical
               states if an Ewald method is used for long-range interactions.
            3. If the system is restrained and the first step of the protocol
               has `lambda_restraints=1.0`, the reweighting also accounts
               for the free energy of applying the restraint.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            Thermodynamic state holding the reference system, temperature
            and pressure.
        sampler_states : openmmtools.states.SamplerState or list
            One or more sets of initial sampler states. If a list of SamplerStates,
            they will be assigned to replicas in a round-robin fashion.
        topography : Topography
            The object holding the topology and labelling the different
            components of the system. This is used to discriminate between
            ligand-receptor and solvation systems.
        protocol : dict
            The dictionary parameter_name: list_of_parameter_values defining
            the protocol. All the parameter values list must have the same
            number of elements.
        storage : str or initialized Reporter class
            If str: Path to the storage file. The default checkpointing options (see the Reporter class of Repex)
                will be used in this case
            If Reporter: Uses files and checkpointing options of the reporter class passed in
        restraint : ReceptorLigandRestraint, optional
            Restraint to add between protein and ligand. This must be specified
            for ligand-receptor systems in non-periodic boxes.
        anisotropic_dispersion_cutoff : simtk.openmm.Quantity, optional
            If specified, this is the cutoff at which to reweight long range
            interactions of the end states to correct for anisotropic dispersions.
            If `None`, the correction won't be applied (units of length, default
            is None).
        alchemical_regions : openmmtools.alchemy.AlchemicalRegion, optional
            If specified, this is the AlchemicalRegion that will be passed
            to the AlchemicalFactory, otherwise the ligand will be alchemically
            modified according to the given protocol.
        alchemical_factory : openmmtools.alchemy.AlchemicalFactory, optional
            If specified, this AlchemicalFactory will be used instead of
            the one created with default options.
        metadata : dict, optional
            Simulation metadata to be stored in the file.

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
            reference_state_expanded = self._expand_state_cutoff(reference_thermodynamic_state,
                                                                 anisotropic_dispersion_cutoff)

            # Add the restraint if any. The free energy of removing the restraint
            # will be taken into account with the standard state correction.
            if restraint is not None:
                restraint.restrain_state(reference_state_expanded)
                # The value of lambda_restraints must be the same as the first state.
                # TODO: handle case with multiple restraints.
                restraint_state.lambda_restraints = compound_states[0].lambda_restraints
                reference_state_expanded = mmtools.states.CompoundThermodynamicState(
                    thermodynamic_state=reference_state_expanded, composable_states=[restraint_state])

            # Create the expanded cutoff decoupled state.
            last_state_expanded = self._expand_state_cutoff(compound_states[-1],
                                                            anisotropic_dispersion_cutoff)

            expanded_cutoff_states = [reference_state_expanded, last_state_expanded]
        elif anisotropic_dispersion_cutoff is not None:
            logger.warning("The requested anisotropic dispersion correction "
                           "won't be computed since the system is non-periodic.")

        # Create simulation.
        # ------------------

        logger.debug("Creating sampler object...")
        self._sampler.create(compound_states, sampler_states,
                             storage=storage, unsampled_thermodynamic_states=expanded_cutoff_states, metadata=metadata)

    def minimize(self, tolerance=1.0*unit.kilojoules_per_mole/unit.nanometers,
                 max_iterations=0):
        """Minimize all the states.

        The minimization is performed in two steps. In the first one, the
        positions are minimized in the reference thermodynamic state (i.e.
        non alchemically-modified). Only then, the positions are minimized
        in their alchemically softened state.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity, optional
            Minimization tolerance (units of energy/mole/length, default is
            1.0 * unit.kilojoules_per_mole / unit.nanometers).
        max_iterations : int, optional
            Maximum number of iterations for minimization. If 0, minimization
            continues until converged.

        """
        metadata = self._sampler.metadata
        serialized_reference_state = metadata['reference_state']
        reference_state = mmtools.utils.deserialize(serialized_reference_state)
        sampler_states = self._sampler.sampler_states

        # We minimize only the sampler states that are in different positions.
        # This depends on how many sampler states have been passed in create()
        # and if the ligand has been randomized before calling minimize().
        similar_sampler_states = self._find_similar_sampler_states(sampler_states)

        logger.debug('Minimizing {} sampler states in the reference '
                     'thermodynamic state'.format(len(similar_sampler_states)))

        # Distribute minimization across nodes.
        minimized_sampler_states_ids = list(similar_sampler_states.keys())
        minimized_positions = mpi.distribute(self._minimize_sampler_state, minimized_sampler_states_ids,
                                             sampler_states, reference_state, tolerance, max_iterations,
                                             send_results_to='all')

        # Update all sampler states.
        for sampler_state_id, minimized_pos in zip(minimized_sampler_states_ids, minimized_positions):
            sampler_states[sampler_state_id].positions = minimized_pos
            for similar_sampler_state_id in similar_sampler_states[sampler_state_id]:
                sampler_states[similar_sampler_state_id].positions = minimized_pos

        # Update sampler and perform second minimization in alchemically modified states.
        self._sampler.sampler_states = sampler_states
        self._sampler.minimize(tolerance=tolerance, max_iterations=max_iterations)

    def randomize_ligand(self, sigma_multiplier=2.0, close_cutoff=1.5*unit.angstrom):
        """Randomize the ligand positions in every state.

        The position and orientation of the ligand in each state will
        be randomized. This works only if the system is a ligand-receptor
        system.

        If you call this before minimizing, each positions will be minimized
        separately in the reference state, so you may want to call it
        afterwards to speed up minimization.

        Parameters
        ----------
        sigma_multiplier : float, optional
            The ligand will be placed close to a random receptor atom at
            a distance that is normally distributed with standard deviation
            sigma_multiplier * receptor_radius_of_gyration (default is 2.0).
        close_cutoff : simtk.unit.Quantity, optional
            Each random placement proposal will be rejected if the ligand
            ends up being closer to the receptor than this cutoff (units of
            length, default is 1.5*unit.angstrom).

        """
        metadata = self._sampler.metadata
        serialized_topography = metadata['topography']
        topography = mmtools.utils.deserialize(serialized_topography)

        # We can randomize the ligand only in implicit solvent.
        is_complex = len(topography.ligand_atoms) > 0
        is_explicit = len(topography.solvent_atoms) > 0
        if not is_complex:
            raise RuntimeError('Cannot find ligand atoms to randomize.')
        if is_explicit:
            raise RuntimeError('Cannot randomize ligand in explict solvent.')

        # Randomize all sampler states.
        sampler_states = self._sampler.sampler_states
        ligand_positions = mpi.distribute(self._randomize_ligand, sampler_states, topography,
                                          sigma_multiplier, close_cutoff, send_results_to='all')

        # Update sampler states with randomized positions.
        for sampler_state, ligand_pos in zip(sampler_states, ligand_positions):
            sampler_state.positions[topography.ligand_atoms] = ligand_pos

        self._sampler.sampler_states = sampler_states

    def equilibrate(self, n_iterations, mcmc_moves=None):
        """Equilibrate all states.

        Parameters
        ----------
        n_iterations : int
            Number of equilibration iterations.
        mcmc_moves : MCMCMove or list of MCMCMove, optional
            Optionally, the MCMCMoves to use for equilibration can be
            different from the ones used in production.

        """
        self._sampler.equilibrate(n_iterations=n_iterations, mcmc_moves=mcmc_moves)

    def run(self, n_iterations=None):
        """Run the alchemical phase simulation.

        Parameters
        ----------
        n_iterations : int, optional
           If specified, only at most the specified number of iterations
           will be run (default is None).

        """
        self._sampler.run(n_iterations=n_iterations)

    def extend(self, n_iterations):
        """Extend the simulation by the given number of iterations.

        Parameters
        ----------
        n_iterations : int
           The number of iterations to run.

        """
        self._sampler.extend(n_iterations)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    @staticmethod
    def _expand_state_cutoff(thermodynamic_state, expanded_cutoff_distance):
        """Expand the thermodynamic state cutoff to the given one."""
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
        """Create a default AlchemicalRegion if the user hasn't provided one."""
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

    @staticmethod
    def _find_similar_sampler_states(sampler_states):
        """Groups SamplerStates that have the same positions.

        Returns
        -------
        similar_sampler_states : dict
            The dict sampler_state_index: list_of_sampler_state_indices
            with same positions.

        """
        # similar_sampler_states is an ordered dict
        #       sampler_state_index: list of sampler_state_indices with same positions
        # we run only 1 minimization for each of these entries.
        similar_sampler_states = collections.OrderedDict()

        # processed_sampler_states_ids is a set containing all the
        # sampler state indices that have been assigned a minimization.
        processed_sampler_states_ids = set()

        # Find minimum number of minimizations required.
        for state_id, sampler_state in enumerate(sampler_states):
            if state_id in processed_sampler_states_ids:
                continue
            similar_sampler_states[state_id] = []
            processed_sampler_states_ids.add(state_id)
            for next_state_id in range(state_id+1, len(sampler_states)):
                next_sampler_state = sampler_states[next_state_id]
                if np.allclose(sampler_state.positions, next_sampler_state.positions):
                    similar_sampler_states[state_id].append(next_state_id)
                    processed_sampler_states_ids.add(next_state_id)

        return similar_sampler_states

    @staticmethod
    def _minimize_sampler_state(sampler_state_id, sampler_states, thermodynamic_state,
                                tolerance, max_iterations):
        """Minimize the specified sampler state at the given thermodynamic state."""
        sampler_state = sampler_states[sampler_state_id]

        # Retrieve a context. Any Integrator works.
        context, integrator = mmtools.cache.global_context_cache.get_context(thermodynamic_state)

        # Set initial positions and box vectors.
        sampler_state.apply_to_context(context)

        # Compute the initial energy of the system for logging.
        initial_energy = thermodynamic_state.reduced_potential(context)
        logger.debug('Sampler state {}/{}: initial energy {:8.3f}kT'.format(
            sampler_state_id + 1, len(sampler_states), initial_energy))

        # Minimize energy.
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)

        # Get the minimized positions.
        sampler_state.update_from_context(context)

        # Compute the final energy of the system for logging.
        final_energy = thermodynamic_state.reduced_potential(sampler_state)
        logger.debug('Sampler state {}/{}: final energy {:8.3f}kT'.format(
            sampler_state_id + 1, len(sampler_states), final_energy))

        # Return minimized positions.
        return sampler_state.positions

    @staticmethod
    def _randomize_ligand(sampler_state, topography, sigma_multiplier, close_cutoff):
        """Randomize ligand positions of the given sampler state."""
        # Shortcut variables.
        ligand_atoms = topography.ligand_atoms
        receptor_atoms = topography.receptor_atoms

        # We set the standard deviation of the displacement
        # proportional to the receptor radius of gyration.
        radius_of_gyration = pipeline.compute_radius_of_gyration(sampler_state.positions[receptor_atoms])
        sigma = sigma_multiplier * radius_of_gyration

        # Convert to dimensionless positions.
        positions_unit = sampler_state.positions.unit
        x = sampler_state.positions / positions_unit
        close_cutoff = close_cutoff / positions_unit

        # We work with Quantity only for ligand atoms for readability.
        ligand_positions = x[ligand_atoms] * positions_unit

        # Try until we have a non-overlapping ligand conformation.
        max_n_attempts = 5000
        n_attempts = 0
        while n_attempts <= max_n_attempts:
            # Center ligand on a random receptor atom.
            ligand_positions_mean = ligand_positions.mean(0)
            receptor_atom_index = receptor_atoms[np.random.randint(0, len(receptor_atoms))]
            ligand_positions[:] += sampler_state.positions[receptor_atom_index] - ligand_positions_mean

            # Randomize ligand orientation and displace.
            ligand_positions = mmtools.mcmc.MCRotationMove.rotate_positions(ligand_positions)
            ligand_positions = mmtools.mcmc.MCDisplacementMove.displace_positions(ligand_positions, sigma)

            # Update array to compute distances.
            x[ligand_atoms, :] = ligand_positions / positions_unit

            # Check if there's overlap.
            min_dist = pipeline.compute_min_dist(x[ligand_atoms], x[receptor_atoms])
            if min_dist >= close_cutoff:
                break
            n_attempts += 1

        # Check if we could find a working configuration.
        if n_attempts > max_n_attempts:
            raise RuntimeError('Could not randomize ligand after {} attempts'.format(max_n_attempts))

        # We return only the randomized ligand positions to minimize MPI traffic.
        return ligand_positions
