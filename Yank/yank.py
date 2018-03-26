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

import abc
import collections
import copy
import functools
import importlib
import logging
import time
from typing import Union, Tuple, List, Set

import mdtraj
import numpy as np
import openmmtools as mmtools
import pandas
from simtk import unit, openmm

from . import pipeline, mpi, multistate
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

    # Built in class attributes
    _BUILT_IN_REGIONS = ('ligand_atoms', 'receptor_atoms', 'solute_atoms', 'solvent_atoms', 'ion_atoms')
    _PROTECTED_REGION_NAMES = ('and', 'or')

    def __init__(self, topology, ligand_atoms=None, solvent_atoms='auto'):
        # Determine if we need to convert the topology to mdtraj.
        if isinstance(topology, mdtraj.Topology):
            self._topology = topology
        else:
            self._topology = mdtraj.Topology.from_openmm(topology)

        # Initialize regions, this has to come before solvent/ligand atoms to ensure
        self._regions = {}

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
        """The atom indices of the ligand as list

        This can be empty if this :class:`Topography` doesn't represent a receptor-ligand
        system. Use solute_atoms to obtain the atom indices of the molecule if
        this is the case.

        If assigned to a string, it will be interpreted as an mdtraj DSL specification
        of the atom indices.

        """
        return self._ligand_atoms

    @ligand_atoms.setter
    def ligand_atoms(self, value):
        self._ligand_atoms = self.select(value)

        # Safety check: with a ligand there should always be a receptor.
        if len(self._ligand_atoms) > 0 and len(self.receptor_atoms) == 0:
            raise ValueError('Specified ligand but cannot find '
                             'receptor atoms. Ligand: {}'.format(value))

    @property
    def receptor_atoms(self):
        """The atom indices of the receptor as list (read-only).

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
            self._solvent_atoms = self.select(value)

    @property
    def ions_atoms(self):
        """The indices of all ions atoms in the solvent (read-only)."""
        # Ions are all atoms of the solvent whose residue name show a charge.
        return [i for i in self._solvent_atoms
                if '-' in self._topology.atom(i).residue.name or
                '+' in self._topology.atom(i).residue.name]

    def add_region(self, region_name, region_selection, subset=None):
        """
        Add a region to the Topography based on a selection string of atoms. The selection accepts multiple formats
        such as a DSL string, a SMIRKS selection string, or hard coded atom indices. The selection string is converted
        to a list of atom indices.

        Parameters
        ----------
        region_name : str
            Name of the region. This must be unique and also not the name of an existing method
        region_selection : str or list of ints
            Atom selection identifier, either a MDTraj DSL string, a SMARTS string, a compound region selection,
            or a hard-coded list of ints.
            The SMARTS string requires the OpenEye OEChem library to correctly select
        subset : str or list of ints, or None
            Atom selection sub-region to filter the atom selection through. This is a way to define your new region
            as a relative selection to the subset. Follows the same conditions as ``region_selection``.

        """
        self._check_existing_regions(region_name)
        self._check_reserved_words(region_name)
        atom_selection = self.select(region_selection, subset=subset)
        self._regions[region_name] = atom_selection

    def remove_region(self, region_name):
        """
        Remove a previously added region from this Topography. This only affects regions added through the
        :func:`add_region` function.

        Does nothing if the region was not previously added

        Parameters
        ----------
        region_name : str
            Name of the region to remove

        """
        self._regions.pop(region_name, None)

    def get_region(self, region_name) -> List[int]:
        """
        Retrieve the atom indices of the given region. This function will also fetch the built-in regions

        Parameters
        ----------
        region_name : str
            Name of the region to fetch. Can use both custom regions or built in ones such as "ligand_atoms"

        Returns
        -------
        region : list of ints
            Atom integers which comprise the region

        Raises
        ------
        KeyError
            If region is not part of the Topography
        """
        if region_name not in self:
            raise KeyError("Cannot find region \"{}\" in this Topography.".format(region_name))
        # Return the built-in if present
        if region_name in self._BUILT_IN_REGIONS:
            return getattr(self, region_name)
        # Return a copy to ensure people cant tweak the region outside of the api
        return copy.copy(self._regions[region_name])

    def select(self, selection, as_set=False, sort_by='auto', subset=None) -> Union[List[int], Set[int]]:
        """
        Select atoms based on selection format which can be a number of formats:

        * Single integer (a bit redundant)
        * Iterable of integer (also a bit redundant)
        * Complex Region selection
        * MDTraj String
        * SMARTS Selection

        This method will never return duplicate atom numbers.

        The ``sort_by`` method controls how the output is sorted before being returned, see the details in the
        Parameters block for information about each option

        The Complex Region string returns an the set of atoms derived from the arguments using logical operators
        ``and`` and ``or``
        along with grouping through parenthesis . For example, assume you
        have two regions ``regionA = [0,1,2,3]`` and ``regionB = [2,3,4,5]``. You can do operations such as the
        following:

             ``regionA and regionB`` yields ``[2,3]``, which is the intersection of the regions.

             ``regionA or regionB`` yields ``[0,1,2,3,4,5]``, which is the union of the regions.

        More complex statements with more regions will also work, and statements can be grouped with ``()``.

        The ``subset`` keyword filters the ``selection`` relative to this subset selection. If not None, the subset is
        processed first through this same function, then the primary selection is processed relative to it.
        ``subset`` follows the same conditions as ``selection``, but sort order for subset is ignored
        If your ``selection`` would pick atoms that are NOT part of the subset, then those atoms are NOT RETURNED.
        If your ``selection`` is an integer or some sequence of integer, then the indices are relative to the
        ``subset``.
        Final atom numbers will be absolute to the whole Topography.

        Parameters
        ----------
        selection : str, list of ints, or int
            String defining the selection
        as_set : bool, Default False
            Determines output format. Returns a Set if True, otherwise output is a list.
        sort_by : str or None, Default: 'auto'
            Determine how to sort the output if ``as_set`` is False.

            * 'auto': Let the selection string determine how to sort it out based on its priorities
            * 'index': Atoms are sorted index, smallest to largest
            * 'region_order': Atoms are sorted by which region in the provided ``selection_string`` occurs first.
                So if your expression is ``region1 and region2``, then the output will be atoms which appear in
                ``region1`` first. Parenthesis are **ignored** so the expression ``region1 and (region2 or region3)``
                will prioritize ``region1`` first for sorting. This option only works for expressions on Regions,
                other selections will fall back to ``None``
            * ``None``: Sorting is left up to however the selection string is processed by their respective drivers,
                or by whatever order the set operations returns it. Not guaranteed to be deterministic in all cases.

        subset : None, str, list of ints, or int; Optional; Default: None
            Set of atoms to make a relative selection to. Follows the same rules as ``selection`` for valid inputs.
            If None, ``selection`` is on whole Topography.


        Returns
        -------
        selection : list or set
            Returns the selected atoms as either a list or set based on ``as_set`` keyword.
            Order of the output is determined by the ``sort_by`` keyword. If a ``as_set`` is ``True``, this option has
            no effect.

        """

        # Handle subset. Define a subset topology to manipulate, then define a common call to convert subset atom
        # into absolute atom
        if subset is not None:
            subset_atoms = self.select(subset, sort_by='index', as_set=False, subset=None)
            topology = self.topology.subset(subset_atoms)
        else:
            subset_atoms = None
            topology = self.topology

        class AtomMap(object):
            """Atom mapper class"""
            def __init__(self, subset_atoms):
                self.subset_atoms = subset_atoms

            def atom_mapping(self, atom):
                """Use a "given x, return x" mapping instead of list(range(n_atoms)) or something memory intensive"""
                if self.subset_atoms is None:
                    return atom
                else:
                    # Return the mapped atom, only if the atom is actually part of the atom map
                    try:
                        return_atom = self.subset_atoms[atom]
                    except IndexError:
                        return_atom = None
                    return return_atom

            def __contains__(self, item):
                if self.subset_atoms is None:
                    return 0 <= item < topology.n_atoms
                else:
                    return item in self.subset_atoms

        atom_map = AtomMap(subset_atoms)
        # Shorthand for later
        atom_mapping = atom_map.atom_mapping

        # Helper functions for handling the sorting, atoms should be in absolute terms at this point
        def sort_output_index(sortable):
            # Dont do list.sort, its an in place action.
            return sorted(list(sortable))

        def sort_output_region_order(sortable):
            # Only valid when selection is a string
            final_output = []
            # Determine which regions are in the list
            region_order = [region_name for region_name in selection.split() if region_name in self]
            # Cycle through regions
            for region_name in region_order:
                region = self.get_region(region_name)
                # Cycle through atom in region
                for atom_number in region:
                    # Ensure atom is part of selection output and not previously added
                    # Because only "and" and "or" arguments are allowed, every value in the sortable input is ensured
                    # to be in the regions
                    if atom_number in sortable and atom_number not in final_output:
                        final_output.append(atom_number)
            return final_output

        def sort_output_none(sortable):
            return list(sortable)

        sortable_dispatch = {'index': sort_output_index,
                             'region_order': sort_output_region_order,
                             None: sort_output_none}

        class Selector(abc.ABC):
            # Implement this to get the valid sort priority. If the sortable is valid, don't include it
            SORT_PRIORITY = (None,)

            def __init__(self):
                """This class and its subclasses are not meant to be used as an instance."""
                pass

            @classmethod
            @abc.abstractmethod
            def select(cls, selection_input) -> Union[Tuple[List[int], None], Tuple[None, Exception]]:
                """Implement this to convert select_string to the output, returning both the output, and the error"""
                return [0], None

            @classmethod
            def sort_selection(cls, selected_atoms) -> Union[List[int], Set[int]]:
                if as_set:
                    return set(selected_atoms)
                elif sort_by in cls.SORT_PRIORITY:
                    return sortable_dispatch[sort_by](selected_atoms)
                else:
                    return sortable_dispatch[cls.SORT_PRIORITY[0]](selected_atoms)

        # Helper functions for unifying string selection processing
        class SelectRegion(Selector):
            SORT_PRIORITY = ('index', 'region_order', None)

            @classmethod
            def select(cls, region_string):
                try:
                    # The self here is inherited from the outer scope
                    region_output_unmapped = list(self._get_region_set(region_string))
                    region_output = [item for item in region_output_unmapped if item in atom_map]
                except (SyntaxError, ValueError) as e:
                    # Make this a local variable
                    region_error = e
                    region_output = None
                else:
                    region_error = None
                return region_output, region_error

        class SelectDsl(Selector):
            SORT_PRIORITY = ('index', None)

            @classmethod
            def select(cls, dsl_string):
                try:
                    mdtraj_output_unmapped_output = (topology.select(dsl_string)).tolist()
                    mdtraj_output = [item for item in map(atom_mapping, mdtraj_output_unmapped_output)
                                     if item is not None]
                except ValueError as e:
                    # Make this a local variable
                    mdtraj_error = e
                    mdtraj_output = None
                else:
                    mdtraj_error = None
                return mdtraj_output, mdtraj_error

        class SelectSmarts(Selector):

            @classmethod
            def select(cls, smarts_string):
                try:
                    # Skeletal structure, not used yet
                    raise NotImplementedError("This method has not been implemented yet")
                except (NotImplementedError, ValueError) as e:
                    smarts_error = e
                    smarts_output = None
                return smarts_output, smarts_error

        class SelectIterable(Selector):
            # Do not allow these to be sorted as they have been manually specified by integer

            @classmethod
            def select(cls, iterable):
                try:
                    iterable_output = iterable.tolist()
                except AttributeError:
                    iterable_output = list(iterable)
                    iterable_error = None
                except Exception as e:
                    iterable_error = e
                    iterable_output = None
                else:
                    iterable_error = None
                if iterable_output is not None:
                    iterable_map = [item for item in map(atom_mapping, iterable_output) if item is not None]
                    iterable_output = iterable_map
                return iterable_output, iterable_error

        class SelectInt(SelectIterable):
            # Verbatim copy of Iterable, but change name to help error processing
            pass

        # Dispatcher to parse the selection type and return the valid selection classes
        @functools.singledispatch
        def selector_picker(selection_input) -> Tuple[Tuple[Selector, ...], str]:
            if not all([isinstance(i, np.integer) or isinstance(i, int) for i in selection_input]):
                raise ValueError("Selection {} is not iterable of ints or any other readable type such as string!"
                                 "Unable to parse!".format(selection))
            return (SelectIterable,), 'iterable'

        @selector_picker.register(int)
        def int_selector(_) -> Tuple[Tuple[Selector, ...], str]:
            return (SelectInt,), 'integer'

        @selector_picker.register(str)
        def string_selection(_) -> Tuple[Tuple[Selector, ...], str]:
            return (SelectRegion, SelectDsl, SelectSmarts), "string"

        registered_selectors, region_selector_types = selector_picker(selection)

        selector_outputs = []
        selector_errors = []
        selector_names = []  # For handling error messages
        for selector in registered_selectors:
            selector_names.append(selector.__name__)
            selection_output, errors = selector.select(selection)
            selector_outputs.append(selection_output)
            selector_errors.append(errors)
        # Show only the valid selectors
        valid_selectors = [index for index, output in enumerate(selector_outputs) if output is not None]
        if len(valid_selectors) > 1:
            # Choose a baseline selector
            base_selection = selector_outputs[valid_selectors[0]]
            for index in valid_selectors[1:]:
                comparison_selection = selector_outputs[valid_selectors[index]]
                if base_selection != comparison_selection:
                    raise ValueError("The selection {} was ambiguous as the following selectors returned valid "
                                     "outputs, but they were different! Consider refining your selection string or "
                                     "changing region names to not align with other selection string: \n"
                                     "    {}".format(selection,
                                                     [selector_names[index] for index in valid_selectors]))
            # If we made it here, it does means the selectors are the same, does not mater which we pull from, so
            # we'll draw from the 0th index at the end
        elif len(valid_selectors) == 0:
            base_error_string = "The selection {} could not be parsed by any " \
                                "selector in the {} class!".format(selection, region_selector_types)
            base_error_string += ("\nThe following errors were thrown by the selectors which may help you determine "
                                  "why the selection was not parsed:")
            for selector_name, selector_error in zip(selector_names, selector_errors):
                base_error_string += "\n    {}: {}".format(selector_name, selector_error)
            raise ValueError(base_error_string)

        return registered_selectors[valid_selectors[0]].sort_selection(selector_outputs[valid_selectors[0]])

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
                    solvent_atoms=self._solvent_atoms,
                    regions=self._regions)

    def __setstate__(self, serialization):
        topology_dict = serialization['topology']
        atoms = pandas.read_json(topology_dict['atoms'], orient='records')
        bonds = np.array(topology_dict['bonds'])
        self._topology = mdtraj.Topology.from_dataframe(atoms, bonds)
        self._ligand_atoms = serialization['ligand_atoms']
        self._solvent_atoms = serialization['solvent_atoms']
        self._regions = serialization['regions']

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _check_existing_regions(self, region_string):
        """Make sure regions don't overlap"""
        if region_string in self:
            raise KeyError("{} is already part of this Topology! "
                           "Cannot overwrite built-in regions!".format(region_string))

    def _check_reserved_words(self, region_string):
        """Make sure region is NOT a protected name"""
        if region_string in self._PROTECTED_REGION_NAMES:
            raise KeyError("{} is a protected keyword for logical operations and "
                           "cannot be used as a region name".format(region_string))

    def __contains__(self, item):
        """Check the in operator to see if region is in this class"""
        return item in self._regions or item in self._BUILT_IN_REGIONS

    def _get_region_set(self, region_set_string):
        """
        Get a new region as a logical combination of several region sets.

        See docs in :func:`select` for details about the logic in Complex

        Parameters
        ----------
        region_set_string : str
            Region combination string using region names, logical operators, and parenthesis grouping.

        Returns
        -------
        combined_region : set
            Set of combined regions

        """
        # Combine regions, start with keys only since we are converting to a set
        combined_region_keys = tuple(self._regions.keys()) + self._BUILT_IN_REGIONS
        # Cast regions to set, but only if they are in the region_set_string
        variables = {key: set(self.get_region(key)) for key in combined_region_keys if key in region_set_string}
        parsed_output = mmtools.utils.math_eval(region_set_string, variables=variables)
        return parsed_output


# ==============================================================================
# Class that define a single thermodynamic leg (phase) of the calculation
# ==============================================================================

class IMultiStateSampler(mmtools.utils.SubhookedABCMeta):
    """A sampler for multiple thermodynamic states.

    This is the interface documents the properties and methods that
    need to be exposed by the sampler object to be compatible with
    the class :class:`AlchemicalPhase`.

    Attributes
    ----------
    number_of_iterations
    iteration
    metadata
    sampler_states

    """

    @classmethod
    @abc.abstractmethod
    def read_status(cls, storage):
        """Read the status of the calculation from the storage file.

        Parameters
        ----------
        storage : str or Reporter
            The path to the storage file or the reporter object to forward
            to the sampler. In the future, this will be able to take a Storage
            class as well.

        Returns
        -------
        status : namedtuple
            The status of the calculation.

        """
        pass

    @property
    @abc.abstractmethod
    def number_of_iterations(self):
        """int: the total number of iterations to run."""
        pass

    @property
    @abc.abstractmethod
    def iteration(self):
        """int: the current iteration."""
        pass

    @property
    @abc.abstractmethod
    def metadata(self):
        """dict: a copy of the metadata dictionary passed on creation."""
        pass

    @property
    @abc.abstractmethod
    def sampler_states(self):
        """list of SamplerState: the sampler states at the current iteration."""
        pass

    @abc.abstractmethod
    def create(self, thermodynamic_state, sampler_states, storage,
               unsampled_thermodynamic_states=None,
               initial_thermodynamic_states=None,
               metadata=None):
        """Create new simulation and initialize the storage.

        Parameters
        ----------
        thermodynamic_state : list of openmmtools.states.ThermodynamicState
            The thermodynamic states for the simulation.
        sampler_states : openmmtools.states.SamplerState or list
            One or more sets of initial sampler states. If a list of SamplerStates,
            they will be assigned to thermodynamic states in a round-robin fashion.
        storage : str or Reporter
            The path to the storage file or a Reporter object to forward
            to the sampler. In the future, this will be able to take a
            Storage class as well.
        unsampled_thermodynamic_states : list of openmmtools.states.ThermodynamicState, Optional, Default: None
            These are ThermodynamicStates that are not propagated, but their
            reduced potential is computed at each iteration for each replica.
            These energy can be used as data for reweighting schemes.
        initial_thermodynamic_states : None or list or array-like of int of length len(sampler_states), optional,
            default: None.
            Initial thermodynamic_state index for each sampler_state.
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
            ``1.0 * unit.kilojoules_per_mole / unit.nanometers``).
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

        This runs at most :attr:`number_of_iterations` iterations. Use :func:`extend`
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

        Contrarily to :func:`run`, this will extend the number of iterations past
        :attr:`number_of_iteration` if requested.

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
    sampler : IMultiStateSampler
        The sampler instance implementing the :class:`IMultiStateSampler` interface.

    Attributes
    ----------
    iteration
    number_of_iterations
    is_completed

    """
    def __init__(self, sampler):
        self._sampler = sampler

    @classmethod
    def from_storage(cls, storage):
        """Static constructor from an existing storage file.

        Parameters
        ----------
        storage : str or Reporter
            The path to the storage file or the reporter object to forward
            to the sampler. In the future, this will be able to take a Storage
            class as well.

        Returns
        -------
        alchemical_phase : AlchemicalPhase
            A new instance of :class:`AlchemicalPhase` in the same state of the
            last stored iteration.

        """
        # Read the MultiStateSampler class from the storage.
        sampler_class = cls._read_sampler_class(storage)
        # Resume sampler and return new AlchemicalPhase.
        sampler = sampler_class.from_storage(storage)
        return AlchemicalPhase(sampler)

    @classmethod
    def read_status(cls, storage):
        """Read the status of the calculation from the storage file.

        This method can be used to quickly check the status of the
        simulation before loading the full ``ReplicaExchange`` object
        from disk.

        Parameters
        ----------
        storage : str or Reporter
            The path to the storage file or the reporter object to forward
            to the sampler. In the future, this will be able to take a Storage
            class as well.

        Returns
        -------
        status : namedtuple
            The status of the calculation.

        """
        # Read the MultiStateSampler class from the storage.
        sampler_class = cls._read_sampler_class(storage)
        # Read sampler status.
        return sampler_class.read_status(storage)

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

    @property
    def is_completed(self):
        """
        Boolean check if if the sampler has been completed by its own determination or if we have exceeded number of
        iterations
        """
        try:
            return self._sampler.is_completed
        except AttributeError:
            return self._sampler.iteration >= self._sampler.number_of_iterations

    def create(self, thermodynamic_state, sampler_states, topography, protocol,
               storage, restraint=None, anisotropic_dispersion_cutoff=None,
               alchemical_regions=None, alchemical_factory=None, metadata=None):
        """Create a new AlchemicalPhase calculation for a specified protocol.

        If ``anisotropic_dispersion_cutoff`` is different than ``None``. The
        end states of the phase will be reweighted. The fully interacting
        state accounts for:

            1. The truncation of nonbonded interactions.

            2. The reciprocal space which is not modeled in alchemical
            states if an Ewald method is used for long-range interactions.

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
            The dictionary ``{parameter_name: list_of_parameter_values}`` defining
            the protocol. All the parameter values list must have the same
            number of elements.
        storage : str or Reporter
            The path to the storage file or a Reporter object to forward
            to the sampler. In the future, this will be able to take a
            Storage class as well.
        restraint : ReceptorLigandRestraint, optional
            Restraint to add between protein and ligand. This must be specified
            for ligand-receptor systems in non-periodic boxes.
        anisotropic_dispersion_cutoff : simtk.openmm.Quantity, 'auto', or None, optional, default None
            If specified, this is the cutoff at which to reweight long range
            interactions of the end states to correct for anisotropic dispersions.

            If `'auto'`, then the distance is automatically chosen based on the minimum possible size it can be given
            the box volume, then behaves as if a Quantity was passed in.

            If `None`, the correction won't be applied (units of length, default
            is None).
        alchemical_regions : openmmtools.alchemy.AlchemicalRegion or None, optional, default: None
            If specified, this is the ``AlchemicalRegion`` that will be passed
            to the ``AbsoluteAlchemicalFactory``, otherwise the ligand will be
            alchemically modified according to the given protocol.
        alchemical_factory : openmmtools.alchemy.AbsoluteAlchemicalFactory, optional
            If specified, this ``AbsoluteAlchemicalFactory`` will be used instead of
            the one created with default options.
        metadata : dict, optional
            Simulation metadata to be stored in the file.

        """
        # Check that protocol has same number of states for each parameter.
        len_protocol_parameters = {par_name: len(path) for par_name, path in protocol.items()}
        if len(set(len_protocol_parameters.values())) != 1:
            raise ValueError('The protocol parameters have a different number '
                             'of states: {}'.format(len_protocol_parameters))

        # Do not modify passed thermodynamic state.
        reference_thermodynamic_state = copy.deepcopy(thermodynamic_state)
        thermodynamic_state = copy.deepcopy(thermodynamic_state)
        reference_system = thermodynamic_state.system
        is_periodic = thermodynamic_state.is_periodic
        is_complex = len(topography.receptor_atoms) > 0

        # We currently don't support reaction field.
        _, nonbonded_force = mmtools.forces.find_forces(reference_system, openmm.NonbondedForce,
                                                        only_one=True)
        nonbonded_method = nonbonded_force.getNonbondedMethod()
        if nonbonded_method == openmm.NonbondedForce.CutoffPeriodic:
            raise RuntimeError('CutoffPeriodic is not supported yet. Use PME for explicit solvent.')

        # Make sure sampler_states is a list of SamplerStates.
        if isinstance(sampler_states, mmtools.states.SamplerState):
            sampler_states = [sampler_states]

        # Initialize metadata storage and handle default argument.
        # We'll use the sampler full name for resuming, the reference
        # thermodynamic state for minimization and the topography for
        # ligand randomization.
        if metadata is None:
            metadata = dict()
        sampler_full_name = mmtools.utils.typename(self._sampler.__class__)
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
            alchemical_factory = mmtools.alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True)
        alchemical_system = alchemical_factory.create_alchemical_system(thermodynamic_state.system,
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

        # Temperature and pressure at the end states should
        # be the same or the analysis won't make sense.
        for state_property in ['temperature', 'pressure']:
            if getattr(compound_states[0], state_property) != getattr(compound_states[-1], state_property):
                raise ValueError('The {}s of the end states must be the same.'.format(state_property))

        # Expanded cutoff unsampled states.
        # ---------------------------------

        # TODO should we allow expanded states for non-periodic systems?
        logger.debug('Creating expanded cutoff states...')
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
            ``1.0 * unit.kilojoules_per_mole / unit.nanometers``).
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
            ``sigma_multiplier * receptor_radius_of_gyration`` (default is 2.0).
        close_cutoff : simtk.unit.Quantity, optional
            Each random placement proposal will be rejected if the ligand
            ends up being closer to the receptor than this cutoff (units of
            length, default is ``1.5*unit.angstrom``).

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
    def _read_sampler_class(storage):
        """Retrieve the MultiStateSampler class used from the storage."""
        # Handle str and Reporter argument value.
        if isinstance(storage, str):
            reporter = multistate.MultiStateReporter(storage)
        else:
            reporter = storage

        # Check if netcdf file exists.
        if not reporter.storage_exists():
            reporter.close()
            raise FileNotFoundError('Storage file at {} does not exists; '
                                    'cannot resume.'.format(reporter.filepath))

        # TODO: this should skip the Reporter and use the Storage to read storage.metadata.
        # Open Reporter for reading and read sampler class name.
        reporter.open(mode='r')
        sampler_full_name = reporter.read_dict('metadata/sampler_full_name')
        reporter.close()

        # Retrieve the sampler class.
        module_name, sampler_class_name = sampler_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, sampler_class_name)
        return sampler_class

    @staticmethod
    def _expand_state_cutoff(thermodynamic_state, expanded_cutoff_distance,
                             replace_reaction_field=False, switch_width=None):
        """Expand the thermodynamic state cutoff to the given one.

        If replace_reaction_field is True, the system will be modified
        to use an UnshiftedReactionFieldForce. In this case switch_width
        must be specified.

        """
        # If we use a barostat we leave more room for volume fluctuations or
        # we risk fatal errors. This is how much we allow the box size to change.
        fluctuation_size = 0.8

        # Do not modify passed thermodynamic state.
        thermodynamic_state = copy.deepcopy(thermodynamic_state)
        system = thermodynamic_state.system

        # Determine minimum box side dimension. The theoretical maximal allowed cutoff
        # is given by half the norm of the smallest vector, but OpenMM limits it to
        # the minimum diagonal element of the box vector matrix for efficiency.
        box_vectors = system.getDefaultPeriodicBoxVectors()
        min_box_dimension = min([vector[i] for i, vector in enumerate(box_vectors)])

        # Determine cutoff automatically if requested.
        # We leave more space if the volume fluctuates.
        if expanded_cutoff_distance == 'auto':
            if thermodynamic_state.pressure is None:
                expanded_cutoff_distance = min_box_dimension * 0.99 / 2.0
            else:
                expanded_cutoff_distance = min_box_dimension * fluctuation_size / 2.0
            expanded_cutoff_distance = min(expanded_cutoff_distance, 16*unit.angstroms)
        # Otherwise check that requested cutoff is within fluctuation limits. If the
        # state is in NVT and the cutoff is too big, OpenMM will raise an exception
        # on Context creation.
        elif (thermodynamic_state.pressure is not None and
                          min_box_dimension * fluctuation_size < 2.0 * expanded_cutoff_distance):
            raise RuntimeError('Barostated box sides must be at least {} Angstroms '
                               'to correct for missing dispersion interactions. The '
                               'minimum dimension of the provided box is {} Angstroms'
                               ''.format(expanded_cutoff_distance/unit.angstrom * 2,
                                         min_box_dimension/unit.angstrom))

        logger.debug('Setting cutoff for fully interacting system to {}. The minimum box '
                     'dimension is {}.'.format(expanded_cutoff_distance, min_box_dimension))

        # Expanded forces cutoff.
        for force in system.getForces():
            try:
                force_cutoff = force.getCutoffDistance()
            except AttributeError:
                    pass
            else:
                # We don't want to reduce the cutoff if it's already large.
                if force_cutoff < expanded_cutoff_distance:
                    cutoff_diff = expanded_cutoff_distance - force_cutoff
                    switching_distance = force.getSwitchingDistance()

                    # Expand cutoff preserving the original switch width.
                    # We don't need to check if we are using a switch since
                    # there is a setting for that.
                    force.setCutoffDistance(expanded_cutoff_distance)
                    force.setSwitchingDistance(switching_distance + cutoff_diff)

        # Replace reaction field NonbondedForce to remove constant shift term.
        # AbsoluteAlchemicalFactory already does it for the other states.
        if replace_reaction_field:
            mmtools.forcefactories.replace_reaction_field(system, return_copy=False,
                                                          switch_width=switch_width)

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
            alchemical_counterions = mpi.run_single_node(0, pipeline.find_alchemical_counterions,
                                                         system, topography, alchemical_region_name,
                                                         broadcast_result=True)
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
