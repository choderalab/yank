#!/usr/local/bin/env python

# ==============================================================================
# FILE DOCSTRING
# ==============================================================================

"""

Restraints
==========

Automated selection and imposition of receptor-ligand restraints for absolute
alchemical binding free energy calculations, along with computation of the
standard state correction.


"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import abc
import math
import random
import inspect
import logging
import functools
import itertools

import numpy as np
import scipy.integrate
import mdtraj as md
import openmmtools as mmtools
from simtk import openmm, unit

from . import pipeline, utils
from .utils import methoddispatch

from typing import Set, Iterable

logger = logging.getLogger(__name__)


# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

V0 = 1660.53928 * unit.angstroms**3  # standard state volume


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class RestraintStateError(mmtools.states.ComposableStateError):
    """Error raised by an :class:`RestraintState`."""
    pass


class RestraintParameterError(Exception):
    """Error raised by a :class:`ReceptorLigandRestraint`."""
    pass


# ==============================================================================
# Dispatch appropriate restraint type from registered restraint classes
# ==============================================================================

def available_restraint_classes():
    """
    Return all available restraint classes.

    Returns
    -------
    restraint_classes : dict of {str : class}
        ``restraint_classes[name]`` is the class corresponding to ``name``

    """
    restraint_subclasses = mmtools.utils.find_all_subclasses(ReceptorLigandRestraint,
                                                             discard_abstract=True)

    # Build an index of all names, ensuring there are no name collisions.
    available_restraints = dict()
    for cls in restraint_subclasses:
        cls_name = cls.__name__
        if cls_name in available_restraints:
            raise ValueError("More than one restraint subclass has the name '{}'.".format(cls_name))
        else:
            available_restraints[cls_name] = cls

    return available_restraints


def available_restraint_types():
    """
    List all available restraint types.

    Returns
    -------
    available_restraint_types : list of str
        List of names of available restraint classes

    """
    available_restraints = available_restraint_classes()
    return list(available_restraints.keys())


def create_restraint(restraint_type, **kwargs):
    """Factory of receptor-ligand restraint objects.

    Parameters
    ----------
    restraint_type : str
        Restraint type name matching a register (imported) subclass of :class:`ReceptorLigandRestraint`.
    kwargs
        Parameters to pass to the restraint constructor.

    """
    cls = mmtools.utils.find_subclass(ReceptorLigandRestraint, restraint_type)
    return cls(**kwargs)


# ==============================================================================
# ComposableState class to control the strength of restraints.
# ==============================================================================

class RestraintState(object):
    """
    The state of a restraint.

    A ``ComposableState`` controlling the strength of a restraint
    through its ``lambda_restraints`` property.

    Parameters
    ----------
    lambda_restraints : float
        The strength of the restraint. Must be between 0 and 1.

    Attributes
    ----------
    lambda_restraints

    Examples
    --------
    Create a system in a thermodynamic state.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 300*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    Apply a Harmonic restraint between receptor and protein. Let the restraint
    automatically determine all the parameters.

    >>> restraint = Harmonic()
    >>> restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    >>> restraint.restrain_state(thermodynamic_state)

    Create a ``RestraintState`` object to control the strength of the restraint.

    >>> restraint_state = RestraintState(lambda_restraints=1.0)

    ``RestraintState`` implements the ``IComposableState`` interface, so it can be
    used with ``CompoundThermodynamicState``.

    >>> compound_state = states.CompoundThermodynamicState(thermodynamic_state=thermodynamic_state,
    ...                                                    composable_states=[restraint_state])
    >>> compound_state.lambda_restraints
    1.0
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = compound_state.create_context(integrator)
    >>> context.getParameter('lambda_restraints')
    1.0

    You can control the parameters in the OpenMM Context by setting the state's
    attributes. To To deactivate the restraint, set `lambda_restraints` to 0.0.

    >>> compound_state.lambda_restraints = 0.0
    >>> compound_state.apply_to_context(context)
    >>> context.getParameter('lambda_restraints')
    0.0

    """
    def __init__(self, lambda_restraints):
        self.lambda_restraints = lambda_restraints

    @property
    def lambda_restraints(self):
        """Float: the strength of the applied restraint (between 0 and 1 inclusive)."""
        return self._lambda_restraints

    @lambda_restraints.setter
    def lambda_restraints(self, value):
        assert 0.0 <= value <= 1.0
        self._lambda_restraints = float(value)

    def apply_to_system(self, system):
        """
        Set the strength of the system's restraint to this.

        System is updated in-place

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.

        Raises
        ------
        RestraintStateError
            If the system does not have any ``CustomForce`` with a
            ``lambda_restraint`` global parameter.

        """
        # Set lambda_restraints in all forces that have it.
        for force, parameter_id in self._get_system_forces_parameters(system):
            force.setGlobalParameterDefaultValue(parameter_id, self._lambda_restraints)

    def check_system_consistency(self, system):
        """
        Check if the system's restraint is in this restraint state.

        It raises a :class:`RestraintStateError` if the restraint is not consistent
        with the state.

        Parameters
        ----------
        system : simtk.openmm.System
            The system with the restraint to test.

        Raises
        ------
        RestraintStateError
            If the system is not consistent with this state.

        """
        # Set lambda_restraints in all forces that have it.
        for force, parameter_id in self._get_system_forces_parameters(system):
            force_lambda = force.getGlobalParameterDefaultValue(parameter_id)
            if force_lambda != self.lambda_restraints:
                err_msg = 'Consistency check failed: system {}, state {}'
                raise RestraintStateError(err_msg.format(force_lambda, self._lambda_restraints))

    def apply_to_context(self, context):
        """Put the restraint in the `Context` into this state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to set.

        Raises
        ------
        RestraintStateError
            If the context does not have the required lambda global variables.

        """
        try:
            context.setParameter('lambda_restraints', self._lambda_restraints)
        except Exception:
            raise RestraintStateError('The context does not have a restraint.')

    def _standardize_system(self, system):
        """Standardize the given system.

        Set lambda_restraints of the system to 1.0.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to standardize.

        Raises
        ------
        RestraintStateError
            If the system is not consistent with this state.

        """
        # Set lambda_restraints to 1.0 in all forces that have it.
        for force, parameter_id in self._get_system_forces_parameters(system):
            force.setGlobalParameterDefaultValue(parameter_id, 1.0)

    def _on_setattr(self, standard_system, attribute_name):
        """Check if the standard system needs changes after a state attribute is set.

        Parameters
        ----------
        standard_system : simtk.openmm.System
            The standard system before setting the attribute.
        attribute_name : str
            The name of the attribute that has just been set or retrieved.

        Returns
        -------
        need_changes : bool
            True if the standard system has to be updated, False if no change
            occurred.

        """
        # There are no attributes that can be set that can alter the standard system.
        return False

    def _find_force_groups_to_update(self, context, current_context_state, memo):
        """Find the force groups whose energy must be recomputed after applying self.

        Parameters
        ----------
        context : Context
            The context, currently in `current_context_state`, that will
            be moved to this state.
        current_context_state : ThermodynamicState
            The full thermodynamic state of the given context. This is
            guaranteed to be compatible with self.
        memo : dict
            A dictionary that can be used by the state for memoization
            to speed up consecutive calls on the same context.

        Returns
        -------
        force_groups_to_update : set of int
            The indices of the force groups whose energy must be computed
            again after applying this state, assuming the context to be in
            `current_context_state`.
        """
        # Check if lambda_restraints will change.
        if self.lambda_restraints == current_context_state.lambda_restraints:
            return set()

        # Update memo if this is the first call for this context.
        if len(memo) == 0:
            system = context.getSystem()
            for force, _ in self._get_system_forces_parameters(system):
                memo['lambda_restraints'] = force.getForceGroup()
        return {memo['lambda_restraints']}

    @staticmethod
    def _get_system_forces_parameters(system):
        """Yields the system's forces having a ``lambda_restraints`` parameter.

        Yields
        ------
        A tuple force, ``parameter_index`` for each force with ``lambda_restraints``.

        """
        found_restraint = False

        # Retrieve all the forces with global supported parameters.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            try:
                n_global_parameters = force.getNumGlobalParameters()
            except AttributeError:
                continue
            for parameter_id in range(n_global_parameters):
                parameter_name = force.getGlobalParameterName(parameter_id)
                if parameter_name == 'lambda_restraints':
                    found_restraint = True
                    yield force, parameter_id
                    break

        # Raise error if the system doesn't have a restraint.
        if found_restraint is False:
            raise RestraintStateError('The system does not have a restraint.')

    def __getstate__(self):
        return dict(lambda_restraints=self._lambda_restraints)

    def __setstate__(self, serialization):
        self.lambda_restraints = serialization['lambda_restraints']


# ==============================================================================
# Atom selector tool
# ==============================================================================

class _AtomSelector(object):
    """
    Helper class to select atoms from semi-arbitrary selections based on a topography
    """

    def __init__(self, topography):
        self.topography = topography

    def compute_atom_intersect(self, input_atoms, topography_key: str, *additional_sets: Iterable[int]) -> Set[int]:

        """
        Compute the intersect of atoms formed from a given input, topography key to reference, and
        any additional sets for cross reference

        Parameters
        ----------
        input_atoms : str, iterable of int, or None
            Atom-like selection which can accept a :func:`yank.Topography.select`, or a sequence of ints, or None
            When given None, only the Topography key and additional sets are used
        topography_key : str
            Key in the :class:`yank.Topography` which is used to initially sub-select atoms
        additional_sets : set of int or iterable of int
            Any additional sets to cross reference

        Returns
        -------
        atom_intersect : set
            Set of atoms intersecting the input, the topography key, and the additional sets
        """

        topography = self.topography
        topography_set = set(getattr(topography, topography_key))
        # Ensure additions are sets
        additional_sets = [set(additional_set) for additional_set in additional_sets]
        if len(additional_sets) == 0:
            # Case no sets were provided
            additional_intersect = topography_set
        else:
            additional_intersect = set.intersection(*additional_sets)

        @functools.singledispatch
        def compute_atom_set(passed_atoms):
            """Helper function for doing set operations on heavy ligand atoms of all other types"""
            input_set = set(passed_atoms)
            intersect_set = input_set & additional_intersect & topography_set
            if intersect_set != input_set:
                return intersect_set
            else:
                # This ensures if no changes are made to the set, then passed atoms are returned unmodied
                return passed_atoms

        @compute_atom_set.register(type(None))
        def compute_atom_none(_):
            """Helper for None type parsing"""
            return topography_set & additional_intersect

        @compute_atom_set.register(str)
        def compute_atom_str(input_string):
            """Helper for string parsing"""
            output = topography.select(input_string, as_set=False)  # Preserve order
            set_output = set(output)
            # Ensure the selection is in the correct set
            set_combined = set_output & topography_set & additional_intersect
            final_output = [particle for particle in output if particle in set_combined]
            # Force output to be a normal int, don't need to worry about floats at this point, there should not be any
            # If they come out as np.int64's, OpenMM complains
            return [*map(int, final_output)]

        return compute_atom_set(input_atoms)


# ==============================================================================
# Base class for receptor-ligand restraints.
# ==============================================================================

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3


class ReceptorLigandRestraint(ABC):
    """
    A restraint preventing a ligand from drifting too far from its receptor.

    With replica exchange simulations, keeping the ligand close to the binding
    pocket can enhance mixing between the interacting and the decoupled state.
    This should be always used in implicit simulation, where there are no periodic
    boundary conditions.

    This restraint strength is controlled by a global context parameter called
    ``lambda_restraints``. You can easily control this variable through the
    ``RestraintState`` object.

    Notes
    -----
    Creating a subclass requires the following:

        1. Implement a constructor. Optionally this can leave all or a subset of
        the restraint parameters undefined. In this case, you need to provide
        an implementation of :func:`determine_missing_parameters`.

        2. Implement :func:`restrain_state` that add the restrain ``Force`` to the state's
        `System`.

        3. Implement :func:`get_standard_state_correction` to return standard state correction.

        4. Optionally, implement :func:`determine_missing_parameters` to fill in
        the parameters left undefined in the constructor.

    """

    @abc.abstractmethod
    def restrain_state(self, thermodynamic_state):
        """Add the restraint force to the state's `System`.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.

        """
        pass

    @abc.abstractmethod
    def get_standard_state_correction(self, thermodynamic_state):
        """Return the standard state correction.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        """
        pass

    def determine_missing_parameters(self, thermodynamic_state, sampler_state, topography):
        """
        Automatically choose undefined parameters.

        Optionally, a :class:`ReceptorLigandRestraint` can support the automatic
        determination of all or a subset of the parameters that can be
        left undefined in the constructor, making implementation of this method optional.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynmaic state to inspect
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor and ligand atoms.

        """
        raise NotImplementedError('{} does not support automatic determination of the '
                                  'restraint parameters'.format(self.__class__.__name__))

    @classmethod
    def _add_force_in_separate_group(cls, system, restraint_force):
        """Add the force to the System in a separate force group when possible."""
        # OpenMM supports a maximum of 32 force groups.
        available_force_groups = set(range(32))
        for force in system.getForces():
            available_force_groups.discard(force.getForceGroup())

        # If the System is full, just separate the force from nonbonded interactions.
        if len(available_force_groups) == 0:
            _, nonbonded_force = mmtools.forces.find_forces(system, openmm.NonbondedForce,
                                                            only_one=True)
            available_force_groups = set(range(32))
            available_force_groups.discard(nonbonded_force.getForceGroup())

        restraint_force.setForceGroup(min(available_force_groups))
        system.addForce(restraint_force)

    @property
    def _excluded_init_parameters(self):
        """
        List of excluded parameters from the :func:`__init__` call to ensure all non-atom selection
        parameters are defined
        """
        return ['self', 'restrained_receptor_atoms', 'restrained_ligand_atoms']

    @property
    def _parameters(self):
        """dict: restraint parameters in dict forms."""
        argspec = inspect.getfullargspec(self.__init__)
        parameter_names = argspec.args

        # Exclude non-parameters arguments.
        for exclusion in self._excluded_init_parameters:
            parameter_names.remove(exclusion)

        # Retrieve and store options.
        parameters = {parameter_name: getattr(self, parameter_name)
                      for parameter_name in parameter_names}
        return parameters


class _RestrainedAtomsProperty(object):
    """
    Descriptor of restrained atoms.

    Casts generic iterables of ints into lists.
    """

    def __init__(self, atoms_type):
        self._atoms_type = atoms_type

    @property
    def _attribute_name(self):
        """Name of the internally stored variable (read-only)."""
        return '_restrained_' + self._atoms_type + '_atoms'

    def __get__(self, instance, owner_class=None):
        return getattr(instance, self._attribute_name)

    def __set__(self, instance, new_restrained_atoms):
        # If we set the restrained attributes to None, no reason to check things.
        if new_restrained_atoms is not None:
            new_restrained_atoms = self._validate_atoms(new_restrained_atoms)
        setattr(instance, self._attribute_name, new_restrained_atoms)

    @methoddispatch
    def _validate_atoms(self, restrained_atoms):
        """Casts a generic iterable of ints into a list to support concatenation."""
        try:
            restrained_atoms = restrained_atoms.tolist()
        except AttributeError:
            restrained_atoms = list(restrained_atoms)
        return restrained_atoms


# ==============================================================================
# Base class for radially-symmetric receptor-ligand restraints.
# ==============================================================================

class RadiallySymmetricRestraint(ReceptorLigandRestraint):
    """
    Base class for radially-symmetric restraints between ligand and protein.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.
    The restraint strength is controlled by a global context parameter called
    'lambda_restraints'.

    With OpenCL, groups with more than 1 atom are supported only on 64bit
    platforms.

    The class allows the restrained atoms to be temporarily undefined, but in
    this case, :func:`determine_missing_parameters` must be called before using
    the restraint.

    Parameters
    ----------
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, any other
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Selection <yank.Topography.select>`.
        This can temporarily be left undefined, but :func:`determine_missing_parameters`
        must be called before using the Restraint object. The same if a DSL
        expression or Topography selection is provided (default is None).
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Selection <yank.Topography.select>`.
        This can temporarily be left undefined, but :func:`determine_missing_parameters`
        must be called before using the Restraint object. The same if a DSL
        expression or Topography selection is provided (default is None).

    Attributes
    ----------
    restrained_receptor_atoms : list of int, str, or None
        The indices of the receptor atoms to restrain, an MDTraj selection string, or a Topography selection
        string.
    restrained_ligand_atoms : list of int, str, None
        The indices of the receptor atoms to restrain, an MDTraj selection string, or a Topography selection
        string.

    Notes
    -----
    To create a subclass, follow these steps:

        1. Implement the :func:`_create_restraint_force` returning the
        force object modeling the restraint.

        2. Implement the property :func:`_are_restraint_parameters_defined`.

        3. Optionally, you can overwrite the :func:`_determine_restraint_parameters`
        method to automatically determine these parameters from the atoms positions.

    """
    def __init__(self, restrained_receptor_atoms=None, restrained_ligand_atoms=None):
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.restrained_ligand_atoms = restrained_ligand_atoms

    # -------------------------------------------------------------------------
    # Public properties.
    # -------------------------------------------------------------------------

    class _RadiallySymmetricRestrainedAtomsProperty(_RestrainedAtomsProperty):
        """
        Descriptor of restrained atoms.

        Extends `_RestrainedAtomsProperty` to handle single integers and strings.
        """
        _CENTROID_COMPUTE_STRING = ("You are specifying {} {} atoms, "
                                    "the final atoms will be chosen as the centroid of this set.")

        @methoddispatch
        def _validate_atoms(self, restrained_atoms):
            restrained_atoms = super()._validate_atoms(restrained_atoms)
            if len(restrained_atoms) > 1:
                logger.debug(self._CENTROID_COMPUTE_STRING.format("more than one", self._atoms_type))
            return restrained_atoms

        @_validate_atoms.register(str)
        def _validate_atoms_string(self, restrained_atoms):
            warn_string = self._CENTROID_COMPUTE_STRING.format("a string for", self._atoms_type)
            warn_string += 'but you MUST run "determine_missing_parameters" to process the string'
            logger.warning(warn_string)
            return restrained_atoms

        @_validate_atoms.register(int)
        def _validate_atoms_int(self, restrained_atoms):
            return [restrained_atoms]

    restrained_receptor_atoms = _RadiallySymmetricRestrainedAtomsProperty('receptor')
    restrained_ligand_atoms = _RadiallySymmetricRestrainedAtomsProperty('ligand')

    # -------------------------------------------------------------------------
    # Public methods.
    # -------------------------------------------------------------------------

    def restrain_state(self, thermodynamic_state):
        """Add the restraining Force(s) to the thermodynamic state's system.

        All the parameters must be defined at this point. An exception is
        raised if they are not.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.

        Raises
        ------
        RestraintParameterError
            If the restraint has undefined parameters.

        """
        # Check that restrained atoms are defined.
        if not self._are_restrained_atoms_defined:
            raise RestraintParameterError('Restraint {}: Undefined restrained '
                                          'atoms.'.format(self.__class__.__name__))

        # Create restraint force.
        restraint_force = self._get_restraint_force(self.restrained_receptor_atoms,
                                                    self.restrained_ligand_atoms)

        # Set periodic conditions on the force if necessary.
        restraint_force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        self._add_force_in_separate_group(system, restraint_force)
        thermodynamic_state.system = system

    def get_standard_state_correction(self, thermodynamic_state):
        """Return the standard state correction.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        correction : float
           The unit-less standard-state correction, in kT (at the
           temperature of the given thermodynamic state).

        """
        benchmark_id = 'Restraint {}: Computing standard state correction'.format(self.__class__.__name__)
        timer = mmtools.utils.Timer()
        timer.start(benchmark_id)

        restraint_force = self._get_restraint_force([0], [1])
        DeltaG = restraint_force.compute_standard_state_correction(thermodynamic_state, max_volume='system')
        logger.debug('Standard state correction: {:.3f} kT'.format(DeltaG))

        # Report elapsed time.
        timer.stop(benchmark_id)
        timer.report_timing()

        # Return standard state correction (in kT).
        return DeltaG

    def determine_missing_parameters(self, thermodynamic_state, sampler_state, topography):
        """Automatically determine missing parameters.

        If some parameters have been left undefined (i.e. the atoms to restrain
        or the restraint force parameters) this attempts to find them using the
        information in the states and the topography.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topography : yank.Topography, optional
            The topography with labeled receptor and ligand atoms.

        """

        # Determine restrained atoms, if needed.
        self._determine_restrained_atoms(sampler_state, topography)

        # Determine missing parameters. This is implemented in the subclass.
        self._determine_restraint_parameters(thermodynamic_state, sampler_state, topography)

    # -------------------------------------------------------------------------
    # Internal-usage: properties and methods for subclasses.
    # -------------------------------------------------------------------------

    @abc.abstractproperty
    def _are_restraint_parameters_defined(self):
        """bool: True if the restraint parameters are defined."""
        pass

    @abc.abstractmethod
    def _create_restraint_force(self, particles1, particles2):
        """Create a new restraint force between specified atoms.

        The property _are_restraint_parameters_defined is guaranteed to
        be True when this is called.

        Parameters
        ----------
        particles1 : list of int
            Indices of first group of atoms to restraint.
        particles2 : list of int
            Indices of second group of atoms to restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        pass

    def _determine_restraint_parameters(self, thermodynamic_state, sampler_state, topography):
        """Determine the missing bond parameters.

        Optionally, a subclass can implement this method to automatically
        define the bond parameters of the restraints from the information
        in the given states and topography. The default implementation just
        raises a NotImplemented error if `_are_restraint_parameters_defined`
        is False.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor, ligand atoms, and any regions defined.

        """
        # Raise exception only if the subclass doesn't already defines parameters.
        if not self._are_restraint_parameters_defined:
            raise NotImplementedError('Restraint {} cannot automatically determine '
                                      'bond parameters.'.format(self.__class__.__name__))

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    @property
    def _are_restrained_atoms_defined(self):
        """Check if the restrained atoms are defined well enough to make a restraint"""
        for atoms in [self.restrained_receptor_atoms, self.restrained_ligand_atoms]:
            # Atoms should be a list or None at this point due to the _RestrainedAtomsProperty class
            if atoms is None or not (isinstance(atoms, list) and len(atoms) > 0):
                return False
        return True

    def _get_restraint_force(self, particles1, particles2):
        """Check that the parameters are defined before calling the force constructor."""
        # Check if restraint parameters have been defined.
        if self._are_restraint_parameters_defined is None:
            err_msg = 'Restraint {}: Undefined bond parameters.'.format(self.__class__.__name__)
            raise RestraintParameterError(err_msg)

        # Create restraint force.
        return self._create_restraint_force(particles1, particles2)

    def _determine_restrained_atoms(self, sampler_state, topography):
        """Determine the atoms to restrain.

        If the user has explicitly specified which atoms to restrained, this
        does nothing, otherwise it picks the centroid of the receptor and
        the centroid of the ligand as the two atoms to restrain.

        Parameters
        ----------
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topography : yank.Topography, optional
            The topography with labeled receptor, ligand atoms, and any regions defined.

        """
        debug_msg = ('Restraint {}: Automatically picked restrained '
                     '{{0}} atom: {{0}}'.format(self.__class__.__name__))

        # No need to determine parameters if atoms have been given.
        if self._are_restrained_atoms_defined:
            return

        # Shortcuts
        positions = sampler_state.positions

        # If receptor and ligand atoms are explicitly provided, use those.
        restrained_ligand_atoms = self.restrained_ligand_atoms
        restrained_receptor_atoms = self.restrained_receptor_atoms

        @functools.singledispatch
        def compute_atom_set(input_atoms, topography_key, mapping_function):
            """
            Helper function for doing set operations on generic atom types.
            mapping_function not used in the generic catch-all, but is used in the None register
            """
            # Ensure the input atoms are only part of the topography_key atoms. Make no changes if they are
            input_atoms_set = set(input_atoms)
            set_topography_atoms = set(getattr(topography, topography_key))
            intersect_set = input_atoms_set & set_topography_atoms
            if intersect_set != input_atoms_set:
                logger.warning("Some atoms specified by {0} were not actual {0}! "
                               "Atoms not part of {0} will be ignored.".format(topography_key))
                final_atoms = list(intersect_set)
            else:
                final_atoms = list(input_atoms)
            return final_atoms

        @compute_atom_set.register(type(None))
        def compute_atom_none(_, topography_key, mapping_function):
            """Helper for None type parsing"""
            # Can't use list() here since mapping function returns a single integer.
            atom_selection = [mapping_function(positions, getattr(topography, topography_key))]
            logger.debug(debug_msg.format(topography_key, atom_selection))
            return atom_selection

        @compute_atom_set.register(str)
        def compute_atom_str(input_string, topography_key, _):
            """Helper for string parsing"""
            selection = topography.select(input_string, as_set=True)
            selection_with_top = selection & set(getattr(topography, topography_key))
            # Force output to be a normal int, dont need to worry about floats at this point, there should not be any
            # If they come out as np.int64's, OpenMM complains
            return [*map(int, selection_with_top)]

        self.restrained_ligand_atoms = compute_atom_set(restrained_ligand_atoms,
                                                        'ligand_atoms',
                                                        self._closest_atom_to_centroid)
        self.restrained_receptor_atoms = compute_atom_set(restrained_receptor_atoms,
                                                          'receptor_atoms',
                                                          self._closest_atom_to_centroid)

    @staticmethod
    def _closest_atom_to_centroid(positions, indices=None, masses=None):
        """
        Identify the closest atom to the centroid of the given coordinate set.

        Parameters
        ----------
        positions : unit.Quantity of natoms x 3 with units compatible with nanometers
           positions of object to identify atom closes to centroid
        indices : list of int, optional, default=None
           List of atoms indices for which closest atom to centroid is to be computed.
        masses : simtk.unit.Quantity of natoms with units compatible with amu
           Masses of particles used to weight distance calculation, if not None (default: None)

        Returns
        -------
        closest_atom : int
           Index of atom closest to centroid of specified atoms.

        """
        if indices is not None:
            positions = positions[indices, :]

        # Get dimensionless positions.
        x_unit = positions.unit
        x = positions / x_unit

        # Determine number of atoms.
        natoms = x.shape[0]

        # Compute (natoms,1) array of normalized weights.
        w = np.ones([natoms, 1])
        if masses is not None:
            w = masses / masses.unit # (natoms,) array
            w = np.reshape(w, (natoms, 1))  # (natoms,1) array
        w /= w.sum()

        # Compute centroid (still in dimensionless units).
        centroid = (np.tile(w, (1, 3)) * x).sum(0)  # (3,) array

        # Compute distances from centroid.
        distances = np.sqrt(((x - np.tile(centroid, (natoms, 1)))**2).sum(1))  # distances[i] is the distance from the centroid to particle i

        # Determine closest atom.
        closest_atom = int(np.argmin(distances))

        if indices is not None:
            closest_atom = indices[closest_atom]

        return closest_atom


# ==============================================================================
# Harmonic protein-ligand restraint.
# ==============================================================================

class Harmonic(RadiallySymmetricRestraint):
    """Impose a single harmonic restraint between ligand and protein.

    This can be used to prevent the ligand from drifting too far from the
    protein in implicit solvent calculations or to keep the ligand close
    to the binding pocket in the decoupled states to increase mixing.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.

    The energy expression of the restraint is given by

       ``E = lambda_restraints * (K/2)*r^2``

    where `K` is the spring constant, `r` is the distance between the
    two group centroids, and `lambda_restraints` is a scale factor that
    can be used to control the strength of the restraint. You can control
    ``lambda_restraints`` through :class:`RestraintState` class.

    The class supports automatic determination of the parameters left undefined or defined by strings
    in the constructor through :func:`determine_missing_parameters`.

    With OpenCL, groups with more than 1 atom are supported only on 64bit
    platforms.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole (default is None).
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression.
        or a :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).

    Attributes
    ----------
    restrained_receptor_atoms : list of int, str, or None
        The indices of the receptor atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.
    restrained_ligand_atoms : list of int, str, or None
        The indices of the ligand atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 300*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    you can create a completely defined restraint

    >>> restraint = Harmonic(spring_constant=8*unit.kilojoule_per_mole/unit.nanometers**2,
    ...                      restrained_receptor_atoms=[1644, 1650, 1678],
    ...                      restrained_ligand_atoms='resname TMP')

    Or automatically identify the parameters. When trying to impose a restraint
    with undefined parameters, RestraintParameterError is raised.

    >>> restraint = Harmonic()
    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, spring_constant=None, **kwargs):
        super(Harmonic, self).__init__(**kwargs)
        self.spring_constant = spring_constant

    @property
    def _are_restraint_parameters_defined(self):
        """bool: True if the restraint parameters are defined."""
        return self.spring_constant is not None

    def _create_restraint_force(self, particles1, particles2):
        """Create a new restraint force between specified atoms.

        Parameters
        ----------
        particles1 : list of int
            Indices of first group of atoms to restraint.
        particles2 : list of int
            Indices of second group of atoms to restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        # Create bond force and lambda_restraints parameter to control it.
        if len(particles1) == 1 and len(particles2) == 1:
            # CustomCentroidBondForce works only on 64bit platforms. When the
            # restrained groups only have 1 particle, we can use the standard
            # CustomBondForce so that we can support 32bit platforms too.
            return mmtools.forces.HarmonicRestraintBondForce(spring_constant=self.spring_constant,
                                                             restrained_atom_index1=particles1[0],
                                                             restrained_atom_index2=particles2[0])
        return mmtools.forces.HarmonicRestraintForce(spring_constant=self.spring_constant,
                                                     restrained_atom_indices1=particles1,
                                                     restrained_atom_indices2=particles2)

    def _determine_restraint_parameters(self, thermodynamic_state, sampler_state, topography):
        """Automatically choose a spring constant for the restraint force.

        The spring constant is selected to give 1 kT at one standard deviation
        of receptor atoms about the receptor restrained atom.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor and ligand atoms.

        """
        # Do not overwrite parameters that are already defined.
        if self.spring_constant is not None:
            return

        receptor_positions = sampler_state.positions[topography.receptor_atoms]
        sigma = pipeline.compute_radius_of_gyration(receptor_positions)

        # Compute corresponding spring constant.
        self.spring_constant = thermodynamic_state.kT / sigma**2

        logger.debug('Spring constant sigma, s = {:.3f} nm'.format(sigma / unit.nanometers))
        logger.debug('K = {:.1f} kcal/mol/A^2'.format(
            self.spring_constant / unit.kilocalories_per_mole * unit.angstroms**2))


# ==============================================================================
# Flat-bottom protein-ligand restraint.
# ==============================================================================

class FlatBottom(RadiallySymmetricRestraint):
    """A receptor-ligand restraint using a flat potential well with harmonic walls.

    An alternative choice to receptor-ligand restraints that uses a flat
    potential inside most of the protein volume with harmonic restraining
    walls outside of this. It can be used to prevent the ligand from
    drifting too far from protein in implicit solvent calculations while
    still exploring the surface of the protein for putative binding sites.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.

    More precisely, the energy expression of the restraint is given by

        ``E = lambda_restraints * step(r-r0) * (K/2)*(r-r0)^2``

    where ``K`` is the spring constant, ``r`` is the distance between the
    restrained atoms, ``r0`` is another parameter defining the distance
    at which the restraint is imposed, and ``lambda_restraints``
    is a scale factor that can be used to control the strength of the
    restraint. You can control ``lambda_restraints`` through the class
    :class:`RestraintState`.

    The class supports automatic determination of the parameters left undefined
    in the constructor through :func:`determine_missing_parameters`.

    With OpenCL, groups with more than 1 atom are supported only on 64bit
    platforms.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole (default is None).
    well_radius : simtk.unit.Quantity, optional
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance (default is None).
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression.
        or a :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).

    Attributes
    ----------
    restrained_receptor_atoms : list of int or None
        The indices of the receptor atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.
    restrained_ligand_atoms : list of int or None
        The indices of the ligand atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 298*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    You can create a completely defined restraint

    >>> restraint = FlatBottom(spring_constant=0.6*unit.kilocalorie_per_mole/unit.angstroms**2,
    ...                        well_radius=5.2*unit.nanometers, restrained_receptor_atoms=[1644, 1650, 1678],
    ...                        restrained_ligand_atoms='resname TMP')

    or automatically identify the parameters. When trying to impose a restraint
    with undefined parameters, RestraintParameterError is raised.

    >>> restraint = FlatBottom()
    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, spring_constant=None, well_radius=None, **kwargs):
        super(FlatBottom, self).__init__(**kwargs)
        self.spring_constant = spring_constant
        self.well_radius = well_radius

    @property
    def _energy_function(self):
        """str: energy expression of the restraint force."""
        return 'step(distance(g1,g2)-r0) * (K/2)*(distance(g1,g2)-r0)^2'

    @property
    def _are_restraint_parameters_defined(self):
        """bool: True if the restraint parameters are defined."""
        return self.spring_constant is not None and self.well_radius is not None

    def _create_restraint_force(self, particles1, particles2):
        """Create a new restraint force between specified atoms.

        Parameters
        ----------
        particles1 : list of int
            Indices of first group of atoms to restraint.
        particles2 : list of int
            Indices of second group of atoms to restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        # Create bond force and lambda_restraints parameter to control it.
        if len(particles1) == 1 and len(particles2) == 1:
            # CustomCentroidBondForce works only on 64bit platforms. When the
            # restrained groups only have 1 particle, we can use the standard
            # CustomBondForce so that we can support 32bit platforms too.
            return mmtools.forces.FlatBottomRestraintBondForce(spring_constant=self.spring_constant,
                                                               well_radius=self.well_radius,
                                                               restrained_atom_index1=particles1[0],
                                                               restrained_atom_index2=particles2[0])
        return mmtools.forces.FlatBottomRestraintForce(spring_constant=self.spring_constant,
                                                       well_radius=self.well_radius,
                                                       restrained_atom_indices1=particles1,
                                                       restrained_atom_indices2=particles2)

    def _determine_restraint_parameters(self, thermodynamic_state, sampler_state, topography):
        """Automatically choose a spring constant and well radius.

        The spring constant, is set to 5.92 kcal/mol/A**2, the well
        radius is set at twice the robust estimate of the standard
        deviation (from mean absolute deviation) plus 5 A.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor and ligand atoms.

        """
        # Determine number of atoms.
        n_atoms = len(topography.receptor_atoms)

        # Check that restrained receptor atoms are in expected range.
        if any(atom_id >= n_atoms for atom_id in self.restrained_receptor_atoms):
            raise ValueError('Receptor atoms {} were selected for restraint, but system '
                             'only has {} atoms.'.format(self.restrained_receptor_atoms, n_atoms))

        # Compute well radius if the user hasn't specified it in the constructor.
        if self.well_radius is None:
            # Get positions of mass-weighted centroid atom.
            # (Working in non-unit-bearing floats for speed.)
            x_unit = sampler_state.positions.unit
            x_restrained_atoms = sampler_state.positions[self.restrained_receptor_atoms, :] / x_unit
            system = thermodynamic_state.system
            masses = np.array([system.getParticleMass(i) / unit.dalton for i in self.restrained_receptor_atoms])
            x_centroid = np.average(x_restrained_atoms, axis=0, weights=masses)

            # Get dimensionless receptor and ligand positions.
            x_receptor = sampler_state.positions[topography.receptor_atoms, :] / x_unit
            x_ligand = sampler_state.positions[topography.ligand_atoms, :] / x_unit

            # Compute maximum square distance from the centroid to any receptor atom.
            # dist2_centroid_receptor[i] is the squared distance from the centroid to receptor atom i.
            dist2_centroid_receptor = pipeline.compute_squared_distances([x_centroid], x_receptor)
            max_dist_receptor = np.sqrt(dist2_centroid_receptor.max()) * x_unit

            # Compute maximum length of the ligand. dist2_ligand_ligand[i][j] is the
            # squared distance between atoms i and j of the ligand.
            dist2_ligand_ligand = pipeline.compute_squared_distances(x_ligand, x_ligand)
            max_length_ligand = np.sqrt(dist2_ligand_ligand.max()) * x_unit

            # Compute the radius of the flat bottom restraint.
            self.well_radius = max_dist_receptor + max_length_ligand/2 + 5*unit.angstrom

        # Set default spring constant if the user hasn't specified it in the constructor.
        if self.spring_constant is None:
            self.spring_constant = 10.0 * thermodynamic_state.kT / unit.angstroms**2

        logger.debug('restraint distance r0 = {:.1f} A'.format(self.well_radius / unit.angstroms))
        logger.debug('K = {:.1f} kcal/mol/A^2'.format(
            self.spring_constant / unit.kilocalories_per_mole * unit.angstroms**2))


# ==============================================================================
# Orientation-dependent receptor-ligand restraints.
# ==============================================================================

class Boresch(ReceptorLigandRestraint):
    """Impose Boresch-style orientational restraints on protein-ligand system.

    This restraints the ligand binding mode by constraining 1 distance, 2
    angles and 3 dihedrals between 3 atoms of the receptor and 3 atoms of
    the ligand.

    More precisely, the energy expression of the restraint is given by

        .. code-block:: python

            E = lambda_restraints * {
                    K_r/2 * [|r3 - l1| - r_aA0]^2 +
                    + K_thetaA/2 * [angle(r2,r3,l1) - theta_A0]^2 +
                    + K_thetaB/2 * [angle(r3,l1,l2) - theta_B0]^2 +
                    + K_phiA/2 * [dihedral(r1,r2,r3,l1) - phi_A0]^2 +
                    + K_phiB/2 * [dihedral(r2,r3,l1,l2) - phi_B0]^2 +
                    + K_phiC/2 * [dihedral(r3,l1,l2,l3) - phi_C0]^2
                }


    , where the parameters are:

        ``r1``, ``r2``, ``r3``: the coordinates of the 3 receptor atoms.

        ``l1``, ``l2``, ``l3``: the coordinates of the 3 ligand atoms.

        ``K_r``: the spring constant for the restrained distance ``|r3 - l1|``.

        ``r_aA0``: the equilibrium distance of ``|r3 - l1|``.

        ``K_thetaA``, ``K_thetaB``: the spring constants for ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        ``theta_A0``, ``theta_B0``: the equilibrium angles of ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        ``K_phiA``, ``K_phiB``, ``K_phiC``: the spring constants for ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        ``phi_A0``, ``phi_B0``, ``phi_C0``: the equilibrium torsion of ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        ``lambda_restraints``: a scale factor that can be used to control the strength
        of the restraint.

    You can control ``lambda_restraints`` through the class :class:`RestraintState`.

    The class supports automatic determination of the parameters left undefined
    in the constructor through :func:`determine_missing_parameters`.

    *Warning*: Symmetry corrections for symmetric ligands are not automatically applied.
    See Ref [1] and [2] for more information on correcting for ligand symmetry.

    *Warning*: Only heavy atoms can be restrained. Hydrogens will automatically be excluded.

    Parameters
    ----------
    restrained_receptor_atoms : iterable of int, str, or None; Optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        If this is a list of three ints, the receptor atoms will be restrained in order, r1, r2, r3. If there are more
        than three entries or the selection string resolves more than three atoms, the three restrained atoms will
        be chosen at random from the selection.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, str, or None; Optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        If this is a list of three ints, the receptor atoms will be restrained in order, l1, l2, l3. If there are more
        than three entries or the selection string resolves more than three atoms, the three restrained atoms will
        be chosen at random from the selection.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    K_r : simtk.unit.Quantity, optional
        The spring constant for the restrained distance ``|r3 - l1|`` (units
        compatible with kilocalories_per_mole/angstrom**2).
    r_aA0 : simtk.unit.Quantity, optional
        The equilibrium distance between r3 and l1 (units of length).
    K_thetaA, K_thetaB : simtk.unit.Quantity, optional
        The spring constants for ``angle(r2, r3, l1)`` and ``angle(r3, l1, l2)``
        (units compatible with kilocalories_per_mole/radians**2).
    theta_A0, theta_B0 : simtk.unit.Quantity, optional
        The equilibrium angles of ``angle(r2, r3, l1)`` and ``angle(r3, l1, l2)``
        (units compatible with radians).
    K_phiA, K_phiB, K_phiC : simtk.unit.Quantity, optional
        The spring constants for ``dihedral(r1, r2, r3, l1)``,
        ``dihedral(r2, r3, l1, l2)`` and ``dihedral(r3,l1,l2,l3)`` (units compatible
        with kilocalories_per_mole/radians**2).
    phi_A0, phi_B0, phi_C0 : simtk.unit.Quantity, optional
        The equilibrium torsion of ``dihedral(r1,r2,r3,l1)``, ``dihedral(r2,r3,l1,l2)``
        and ``dihedral(r3,l1,l2,l3)`` (units compatible with radians).
    standard_state_correction_method : 'analytical' or 'numeric', optional
        The method to use to estimate the standard state correction (default
        is 'analytical').

    Attributes
    ----------
    restrained_receptor_atoms : list of int
        The indices of the 3 receptor atoms to restrain [r1, r2, r3].
    restrained_ligand_atoms : list of int
        The indices of the 3 ligand atoms to restrain [l1, l2, l3].
    standard_state_correction_method

    References
    ----------
    [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839
    [2] Mobley DL, Chodera JD, and Dill KA. J Chem Phys 125:084902, 2006.
        https://dx.doi.org/10.1063%2F1.2221683

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 298*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    Create a partially defined restraint

    >>> restraint = Boresch(restrained_receptor_atoms=[1335, 1339, 1397],
    ...                     restrained_ligand_atoms=[2609, 2607, 2606],
    ...                     K_r=20.0*unit.kilocalories_per_mole/unit.angstrom**2,
    ...                     r_aA0=0.35*unit.nanometer)

    and automatically identify the other parameters. When trying to impose
    a restraint with undefined parameters, RestraintParameterError is raised.

    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_r=None, r_aA0=None,
                 K_thetaA=None, theta_A0=None,
                 K_thetaB=None, theta_B0=None,
                 K_phiA=None, phi_A0=None,
                 K_phiB=None, phi_B0=None,
                 K_phiC=None, phi_C0=None,
                 standard_state_correction_method='analytical'):
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.restrained_ligand_atoms = restrained_ligand_atoms
        self.K_r = K_r
        self.r_aA0 = r_aA0
        self.K_thetaA, self.K_thetaB = K_thetaA, K_thetaB
        self.theta_A0, self.theta_B0 = theta_A0, theta_B0
        self.K_phiA, self.K_phiB, self.K_phiC = K_phiA, K_phiB, K_phiC
        self.phi_A0, self.phi_B0, self.phi_C0 = phi_A0, phi_B0, phi_C0
        self.standard_state_correction_method = standard_state_correction_method

    # -------------------------------------------------------------------------
    # Public properties.
    # -------------------------------------------------------------------------

    class _BoreschRestrainedAtomsProperty(_RestrainedAtomsProperty):
        """
        Descriptor of restrained atoms.

        Extends `_RestrainedAtomsProperty` to handle single integers and strings.
        """

        _MUST_COMPUTE_STRING = ('You are specifying {} {} atoms, '
                                'the final atoms will be chosen at from this set but you MUST '
                                'run "determine_missing_parameters"')

        @methoddispatch
        def _validate_atoms(self, restrained_atoms):
            restrained_atoms = super()._validate_atoms(restrained_atoms)
            if len(restrained_atoms) < 3:
                raise ValueError('At least three {} atoms are required to impose a '
                                 'Boresch-style restraint.'.format(self._atoms_type))
            elif len(restrained_atoms) > 3:
                logger.warning(self._MUST_COMPUTE_STRING.format("more than three", self._atoms_type))
            return restrained_atoms

        @_validate_atoms.register(str)
        def _cast_atom_string(self, restrained_atoms):
            logger.warning(self._MUST_COMPUTE_STRING.format("a string for", self._atoms_type))
            return restrained_atoms

    restrained_receptor_atoms = _BoreschRestrainedAtomsProperty('receptor')
    restrained_ligand_atoms = _BoreschRestrainedAtomsProperty('ligand')

    @property
    def standard_state_correction_method(self):
        """str: The default method to use in :func:`get_standard_state_correction`.

        This can be either 'analytical' or 'numerical'.

        """
        return self._standard_state_correction_method

    @standard_state_correction_method.setter
    def standard_state_correction_method(self, value):
        if value not in ['analytical', 'numerical']:
            raise ValueError("The standard state correction method must be one between "
                             "'analytical' and 'numerical', got {}.".format(value))
        self._standard_state_correction_method = value

    # -------------------------------------------------------------------------
    # Public methods.
    # -------------------------------------------------------------------------

    def restrain_state(self, thermodynamic_state):
        """Add the restraint force to the state's ``System``.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.

        """
        # TODO replace dihedral restraints with negative log von Mises distribution?
        #       https://en.wikipedia.org/wiki/Von_Mises_distribution, the von Mises parameter
        #       kappa would be computed from the desired standard deviation (kappa ~ sigma**(-2))
        #       and the standard state correction would need to be modified.

        # Check if all parameters are defined.
        self._check_parameters_defined()

        energy_function = """
            lambda_restraints * E;
            E = (K_r/2)*(distance(p3,p4) - r_aA0)^2
            + (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2
            + (K_phiA/2)*dphi_A^2 + (K_phiB/2)*dphi_B^2 + (K_phiC/2)*dphi_C^2;
            dphi_A = dA - floor(dA/(2*pi)+0.5)*(2*pi); dA = dihedral(p1,p2,p3,p4) - phi_A0;
            dphi_B = dB - floor(dB/(2*pi)+0.5)*(2*pi); dB = dihedral(p2,p3,p4,p5) - phi_B0;
            dphi_C = dC - floor(dC/(2*pi)+0.5)*(2*pi); dC = dihedral(p3,p4,p5,p6) - phi_C0;
            pi = %f;
            """ % np.pi

        # Add constant definitions to the energy function
        for name, value in self._parameters.items():
            energy_function += '%s = %f; ' % (name, value.value_in_unit_system(unit.md_unit_system))

        # Create the force
        n_particles = 6  # number of particles involved in restraint: p1 ... p6
        restraint_force = openmm.CustomCompoundBondForce(n_particles, energy_function)
        restraint_force.addGlobalParameter('lambda_restraints', 1.0)
        restraint_force.addBond(self.restrained_receptor_atoms + self.restrained_ligand_atoms, [])
        restraint_force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        self._add_force_in_separate_group(system, restraint_force)
        thermodynamic_state.system = system

    def get_standard_state_correction(self, thermodynamic_state):
        """Return the standard state correction.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT).

        """
        if self.standard_state_correction_method == 'analytical':
            return self._get_standard_state_correction_analytical(thermodynamic_state)
        else:  # The property checks that the value is known in the setter.
            return self._get_standard_state_correction_numerical(thermodynamic_state)

    def determine_missing_parameters(self, thermodynamic_state, sampler_state, topography):
        """Determine parameters and restrained atoms automatically.

        Currently, all equilibrium values are measured from the initial structure,
        while spring constants are set to 20 kcal/(mol A**2) or 20 kcal/(mol rad**2)
        as in Ref [1]. The restrained atoms are selected so that the analytical
        standard state correction will be valid.

        Parameters that have been already specified are left untouched.

        Future iterations of this feature will introduce the ability to extract
        equilibrium parameters and spring constants from a short simulation.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topography : yank.Topography, optional
            The topography with labeled receptor and ligand atoms.

        """
        MAX_ATTEMPTS = 100

        logger.debug('Automatically selecting restraint atoms and parameters:')

        # If restrained atoms are already specified, we only need to determine parameters.
        if self._are_restrained_atoms_defined:
            self._determine_restraint_parameters(sampler_state, topography)
        else:
            # Keep selecting random retrained atoms until the parameters
            # make the standard state correction robust.
            for attempt in range(MAX_ATTEMPTS):
                logger.debug('Attempt {} / {} at automatically selecting atoms and '
                             'restraint parameters...'.format(attempt, MAX_ATTEMPTS))

                # Randomly pick non-collinear atoms.
                restrained_atoms = self._pick_restrained_atoms(sampler_state, topography)
                self.restrained_receptor_atoms = restrained_atoms[:3]
                self.restrained_ligand_atoms = restrained_atoms[3:]

                # Determine restraint parameters for these atoms.
                self._determine_restraint_parameters(sampler_state, topography)

                # Check if we have found a good solution.
                if self._is_analytical_correction_robust(thermodynamic_state.kT) is True:
                    break

        # Check if the analytical standard state correction is robust with these parameters.
        # This check is must be performed both in the case where the user has provided the
        # restrained atoms, and in the case where we exhausted the number of attempts.
        if (not self._is_analytical_correction_robust(thermodynamic_state.kT) and
                self.standard_state_correction_method == 'analytical'):
            logger.warning('The provided restrained atoms do not guarantee a robust calculation of '
                           'the standard state correction. Switching to the numerical scheme.')
            self.standard_state_correction_method = 'numerical'

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    @property
    def _excluded_init_parameters(self):
        param_list = super()._excluded_init_parameters
        param_list.append("standard_state_correction_method")
        return param_list

    def _check_parameters_defined(self):
        """Raise an exception there are still parameters undefined."""
        if not self._are_restrained_atoms_defined:
            raise RestraintParameterError('Undefined restrained atoms.')

        # Find undefined parameters and raise error.
        undefined_parameters = [name for name, value in self._parameters.items() if value is None]
        if len(undefined_parameters) > 0:
            err_msg = 'Undefined parameters for Boresch restraint: {}'.format(undefined_parameters)
            raise RestraintParameterError(err_msg)

    @property
    def _are_restrained_atoms_defined(self):
        """Check if the restrained atoms are defined well enough to make a restraint"""
        for atoms in [self.restrained_receptor_atoms, self.restrained_ligand_atoms]:
            # Atoms should be a list or None at this point due to the _RestrainedAtomsProperty class
            if atoms is None or not (isinstance(atoms, list) and len(atoms) == 3):
                return False
        return True

    def _is_analytical_correction_robust(self, kT):
        """Check that the analytical standard state correction is valid for the current parameters."""
        N_SIGMA = 4
        is_robust = True

        for name in ['A', 'B']:
            theta0 = getattr(self, 'theta_' + name + '0')
            K = getattr(self, 'K_theta' + name)
            sigma = unit.sqrt(N_SIGMA * kT * 2.0 / K)
            if theta0 < sigma or theta0 > np.pi*unit.radians - sigma:
                logger.debug('theta_' + name + '0 is too close to 0 or pi '
                             'for standard state correction to be accurate.')
                is_robust = False

        r0 = getattr(self, 'r_aA0')
        K = getattr(self, 'K_r')
        sigma = unit.sqrt(N_SIGMA * kT * 2.0 / K)
        if r0 < sigma:
            logger.debug('r_aA0 is too close to 0 for standard state correction to be accurate.')
            is_robust = False

        return is_robust

    def _get_standard_state_correction_analytical(self, thermodynamic_state):
        """Return the standard state correction using the analytical method.

        Uses analytical approach from [1], but this approach is known to be inexact.
        This approach breaks down when the equilibrium restraint angles are near the
        limits of their domains and when equilibrium distance is near 0.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT).

        """
        # Check if all parameters are defined.
        self._check_parameters_defined()

        # Shortcuts variables.
        pi = np.pi
        kT = thermodynamic_state.kT
        p = self  # For the parameters.

        # Eq 32 of Ref [1]. Multiply by unit.radian**5 to remove the
        # expected unit value radians is a soft unit in this case, it
        # cancels in the math, but not in the equations above.
        DeltaG = -np.log(
            (8. * pi ** 2 * V0) / (p.r_aA0 ** 2 * unit.sin(p.theta_A0) * unit.sin(p.theta_B0)) *
            unit.sqrt(p.K_r * p.K_thetaA * p.K_thetaB * p.K_phiA * p.K_phiB * p.K_phiC) / (2 * pi * kT) ** 3 *
            unit.radian**5
        )

        return DeltaG

    def _get_standard_state_correction_numerical(self, thermodynamic_state):
        """Return the standard state correction using the numerical method.

        Uses numerical integral to the partition function contributions for
        r and theta, analytical for phi

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT).

        """
        # Check if all parameters are defined.
        self._check_parameters_defined()

        def strip(passed_unit):
            """Cast the passed_unit into md unit system for integrand lambda functions"""
            return passed_unit.value_in_unit_system(unit.md_unit_system)

        # Shortcuts variables.
        pi = np.pi
        kT = thermodynamic_state.kT
        p = self  # For the parameters.

        # Radial
        sigma = 1 / unit.sqrt(p.K_r / kT)
        rmin = min(0*unit.angstrom, p.r_aA0 - 8 * sigma)
        rmax = p.r_aA0 + 8 * sigma
        I = lambda r: r ** 2 * np.exp(-strip(p.K_r) / (2 * strip(kT)) * (r - strip(p.r_aA0)) ** 2)
        DGIntegral, dDGIntegral = scipy.integrate.quad(I, strip(rmin), strip(rmax)) * unit.nanometer**3
        ExpDeltaG = DGIntegral

        # Angular
        for name in ['A', 'B']:
            theta0 = getattr(p, 'theta_' + name + '0')
            K_theta = getattr(p, 'K_theta' + name)
            I = lambda theta: np.sin(theta) * np.exp(-strip(K_theta) / (2 * strip(kT)) * (theta - strip(theta0)) ** 2)
            DGIntegral, dDGIntegral = scipy.integrate.quad(I, 0, pi)
            ExpDeltaG *= DGIntegral

        # Torsion
        for name in ['A', 'B', 'C']:
            phi0 = getattr(p, 'phi_' + name + '0')
            K_phi = getattr(p, 'K_phi' + name)
            kshort = strip(K_phi/kT)
            ExpDeltaG *= math.sqrt(pi/2.0) * (
                math.erf((strip(phi0)+pi)*unit.sqrt(kshort)/math.sqrt(2)) -
                math.erf((strip(phi0)-pi)*unit.sqrt(kshort)/math.sqrt(2))
            ) / unit.sqrt(kshort)

        DeltaG = -np.log(8 * pi**2 * V0 / ExpDeltaG)
        return DeltaG

    @staticmethod
    def _is_collinear(positions, atoms, threshold=0.9):
        """Report whether any sequential vectors in a sequence of atoms are collinear.

        Parameters
        ----------
        positions : n_atoms x 3 simtk.unit.Quantity
            Reference positions to use for imposing restraints (units of length).
        atoms : iterable of int
            The indices of the atoms to test.
        threshold : float, optional, default=0.9
            Atoms are not collinear if their sequential vector separation dot
            products are less than ``threshold``.

        Returns
        -------
        result : bool
            Returns True if any sequential pair of vectors is collinear; False otherwise.

        """
        result = False
        for i in range(len(atoms)-2):
            v1 = positions[atoms[i+1], :] - positions[atoms[i], :]
            v2 = positions[atoms[i+2], :] - positions[atoms[i+1], :]
            normalized_inner_product = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
            result = result or (normalized_inner_product > threshold)

        return result

    def _pick_restrained_atoms(self, sampler_state, topography):
        """Select atoms to be used in restraint.

        Parameters
        ----------
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topography : yank.Topography, optional
            The topography with labeled receptor and ligand atoms.

        Returns
        -------
        restrained_atoms : list of int
            List of six atom indices used in the restraint.
            restrained_atoms[0:3] belong to the receptor,
            restrained_atoms[4:6] belong to the ligand.

        Notes
        -----
        The current algorithm simply selects random subsets of receptor
        and ligand atoms and rejects those that are too close to collinear.
        Future updates can further refine this algorithm.

        """
        # If receptor and ligand atoms are explicitly provided, use those.
        heavy_ligand_atoms = self.restrained_ligand_atoms
        heavy_receptor_atoms = self.restrained_receptor_atoms

        # Otherwise we restrain only heavy atoms.
        heavy_atoms = set(topography.topology.select('not element H').tolist())
        # Intersect heavy atoms with receptor/ligand atoms (s1&s2 is intersect).

        atom_selector = _AtomSelector(topography)

        heavy_ligand_atoms = atom_selector.compute_atom_intersect(heavy_ligand_atoms, 'ligand_atoms', heavy_atoms)
        heavy_receptor_atoms = atom_selector.compute_atom_intersect(heavy_receptor_atoms, 'receptor_atoms', heavy_atoms)

        if len(heavy_receptor_atoms) < 3 or len(heavy_ligand_atoms) < 3:
            raise ValueError('There must be at least three heavy atoms in receptor_atoms '
                             '(# heavy {}) and ligand_atoms (# heavy {}).'.format(
                                     len(heavy_receptor_atoms), len(heavy_ligand_atoms)))

        # If r3 or l1 atoms are given. We have to pick those.
        if isinstance(heavy_receptor_atoms, list):
            r3_atoms = [heavy_receptor_atoms[2]]
        else:
            r3_atoms = heavy_receptor_atoms
        if isinstance(heavy_ligand_atoms, list):
            l1_atoms = [heavy_ligand_atoms[0]]
        else:
            l1_atoms = heavy_ligand_atoms
        # TODO: Cast itertools generator to np array more efficiently
        r3_l1_pairs = np.array(list(itertools.product(r3_atoms, l1_atoms)))

        # Filter r3-l1 pairs that are too close/far away for the distance constraint.
        max_distance = 4 * unit.angstrom/unit.nanometer
        min_distance = 1 * unit.angstrom/unit.nanometer
        t = md.Trajectory(sampler_state.positions / unit.nanometers, topography.topology)
        distances = md.geometry.compute_distances(t, r3_l1_pairs)[0]
        indices_of_in_range_pairs = np.where(np.logical_and(distances > min_distance, distances <= max_distance))[0]

        if len(indices_of_in_range_pairs) == 0:
            error_msg = ('There are no heavy ligand atoms within the range of [{},{}] nm heavy receptor atoms!\n'
                         'Please Check your input files or try another restraint class')
            raise ValueError(error_msg.format(min_distance, max_distance))
        r3_l1_pairs = r3_l1_pairs[indices_of_in_range_pairs].tolist()

        # Iterate until we have found a set of non-collinear atoms.
        accepted = False
        while not accepted:
            # Select a receptor/ligand atom in range of each other for the distance constraint.
            r3_l1_atoms = random.sample(r3_l1_pairs, 1)[0]
            r3_l1_atoms_set = set(r3_l1_atoms)

            # Determine remaining receptor/ligand atoms.
            if isinstance(heavy_receptor_atoms, list):
                r1_r2_atoms = heavy_receptor_atoms[:2]
            else:
                r1_r2_atoms = random.sample(heavy_receptor_atoms - r3_l1_atoms_set, 2)
            if isinstance(heavy_ligand_atoms, list):
                l2_l3_atoms = heavy_ligand_atoms[1:]
            else:
                l2_l3_atoms = random.sample(heavy_ligand_atoms - r3_l1_atoms_set, 2)

            # Reject collinear sets of atoms.
            restrained_atoms = r1_r2_atoms + r3_l1_atoms + l2_l3_atoms
            accepted = not self._is_collinear(sampler_state.positions, restrained_atoms)

        logger.debug('Selected atoms to restrain: {}'.format(restrained_atoms))
        return restrained_atoms

    def _determine_restraint_parameters(self, sampler_states, topography):
        """Determine restraint parameters.

        Currently, all equilibrium values are measured from the initial structure,
        while spring constants are set to 20 kcal/(mol A**2) or 20 kcal/(mol rad**2)
        as in [1].

        Future iterations of this feature will introduce the ability to extract
        equilibrium parameters and spring constants from a short simulation.

        References
        ----------
        [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839

        """
        # We determine automatically only the parameters that have been left undefined.
        def _assign_if_undefined(attr_name, attr_value):
            """Assign value to self.name only if it is None."""
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, attr_value)

        # Merge receptor and ligand atoms in a single array for easy manipulation.
        restrained_atoms = self.restrained_receptor_atoms + self.restrained_ligand_atoms

        # Set spring constants uniformly, as in Ref [1] Table 1 caption.
        _assign_if_undefined('K_r', 20.0 * unit.kilocalories_per_mole / unit.angstrom**2)
        for parameter_name in ['K_thetaA', 'K_thetaB', 'K_phiA', 'K_phiB', 'K_phiC']:
            _assign_if_undefined(parameter_name, 20.0 * unit.kilocalories_per_mole / unit.radian**2)

        # Measure equilibrium geometries from static reference structure
        t = md.Trajectory(sampler_states.positions / unit.nanometers, topography.topology)

        atom_pairs = [restrained_atoms[2:4]]
        distances = md.geometry.compute_distances(t, atom_pairs, periodic=False)
        _assign_if_undefined('r_aA0', distances[0][0] * unit.nanometers)

        atom_triplets = [restrained_atoms[i:(i+3)] for i in range(1, 3)]
        angles = md.geometry.compute_angles(t, atom_triplets, periodic=False)
        for parameter_name, angle in zip(['theta_A0', 'theta_B0'], angles[0]):
            _assign_if_undefined(parameter_name, angle * unit.radians)

        atom_quadruplets = [restrained_atoms[i:(i+4)] for i in range(3)]
        dihedrals = md.geometry.compute_dihedrals(t, atom_quadruplets, periodic=False)
        for parameter_name, angle in zip(['phi_A0', 'phi_B0', 'phi_C0'], dihedrals[0]):
            _assign_if_undefined(parameter_name, angle * unit.radians)

        # Write restraint parameters
        msg = 'restraint parameters:\n'
        for parameter_name, parameter_value in self._parameters.items():
            msg += '%24s : %s\n' % (parameter_name, parameter_value)
        logger.debug(msg)


# ==============================================================================
# Orientation-dependent receptor-ligand restraints.
# ==============================================================================

class RMSD(ReceptorLigandRestraint):
    """Impose RMSD restraint on protein-ligand system.

    This restrains both protein and ligand using a flat-bottom RMSD restraint of the form:

    ``E = lambda_restraints * step(RMSD-RMSD0) * (K/2)*(RMSD-RMSD0)^2``

    Parameters
    ----------
    restrained_receptor_atoms : iterable of int, str, or None; Optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        Any number of receptor atoms can be selected.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided.
        If no selection is given, all receptor atoms will be restrained.
        If an empty list is provided, no receptor atoms will be restrained.
        (default is None).
    restrained_ligand_atoms : iterable of int, str, or None; Optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        Any number of ligand atoms can be selected
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided
        If no selection is given, all ligand atoms will be restrained.
        If an empty list is provided, no receptor atoms will be restrained.
        (default is None).
    K_RMSD : simtk.unit.Quantity, optional, default=0.6*kilocalories_per_mole/angstrom**2
        The spring constant (units compatible with kilocalories_per_mole/angstrom**2).
    RMSD0 : simtk.unit.Quantity, optional, default=2.0*angstrom
        The RMSD at which the restraint becomes nonzero.
    reference_state : openmmtools.states.SamplerState or None, Optional
        Reference sampler state with coordinates to use as the structure to align the RMSD to.
        The sampler state must have the same number of particles as the thermodynamic state you
        will apply this force to. This can temporarily be left undefined, but
        ``determine_missing_parameters()`` must be called before using the Restraint object.


    Attributes
    ----------
    restrained_receptor_atoms : list of int
        The indices of the restrained receptor atoms
    restrained_ligand_atoms : list of int
        The indices of the restrained_ligand_atoms

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 298*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    Create a restraint

    >>> restraint = RMSD(restrained_receptor_atoms=[1335, 1339, 1397],
    ...                  restrained_ligand_atoms=[2609, 2607, 2606],
    ...                  K_RMSD=1.0*unit.kilocalories_per_mole/unit.angstrom**2,
    ...                  RMSD0=1*unit.angstroms)

    Find missing parameters

    >>> restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_RMSD=None, RMSD0=None,
                 reference_state=None):
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.restrained_ligand_atoms = restrained_ligand_atoms
        self.K_RMSD = K_RMSD
        self.RMSD0 = RMSD0
        self.reference_state = reference_state

    # -------------------------------------------------------------------------
    # Public properties.
    # -------------------------------------------------------------------------

    class _RMSDRestrainedAtomsProperty(_RestrainedAtomsProperty):
        """
        Descriptor of restrained atoms.

        Extends `_RestrainedAtomsProperty` to handle single integers and strings.

        Extension allows individual atom lists to be defined as empty through the ``allowed_empty`` boolean
        """

        _MUST_COMPUTE_STRING = ('You are specifying {} {} atoms, '
                                'the final atoms will be chosen at from this set but you MUST '
                                'run "determine_missing_parameters"')

        def __init__(self, atoms_type, allowed_empty=False):
            self._allowed_empty = allowed_empty
            super().__init__(atoms_type)

        @methoddispatch
        def _validate_atoms(self, restrained_atoms):
            restrained_atoms = super()._validate_atoms(restrained_atoms)
            # TODO: Determine the minimum number of atoms needed for this restraint (can it be 0?)
            if len(restrained_atoms) < 3 and not (len(restrained_atoms) == 0 and self._allowed_empty):
                raise ValueError('At least three {} atoms are required to impose an '
                                 'RMSD restraint.'.format(self._atoms_type))
            return restrained_atoms

        @_validate_atoms.register(str)
        def _cast_atom_string(self, restrained_atoms):
            logger.warning(self._MUST_COMPUTE_STRING.format("a string for", self._atoms_type))
            return restrained_atoms

    restrained_receptor_atoms = _RMSDRestrainedAtomsProperty('receptor', allowed_empty=True)
    restrained_ligand_atoms = _RMSDRestrainedAtomsProperty('ligand')

    # -------------------------------------------------------------------------
    # Public methods.
    # -------------------------------------------------------------------------

    def restrain_state(self, thermodynamic_state):
        """Add the restraint force to the state's ``System``.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.

        """

        # Check if all parameters are defined.
        self._check_parameters_defined()

        # Merge receptor and ligand atoms in a single array for easy manipulation.
        restrained_atoms = self.restrained_receptor_atoms + self.restrained_ligand_atoms

        # Create RMSDForce CV for all restrained atoms
        rmsd_cv = openmm.RMSDForce(self.reference_state.positions, restrained_atoms)

        # Create an CustomCVForce
        energy_expression = 'lambda_restraints * step(dRMSD) * (K_RMSD/2)*dRMSD^2; dRMSD = (RMSD-RMSD0);'
        energy_expression += 'K_RMSD = %f;' % self.K_RMSD.value_in_unit_system(unit.md_unit_system)
        energy_expression += 'RMSD0 = %f;' % self.RMSD0.value_in_unit_system(unit.md_unit_system)
        restraint_force = openmm.CustomCVForce(energy_expression)
        restraint_force.addCollectiveVariable('RMSD', rmsd_cv)
        restraint_force.addGlobalParameter('lambda_restraints', 1.0)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        self._add_force_in_separate_group(system, restraint_force)
        thermodynamic_state.system = system

    def get_standard_state_correction(self, thermodynamic_state):
        """Return the standard state correction.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT).

        """
        return self._get_standard_state_correction_numerical(thermodynamic_state)
        # TODO: Is this required, or is there only one way to do it?
        # if self.standard_state_correction_method == 'analytical' and False:
        #     # This code should not be reachable on purpose until we figure out what the SSC is
        #     return self._get_standard_state_correction_analytical(thermodynamic_state)
        # else:  # The property checks that the value is known in the setter.
        #     return self._get_standard_state_correction_numerical(thermodynamic_state)

    def determine_missing_parameters(self, thermodynamic_state, sampler_state, topography):
        """Set reference positions for RMSD restraint.

        Future iterations of this feature will introduce the ability to extract
        equilibrium parameters and spring constants from a short simulation.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms to be used as reference
        topography : yank.Topography, optional
            The topography with labeled receptor and ligand atoms.

        """

        # Pick out the restrained atoms
        if not self._are_restrained_atoms_defined:
            self._pick_restrained_atoms(topography)

        # We determine automatically only the parameters that have been left undefined.
        def _assign_if_undefined(attr_name, attr_value):
            """Assign value to self.name only if it is None."""
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, attr_value)

        # Set spring constants uniformly, as in Ref [1] Table 1 caption.
        _assign_if_undefined('K_RMSD', 0.6 * unit.kilocalories_per_mole / unit.angstrom**2)
        _assign_if_undefined('RMSD0', 2.0 * unit.angstroms)

        # Write restraint parameters
        msg = 'restraint parameters:\n'
        for parameter_name, parameter_value in self._parameters.items():
            msg += '{0:24s} : {1}\n'.format(parameter_name, parameter_value)
        logger.debug(msg)

        # Assign the sampler state
        _assign_if_undefined('reference_state', sampler_state)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _check_parameters_defined(self):
        """Raise an exception there are still parameters undefined."""
        if not self._are_restrained_atoms_defined:
            raise RestraintParameterError('Undefined restrained atoms. Please run `determine_missing_parameters`')
        if None in [self.K_RMSD, self.RMSD0]:
            raise RestraintParameterError("Undefined RMSD parameters. Please run `determine_missing_parameters`")
        if self.reference_state is None:
            raise RestraintParameterError("No reference configuration as defined, "
                                          "Please run `determine_missing_parameters`")

    @property
    def _are_restrained_atoms_defined(self):
        """Check if the restrained atoms are defined well enough to make a restraint"""
        for atoms in [self.restrained_receptor_atoms, self.restrained_ligand_atoms]:
            # Atoms should be a list or None at this point due to the _RestrainedAtomsProperty class
            if not self._are_single_atoms_defined(atoms):
                return False
        return True

    @staticmethod
    def _are_single_atoms_defined(atom_list):
        """Check that a set of atoms matches the specific format we expect"""
        if isinstance(atom_list, list):
            return True
        return False

    def _get_standard_state_correction_numerical(self, thermodynamic_state):
        """Return the standard state correction using the numerical method.

        Uses numerical integral to the partition function contributions for
        r and theta, analytical for phi

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT).

        """
        # TODO: Compute standard state correction
        return 0.0

    def _pick_restrained_atoms(self, topography):
        """Select the restrained atoms to use for this system"""
        atom_selector = _AtomSelector(topography)
        for atom_word, top_key in zip(["restrained_ligand_atoms", "restrained_receptor_atoms"],
                                      ["ligand_atoms",            "receptor_atoms"]):
            atoms = getattr(self, atom_word)
            if self._are_single_atoms_defined(atoms):
                continue
            defined_atoms = atom_selector.compute_atom_intersect(atoms, top_key)
            setattr(self, atom_word, defined_atoms)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(Harmonic, globals())
