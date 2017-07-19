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
import itertools

import numpy as np
import scipy.integrate
import mdtraj as md
import openmmtools as mmtools
from simtk import openmm, unit

from . import pipeline

logger = logging.getLogger(__name__)


# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

V0 = 1660.53928 * unit.angstroms**3  # standard state volume


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class RestraintStateError(mmtools.states.ComposableStateError):
    """Error raised by an RestraintState."""
    pass


class RestraintParameterError(Exception):
    """Error raised by a ReceptorLigandRestraint."""
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
        restraint_classes[name] is the class corresponding to `name`

    """
    # Get a list of all subclasses of ReceptorLigandRestraint
    def get_all_subclasses(check_cls):
        """Find all subclasses of a given class recursively."""
        all_subclasses = []

        for subclass in check_cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    # Build an index of all names, ensuring there are no name collisions.
    available_restraints = dict()
    for cls in get_all_subclasses(ReceptorLigandRestraint):
        classname = cls.__name__
        if inspect.isabstract(cls):
            # Skip abstract base classes
            pass
        elif classname in available_restraints:
            raise ValueError("More than one restraint subclass has the name '{}'.".format(classname))
        else:
            available_restraints[classname] = cls

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
    return available_restraints.keys()


def create_restraint(restraint_type, **kwargs):
    """Factory of receptor-ligand restraint objects.

    Parameters
    ----------
    restraint_type : str
        Restraint type name matching a register (imported) subclass of ReceptorLigandRestraint.
    kwargs
        Parameters to pass to the restraint constructor.

    """
    available_restraints = available_restraint_classes()
    if restraint_type not in available_restraints:
        raise ValueError("Restraint type {} unknown. Options are: {}".format(
            restraint_type, str(available_restraints.keys())))
    cls = available_restraints[restraint_type]
    return cls(**kwargs)


# ==============================================================================
# ComposableState class to control the strength of restraints.
# ==============================================================================

class RestraintState(object):
    """
    The state of a restraint.

    A `ComposableState` controlling the strength of a restraint
    through its `lambda_restraints` property.

    Parameters
    ----------
    lambda_restraints : float
        The strength of the restraint. Must be between 0 and 1.

    Attributes
    ----------
    lambda_restraints

    Methods
    -------
    apply_to_system
    check_system_consistency
    apply_to_context

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

    Create a `RestraintState` object to control the strength of the restraint.

    >>> restraint_state = RestraintState(lambda_restraints=1.0)

    `RestraintState` implements the IComposableState interface, so it can be
    used with `CompoundThermodynamicState`.

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
        self._lambda_restraints = lambda_restraints

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
            If the system does not have any `CustomForce` with a
            `lambda_restraint` global parameter.

        """
        # Set lambda_restraints in all forces that have it.
        for force, parameter_id in self._get_system_forces_parameters(system):
            force.setGlobalParameterDefaultValue(parameter_id, self._lambda_restraints)

    def check_system_consistency(self, system):
        """
        Check if the system's restraint is in this restraint state.

        It raises a RestraintStateError if the restraint is not consistent
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

    @classmethod
    def _standardize_system(cls, system):
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
        for force, parameter_id in cls._get_system_forces_parameters(system):
            force.setGlobalParameterDefaultValue(parameter_id, 1.0)

    @staticmethod
    def _get_system_forces_parameters(system):
        """Yields the system's forces having a lambda_restraints parameter.

        Yields
        ------
        A tuple force, parameter_index for each force with lambda_restraints.

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

        # Raise error if the system doesn't have a restraint.
        if found_restraint is False:
            raise RestraintStateError('The system does not have a restraint.')

    def __getstate__(self):
        return dict(lambda_restraints=self._lambda_restraints)

    def __setstate__(self, serialization):
        self.lambda_restraints = serialization['lambda_restraints']


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
    'lambda_restraints'. You can easily control this variable through the
    `RestraintState` object.

    Notes
    -----
    Creating a subclass requires the following:

        1. Implement a constructor. Optionally this can leave all or a subset of
        the restraint parameters undefined. In this case, you need to provide
        an implementation of `determine_missing_parameters()`.

        2. Implement `restrain_state()` that add the restrain `Force` to the state's
        `System`.

        3. Implement `get_standard_state_correction()` to return standard state correction.

        4. Optionally, implement `determine_missing_parameters()` to fill in
        the parameters left undefined in the constructor.

    Methods
    -------
    restrain_state
    get_standard_state_correction
    determine_missing_parameters

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

        Optionally, a ReceptorLigandRestraint can support the automatic
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


# ==============================================================================
# Base class for radially-symmetric receptor-ligand restraints.
# ==============================================================================

class RadiallySymmetricRestraint(ReceptorLigandRestraint):
    """
    Base class for radially-symmetric restraints between ligand and protein.

    This class is intended to be inherited by Restraints using a `CustomBondForce`.
    The restraint strength is controlled by a global context parameter called
    'lambda_restraints'.

    The class allows the restrained atoms to be temporarily, but in this case
    `determine_missing_parameters()` must be called before using the restraint.

    Parameters
    ----------
    restrained_receptor_atom : int, optional
        The index of the receptor atom to restrain. This can temporarily be left
        undefined, but in this case `determine_missing_parameters()` must be called
        before using the Restraint object (default is None).
    restrained_ligand_atom : int, optional
        The index of the ligand atom to restrain. This can temporarily be left
        undefined, but in this case `determine_missing_parameters()` must be called
        before using the Restraint object (default is None).

    Notes
    -----
    To create a subclass, follow these steps:

        1. Implement the property '_energy_function' with the energy function of choice.

        2. Implement the property '_bond_parameters' to return the `_energy_function`
        parameters as a dict parameter_name: parameter_value.

        3. Optionally, you can overwrite the `_determine_bond_parameters()` member
        function to automatically determine these parameters from the atoms positions.

    Methods
    -------
    restrain_state
    get_standard_state_correction
    determine_missing_parameters

    """
    def __init__(self, restrained_receptor_atom=None, restrained_ligand_atom=None):
        self.restrained_receptor_atom = restrained_receptor_atom
        self.restrained_ligand_atom = restrained_ligand_atom

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
        if self.restrained_receptor_atom is None or self.restrained_ligand_atom is None:
            raise RestraintParameterError('Restraint {}: Undefined restrained '
                                          'atoms.'.format(self.__class__.__name__))

        # Create restraint force.
        restraint_force = self._create_restraint_force(self.restrained_receptor_atom,
                                                       self.restrained_ligand_atom)

        # Set periodic conditions on the force if necessary.
        restraint_force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        system.addForce(restraint_force)
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

        r_min = 0 * unit.nanometers
        r_max = 100 * unit.nanometers  # TODO: Use maximum distance between atoms?

        # Create a System object containing two particles connected by the reference force
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)
        system.addParticle(1.0 * unit.amu)
        force = self._create_restraint_force(0, 1)
        # Disable the PBC if on for this approximation of the analytical solution
        force.setUsesPeriodicBoundaryConditions(False)
        system.addForce(force)

        # Create a Reference context to evaluate energies on the CPU.
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, integrator, platform)

        # Set default positions.
        positions = unit.Quantity(np.zeros([2,3]), unit.nanometers)
        context.setPositions(positions)

        # Create a function to compute integrand as a function of interparticle separation.
        beta = thermodynamic_state.beta

        def integrand(r):
            """
            Parameters
            ----------
            r : float
                Inter-particle separation in nanometers

            Returns
            -------
            dI : float
               Contribution to integrand (in nm^2).

            """
            positions[1, 0] = r * unit.nanometers
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            potential = state.getPotentialEnergy()
            dI = 4.0 * math.pi * r**2 * math.exp(-beta * potential)
            return dI

        # Integrate shell volume.
        shell_volume, shell_volume_error = scipy.integrate.quad(lambda r: integrand(r), r_min / unit.nanometers,
                                                                r_max / unit.nanometers) * unit.nanometers**3
        logger.debug("shell_volume = %f nm^3" % (shell_volume / unit.nanometers**3))

        # Compute standard-state volume for a single molecule in a box of size (1 L) / (avogadros number)
        # Should also generate constant V0
        liter = 1000.0 * unit.centimeters**3  # one liter
        box_volume = liter / (unit.AVOGADRO_CONSTANT_NA*unit.mole) # standard state volume
        logger.debug("box_volume = %f nm^3" % (box_volume / unit.nanometers**3))

        # Compute standard state correction for releasing shell restraints into standard-state box (in units of kT).
        DeltaG = - math.log(box_volume / shell_volume)
        logger.debug("Standard state correction: %.3f kT" % DeltaG)

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
        self._determine_bond_parameters(thermodynamic_state, sampler_state, topography)

    # -------------------------------------------------------------------------
    # Internal-usage: properties and methods for subclasses.
    # -------------------------------------------------------------------------

    @abc.abstractproperty
    def _energy_function(self):
        """str: energy expression of the restraint force.

        This must be implemented by the inheriting class.

        """
        pass

    @abc.abstractproperty
    def _bond_parameters(self):
        """dict: the bond parameters of the restraint force.

        This is a dictionary parameter_name: parameter_value that
        will be used to configure the `CustomBondForce` added to
        the `System`.

        If there are parameters undefined, this must be None.

        """
        pass

    def _determine_bond_parameters(self, thermodynamic_state, sampler_state, topography):
        """Determine the missing bond parameters.

        Optionally, a subclass can implement this method to automatically
        define the bond parameters of the restraints from the information
        in the given states and topography. The default implementation just
        raises a NotImplemented error if `_bond_parameters` are undefined.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor and ligand atoms.

        """
        # Raise exception only if the subclass doesn't already defines parameters.
        if self._bond_parameters is None:
            raise NotImplementedError('Restraint {} cannot automatically determine '
                                      'bond parameters.'.format(self.__class__.__name__))

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

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
            The topography with labeled receptor and ligand atoms.

        """
        debug_msg = ('Restraint {}: Automatically picked restrained '
                     'receptor atom: {{}}'.format(self.__class__.__name__))

        # Shortcuts
        positions = sampler_state.positions
        ligand_atoms = topography.ligand_atoms
        receptor_atoms = topography.receptor_atoms

        # Automatically determine receptor atom to restrain only
        # if the user has not defined specific atoms to restrain.
        if self.restrained_receptor_atom is None:
            self.restrained_receptor_atom = self._closest_atom_to_centroid(positions, receptor_atoms)
            logger.debug(debug_msg.format(self.restrained_receptor_atom))

        # Same for ligand atom.
        if self.restrained_ligand_atom is None:
            self.restrained_ligand_atom = self._closest_atom_to_centroid(positions, ligand_atoms)
            logger.debug(debug_msg.format(self.restrained_ligand_atom))

    def _create_restraint_force(self, particle1, particle2):
        """Create a new restraint force between specified atoms.

        Parameters
        ----------
        particle1 : int
            Index of first atom in restraint
        particle2 : int
            Index of second atom in restraint

        Returns
        -------
        force : simtk.openmm.CustomBondForce
           The created restraint force.

        """
        # Check if parameters have been defined.
        if self._bond_parameters is None:
            err_msg = 'Restraint {}: Undefined bond parameters.'.format(self.__class__.__name__)
            raise RestraintParameterError(err_msg)

        # Unzip bond parameters names and values from dict.
        parameter_names, parameter_values = zip(*self._bond_parameters.items())

        # Create bond force and lambda_restraints parameter to control it.
        force = openmm.CustomBondForce('lambda_restraints * ' + self._energy_function)
        force.addGlobalParameter('lambda_restraints', 1.0)

        # Add all parameters.
        for parameter in parameter_names:
            force.addPerBondParameter(parameter)

        # Create restraining bond.
        try:
            force.addBond(particle1, particle2, parameter_values)
        except Exception:
            err_msg = 'particle1: {}\nparticle2: {}\nbond_parameters: {}'.format(
                particle1, particle2, self._bond_parameters)
            logger.error(err_msg)
            raise

        return force

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

    The energy expression of the restraint is given by

       ``E = lambda_restraints * (K/2)*r^2``

    where `K` is the spring constant, `r` is the distance between the
    restrained atoms, and `lambda_restraints` is a scale factor that
    can be used to control the strength of the restraint. You can control
    `lambda_restraints` through `RestraintState` class.

    The class supports automatic determination of the parameters left undefined
    in the constructor through `determine_missing_parameters()`.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole (default is None).
    restrained_receptor_atom : int, optional
        The index of the receptor atom to restrain (default is None).
    restrained_ligand_atom : int, optional
        The index of the ligand atom to restrain (default is None).

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
    ...                      restrained_receptor_atom=1644, restrained_ligand_atom=2609)

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
    def _energy_function(self):
        """str: energy expression of the restraint force."""
        return '(K/2)*r^2'

    @property
    def _bond_parameters(self):
        """dict: the bond parameters of the restraint force.

        If there are parameters undefined, this is None.

        """
        if self.spring_constant is None:
            return None
        return {'K': self.spring_constant}

    def _determine_bond_parameters(self, thermodynamic_state, sampler_state, topography):
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
        logger.debug("Spring Constant Sigma, s = {:.3f} nm".format(sigma / unit.nanometers))

        # Compute corresponding spring constant.
        self.spring_constant = thermodynamic_state.kT / sigma**2


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

    More precisely, the energy expression of the restraint is given by

        ``E = lambda_restraints * step(r-r0) * (K/2)*(r-r0)^2``

    where `K` is the spring constant, `r` is the distance between the
    restrained atoms, `r0` is another parameter defining the distance
    at which the harmonic restraint is imposed, and `lambda_restraints`
    is a scale factor that can be used to control the strength of the
    restraint. You can control `lambda_restraints` through the class
    `RestraintState`.

    The class supports automatic determination of the parameters left undefined
    in the constructor through `determine_missing_parameters()`.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole (default is None).
    well_radius : simtk.unit.Quantity, optional
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance (default is None).
    restrained_receptor_atom : int, optional
        The index of the receptor atom to restrain (default is None).
    restrained_ligand_atom : int, optional
        The index of the ligand atom to restrain (default is None).

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
    ...                        well_radius=5.2*unit.nanometers, restrained_receptor_atom=1644,
    ...                        restrained_ligand_atom=2609)

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
        return 'step(r-r0) * (K/2)*(r-r0)^2'

    @property
    def _bond_parameters(self):
        """dict: the bond parameters of the restraint force.

        If there are parameters undefined, this is None.

        """
        if self.spring_constant is None or self.well_radius is None:
            return None
        return {'K': self.spring_constant, 'r0': self.well_radius}

    def _determine_bond_parameters(self, thermodynamic_state, sampler_state, topography):
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
        x_unit = sampler_state.positions.unit

        # Get dimensionless receptor positions.
        x = sampler_state.positions[topography.receptor_atoms, :] / x_unit

        # Determine number of atoms.
        n_atoms = x.shape[0]

        if n_atoms > 3:
            # Check that restrained receptor atom is in expected range.
            if self.restrained_receptor_atom > n_atoms:
                raise ValueError('Receptor atom {} was selected for restraint, but system '
                                 'only has {} atoms.'.format(self.restrained_receptor_atom, n_atoms))

            # Compute median absolute distance to central atom.
            # (Working in non-unit-bearing floats for speed.)
            xref = np.reshape(x[self.restrained_receptor_atom, :], (1, 3))  # (1,3) array
            # distances[i] is the distance from the centroid to particle i
            distances = np.sqrt(((x - np.tile(xref, (n_atoms, 1)))**2).sum(1))
            median_absolute_distance = np.median(abs(distances))

            # Convert back to unit-bearing quantity.
            median_absolute_distance *= x_unit

            # Convert to estimator of standard deviation for normal distribution.
            sigma = 1.4826 * median_absolute_distance

            # Calculate r0, which is a multiple of sigma plus 5 A.
            r0 = 2*sigma + 5.0 * unit.angstroms
        else:
            DEFAULT_DISTANCE = 15.0 * unit.angstroms
            logger.warning("Receptor only contains {} atoms; using default "
                           "distance of {}".format(n_atoms, str(DEFAULT_DISTANCE)))
            r0 = DEFAULT_DISTANCE

        logger.debug("restraint distance r0 = %.1f A" % (r0 / unit.angstroms))

        # Set spring constant/
        # K = (2.0 * 0.0083144621 * 5.0 * 298.0 * 100) * unit.kilojoules_per_mole/unit.nanometers**2
        K = 0.6 * unit.kilocalories_per_mole / unit.angstroms**2
        logger.debug("K = %.1f kcal/mol/A^2" % (K / (unit.kilocalories_per_mole / unit.angstroms**2)))

        # Store parameters if not already defined.
        if self.spring_constant is None:
            self.spring_constant = K
        if self.well_radius is None:
            self.well_radius = r0


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

        r1, r2, r3: the coordinates of the 3 receptor atoms.

        l1, l2, l3: the coordinates of the 3 ligand atoms.

        K_r: the spring constant for the restrained distance ``|r3 - l1|``.

        r_aA0: the equilibrium distance of ``|r3 - l1|``.

        K_thetaA, K_thetaB: the spring constants for ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        theta_A0, theta_B0: the equilibrium angles of ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        K_phiA, K_phiB, K_phiC: the spring constants for ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        phi_A0, phi_B0, phi_C0: the equilibrium torsion of ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        lambda_restraints: a scale factor that can be used to control the strength
        of the restraint.

    You can control ``lambda_restraints`` through the class ``RestraintState``.

    The class supports automatic determination of the parameters left undefined
    in the constructor through ``determine_missing_parameters()``.

    Parameters
    ----------
    restrained_atoms : iterable of int, optional
        The indices of the atoms to restrain, in order r1, r2, r3, l1, l2, l3.
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

    WARNING
    -------
    Symmetry corrections for symmetric ligands are not automatically applied.
    See Ref [1] and [2] for more information on correcting for ligand symmetry.

    Attributes
    ----------
    restrained_atoms
    standard_state_correction_method

    Methods
    -------
    restrain_state
    get_standard_state_correction
    determine_missing_parameters

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

    >>> restraint = Boresch(restrained_atoms=[1335, 1339, 1397, 2609, 2607, 2606],
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
    def __init__(self, restrained_atoms=None,
                 K_r=None, r_aA0=None,
                 K_thetaA=None, theta_A0=None,
                 K_thetaB=None, theta_B0=None,
                 K_phiA=None, phi_A0=None,
                 K_phiB=None, phi_B0=None,
                 K_phiC=None, phi_C0=None,
                 standard_state_correction_method='analytical'):
        self.restrained_atoms = restrained_atoms
        self.K_r = K_r
        self.r_aA0 = r_aA0
        self.K_thetaA, self.K_thetaB = K_thetaA, K_thetaB
        self.theta_A0, self.theta_B0 = theta_A0, theta_B0
        self.K_phiA, self.K_phiB, self.K_phiC = K_phiA, K_phiB, K_phiC
        self.phi_A0, self.phi_B0, self.phi_C0 = phi_A0, phi_B0, phi_C0
        self.standard_state_correction_method = standard_state_correction_method

    @property
    def restrained_atoms(self):
        """list of int: The indices of the six atoms to restrain."""
        return self._restrained_atoms

    @restrained_atoms.setter
    def restrained_atoms(self, value):
        if value is not None and len(value) != 6:
            raise ValueError('Six atoms are required to impose a Boresch-style restraint.')
        self._restrained_atoms = value

    @property
    def standard_state_correction_method(self):
        """The default method to use in `get_standard_state_correction`.

        This can be either 'analytical' or 'numerical'.

        """
        return self._standard_state_correction_method

    @standard_state_correction_method.setter
    def standard_state_correction_method(self, value):
        if value not in ['analytical', 'numerical']:
            raise ValueError("The standard state correction method must be one between "
                             "'analytical' and 'numerical', got {}.".format(value))
        self._standard_state_correction_method = value

    def restrain_state(self, thermodynamic_state):
        """Add the restraint force to the state's `System`.

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
        restraint_force.addBond(self.restrained_atoms, [])
        restraint_force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        system.addForce(restraint_force)
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
        if self.restrained_atoms is not None:
            self._determine_restraint_parameters(sampler_state, topography)
        else:
            # Keep selecting random retrained atoms until the parameters
            # make the standard state correction robust.
            for attempt in range(MAX_ATTEMPTS):
                logger.debug('Attempt {} / {} at automatically selecting atoms and '
                             'restraint parameters...'.format(attempt, MAX_ATTEMPTS))

                # Randomly pick non-collinear atoms.
                self.restrained_atoms = self._pick_restrained_atoms(sampler_state, topography)

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
    def _parameters(self):
        """dict: restraint parameters in dict forms."""
        parameter_names, _, _, _ = inspect.getargspec(self.__init__)

        # Exclude non-parameters arguments.
        for exclusion in ['self', 'restrained_atoms', 'standard_state_correction_method']:
            parameter_names.remove(exclusion)

        # Retrieve and store options.
        parameters = {parameter_name: getattr(self, parameter_name)
                      for parameter_name in parameter_names}
        return parameters

    def _check_parameters_defined(self):
        """Raise an exception there are still parameters undefined."""
        if self.restrained_atoms is None:
            raise RestraintParameterError('Undefined restrained atoms.')

        # Find undefined parameters and raise error.
        undefined_parameters = [name for name, value in self._parameters.items() if value is None]
        if len(undefined_parameters) > 0:
            err_msg = 'Undefined parameters for Boresch restraint: {}'.format(undefined_parameters)
            raise RestraintParameterError(err_msg)

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

        Uses numerical integral to the partition function contriubtions for
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
            products are less than `threshold`.

        Returns
        -------
        result : bool
            Returns True if any sequential pair of vectors is collinear; False otherwise.

        """
        result = False
        for i in range(len(atoms)-2):
            v1 = positions[atoms[i+1],:] - positions[atoms[i],:]
            v2 = positions[atoms[i+2],:] - positions[atoms[i+1],:]
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
        t = md.Trajectory(sampler_state.positions / unit.nanometers, topography.topology)

        # Determine heavy atoms. Using sets since lists should be unique anyways
        heavy_atoms = set(topography.topology.select('not element H'))

        # Intersect heavy atoms with receptor/ligand atoms (s1&s2 is intersect)
        heavy_ligand_atoms = set(topography.ligand_atoms) & heavy_atoms
        heavy_receptor_atoms = set(topography.receptor_atoms) & heavy_atoms
        if len(heavy_receptor_atoms) < 3 or len(heavy_ligand_atoms) < 3:
            raise ValueError('There must be at least three heavy atoms in receptor_atoms '
                             '(# heavy {}) and ligand_atoms (# heavy {}).'.format(
                                     len(heavy_receptor_atoms), len(heavy_ligand_atoms)))

        # Find valid pairs of ligand/receptor atoms within a cutoff
        max_distance = 4 * unit.angstrom/unit.nanometer
        min_distance = 1 * unit.angstrom/unit.nanometer
        # TODO: Cast itertools generator to np array more efficiently
        all_pairs = np.array(list(itertools.product(heavy_receptor_atoms, heavy_ligand_atoms)))
        distances = md.geometry.compute_distances(t, all_pairs)[0]
        index_of_in_range_atoms = np.where(np.logical_and(distances > min_distance, distances <= max_distance))[0]
        if len(index_of_in_range_atoms) == 0:
            error_msg = ('There are no heavy ligand atoms within the range of [{},{}] nm heavy receptor atoms!\n'
                         'Please Check your input files or try another restraint class')
            raise ValueError(error_msg.format(min_distance, max_distance))

        # Iterate until we have found a set of non-collinear atoms.
        accepted = False
        while not accepted:
            # Select a receptor/ligand atom in range of each other
            raA_atoms = all_pairs[random.sample(list(index_of_in_range_atoms), 1)[0]].tolist()
            # Cast to set for easy comparison operations
            raA_set = set(raA_atoms)
            # Select two additional random atoms from the receptor and ligand
            restrained_atoms = (random.sample(heavy_receptor_atoms-raA_set, 2) + raA_atoms +
                                random.sample(heavy_ligand_atoms-raA_set, 2))
            # Reject collinear sets of atoms.
            accepted = not self._is_collinear(sampler_state.positions, restrained_atoms)

        # Cast to Python ints to avoid type issues when passing to OpenMM
        restrained_atoms = [int(i) for i in restrained_atoms]
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
        # We determine automatically determine only the
        # parameters that have been left undefined.
        def _assign_if_undefined(attr_name, attr_value):
            """Assign value to self.name only if it is None."""
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, attr_value)

        # Set spring constants uniformly, as in Ref [1] Table 1 caption.
        _assign_if_undefined('K_r', 20.0 * unit.kilocalories_per_mole / unit.angstrom**2)
        for parameter_name in ['K_thetaA', 'K_thetaB', 'K_phiA', 'K_phiB', 'K_phiC']:
            _assign_if_undefined(parameter_name, 20.0 * unit.kilocalories_per_mole / unit.radian**2)

        # Measure equilibrium geometries from static reference structure
        t = md.Trajectory(sampler_states.positions / unit.nanometers, topography.topology)

        distances = md.geometry.compute_distances(t, [self.restrained_atoms[2:4]], periodic=False)
        _assign_if_undefined('r_aA0', distances[0][0] * unit.nanometers)

        angles = md.geometry.compute_angles(t, [self.restrained_atoms[i:(i+3)] for i in range(1, 3)], periodic=False)
        for parameter_name, angle in zip(['theta_A0', 'theta_B0'], angles[0]):
            _assign_if_undefined(parameter_name, angle * unit.radians)

        dihedrals = md.geometry.compute_dihedrals(t, [self.restrained_atoms[i:(i+4)] for i in range(3)], periodic=False)
        for parameter_name, angle in zip(['phi_A0', 'phi_B0', 'phi_C0'], dihedrals[0]):
            _assign_if_undefined(parameter_name, angle * unit.radians)

        # Write restraint parameters
        msg = 'restraint parameters:\n'
        for parameter_name, parameter_value in self._parameters.items():
            msg += '%24s : %s\n' % (parameter_name, parameter_value)
        logger.debug(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(Harmonic, globals())
