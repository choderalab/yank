#!/usr/local/bin/env python

# ==============================================================================
# FILE DOCSTRING
# ==============================================================================

"""

Automated selection and imposition of receptor-ligand restraints for absolute alchemical binding
free energy calculations, along with computation of the standard state correction.

"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import abc
import copy
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

logger = logging.getLogger(__name__)


# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

V0 = 1660.53928 * unit.angstroms**3  # standard state volume


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class RestraintParameterError(Exception):
    pass


# ==============================================================================
# Dispatch appropriate restraint type from registered restraint classes
# ==============================================================================

def available_restraint_classes():
    """
    Return all available restraint classes.

    Returns
    -------
    restraint_classes : dict of str : class
        restraint_classes[name] is the class corresponding to `name`

    """
    # Get a list of all subclasses of ReceptorLigandRestraint
    def get_all_subclasses(cls):
        """Find all subclasses of a given class recursively."""
        all_subclasses = []

        for subclass in cls.__subclasses__():
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
            raise ValueError("More than one restraint subclass has the name '%s'." % classname)
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


def create_restraints(restraint_type, topology, state, system, positions, receptor_atoms, ligand_atoms):
    """
    Initialize a receptor-ligand restraint class matching the specified restraint type name.

    Parameters
    ----------
    restraint_type : str
        Restraint type name matching a register (imported) subclass of ReceptorLigandRestraint.
    topology : openmm.app.Topology
        Reference topology for the complex
    system : openmm.System
        System containing all the forces and atoms that restraint will be added to
    state : thermodynamics.ThermodynamicState
        The thermodynamic state specifying temperature, pressure, etc. to which restraints are to be added
    positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
        Reference positions to use for imposing restraints
    receptor_atoms : list of int
        A complete list of receptor atoms
    ligand_atoms : list of int
        A complete list of ligand atoms

    """
    available_restraints = available_restraint_classes()
    if not (restraint_type in available_restraints):
        raise ValueError("Restraint type '%s' unknown. Options are: %s" % (restraint_type, str(available_restraints.keys())))
    cls = available_restraints[restraint_type]
    return cls(topology, state, system, positions, receptor_atoms, ligand_atoms)


# =============================================================================================
# Base class for receptor-ligand restraints.
# =============================================================================================

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3


class ReceptorLigandRestraint(ABC):
    """
    Impose a single restraint between ligand and protein to prevent ligand from drifting too far
    from protein in implicit solvent calculations.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    Notes
    -----
    Creating a subclass requires the following:
    * Perform any necessary processing in subclass `__init__` after calling base class `__init__`
    * Override getRestraintForce() to return a new `Force` instance imposing the restraint
    * Override getStandardStateCorrection() to return standard state correction

    """
    def __init__(self, topology, state, system, positions, receptor_atoms, ligand_atoms):
        """
        Initialize a receptor-ligand restraint class.

        Parameters
        ----------
        topology : openmm.app.Topology
            Reference topology for the complex
        state : thermodynamics.ThermodynamicState
           The thermodynamic state specifying temperature, pressure, etc. to which restraints are to be added
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
           Reference positions to use for imposing restraints
        receptor_atoms : list of int
            A complete list of receptor atoms
        ligand_atoms : list of int
            A complete list of ligand atoms

        """
        self._topology = topology
        self._state = state
        self._system = system
        self._positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)
        self._receptor_atoms = list(receptor_atoms)
        self._ligand_atoms = list(ligand_atoms)

        # Perform some sanity checks.
        natoms = system.getNumParticles()
        if np.any(np.array(receptor_atoms) >= natoms):
            raise ValueError("Receptor atom indices fall outside [0,natoms), where natoms = %d.  receptor_atoms = %s" % (natoms, str(receptor_atoms)))
        if np.any(np.array(ligand_atoms) >= natoms):
            raise ValueError("Ligand atom indices fall outside [0,natoms), where natoms = %d.  ligand_atoms = %s" % (natoms, str(ligand_atoms)))

    @property
    def temperature(self):
        return self._state.temperature

    @property
    def kT(self):
        return kB * self.temperature

    @property
    def beta(self):
        return 1.0 / self.kT

    def get_restrained_system_copy(self):
        """
        Returns a copy of the restrained system.

        Returns
        -------
        system : simtk.openmm.System
           A copy of the restrained system

        """
        system = copy.deepcopy(self._system)
        force = self.get_restraint_force()
        system.addForce(force)

        return system

    @abc.abstractmethod
    def get_restraint_force(self):
        """
        Returns a new Force object that imposes the receptor-ligand restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        raise NotImplementedError('getRestraintForce not implemented')

    @abc.abstractmethod
    def get_standard_state_correction(self):
        """
        Return the standard state correction.

        Returns
        -------
        correction : float
           The standard-state correction, in kT

        """
        raise NotImplementedError('getStandardStateCorrection not implemented')


# =============================================================================================
# Base class for radially-symmetric receptor-ligand restraints.
# =============================================================================================

class RadiallySymmetricRestraint(ReceptorLigandRestraint):
    """Base class for radially-symmetric restraints between ligand and protein.

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
    * Implement the property '_energy_function' with the energy function of choice.
    * Implement the property '_bond_parameters' to return the `_energy_function`
      parameters as a dict parameter_name: parameter_value.
    * Optionally, you can overwrite the `_determine_bond_parameters()` member
      function to automatically determine these parameters from the atoms positions.

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

        This must be implemented by the inheriting class..

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
            raise NotImplemented('Restraint {} cannot automatically determine '
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

    @staticmethod
    def _compute_radius_of_gyration(positions):
        """
        Compute the radius of gyration of the specified coordinate set.

        Parameters
        ----------
        positions : simtk.unit.Quantity with units compatible with angstrom
           The coordinate set (natoms x 3) for which the radius of gyration is to be computed.

        Returns
        -------
        radius_of_gyration : simtk.unit.Quantity with units compatible with angstrom
           The radius of gyration

        """
        # TODO this is currently unused, move somewhere else? Make it a module function?
        unit = positions.unit

        # Get dimensionless receptor positions.
        x = positions / unit

        # Get dimensionless restrained atom coordinate.
        xref = x.mean(0)
        xref = np.reshape(xref, (1,3)) # (1,3) array

        # Compute distances from restrained atom.
        natoms = x.shape[0]
        distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1)) #  distances[i] is the distance from the centroid to particle i

        # Compute std dev of distances from restrained atom.
        radius_of_gyration = distances.std() * unit

        return radius_of_gyration


# =============================================================================================
# Harmonic protein-ligand restraint.
# =============================================================================================

class Harmonic(RadiallySymmetricRestraint):
    """Impose a single harmonic restraint between ligand and protein.

    This can be used to prevent the ligand from drifting too far from the
    protein in implicit solvent calculations or to keep the ligand close
    to the binding pocket in the decoupled states to increase mixing.

    The energy expression of the restraint is given by

        E = lambda_restraints * (K/2)*r^2

    where `K` is the spring constant, `r` is the distance between the
    restrained atoms, and `lambda_restraints` is a scale factor that
    can be used to control the strength of the restraint. You can control
    `lambda_restraints` through `RestraintState` class.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole. This can temporarily be left undefined,
        but in this case `determine_missing_parameters()` must be called before
        using the Restraint object (default is None).
    restrained_receptor_atom : int, optional
        The index of the receptor atom to restrain. This can temporarily
        be left undefined, but in this case `determine_missing_parameters()`
        must be called before using the Restraint object (default is None).
    restrained_ligand_atom : int, optional
        The index of the ligand atom to restrain. This can temporarily
        be left undefined, but in this case `determine_missing_parameters()`
        must be called before using the Restraint object (default is None).

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

    Create a completely defined restraint.

    >>> restraint = Harmonic(spring_constant=8*unit.kilojoule_per_mole/unit.nanometers**2,
    ...                      restrained_receptor_atom=1644, restrained_ligand_atom=2609)

    Automatically identify parameters and apply the Restraint. When trying
    to impose a restraint with undefined parameters, RestraintParameterError
    is raised.

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

        x_unit = sampler_state.positions.unit

        # Get dimensionless receptor positions.
        x = sampler_state.positions[topography.receptor_atoms, :] / x_unit

        # Get dimensionless restrained atom coordinate.
        xref = sampler_state.positions[self.restrained_receptor_atom, :] / x_unit  # (3,) array
        xref = np.reshape(xref, (1, 3))  # (1,3) array

        # Compute distances from restrained atom.
        n_atoms = x.shape[0]
        # distances[i] is the distance from the centroid to particle i
        distances = np.sqrt(((x - np.tile(xref, (n_atoms, 1)))**2).sum(1))

        # Compute std dev of distances from restrained atom.
        sigma = distances.std() * x_unit
        logger.debug("Spring Constant Sigma, s = {:.3f} nm".format(sigma / unit.nanometers))

        # Compute corresponding spring constant.
        self.spring_constant = thermodynamic_state.kT / sigma**2


# =============================================================================================
# Flat-bottom protein-ligand restraint.
# =============================================================================================

class FlatBottom(RadiallySymmetricRestraint):
    """A receptor-ligand restraint using a flat potential well with harmonic walls.

    An alternative choice to receptor-ligand restraints that uses a flat
    potential inside most of the protein volume with harmonic restraining
    walls outside of this. It can be used to prevent the ligand from
    drifting too far from protein in implicit solvent calculations while
    still exploring the surface of the protein for putative binding sites.

    More precisely, the energy expression of the restraint is given by

        E = lambda_restraints * step(r-r0) * (K/2)*(r-r0)^2

    where `K` is the spring constant, `r` is the distance between the
    restrained atoms, `r0` is another parameter defining the distance
    at which the harmonic restraint is imposed, and `lambda_restraints`
    is a scale factor that can be used to control the strength of the
    restraint. You can control `lambda_restraints` through the class
    `RestraintState`.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole. This can temporarily be left undefined,
        but in this case `determine_missing_parameters()` must be called before
        using the Restraint object (default is None).
    well_radius : simtk.unit.Quantity, optional
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance. This can temporarily be
        left undefined, but in this case `determine_missing_parameters()`
        must be called before using the Restraint object (default is None).
    restrained_receptor_atom : int, optional
        The index of the receptor atom to restrain. This can temporarily
        be left undefined, but in this case `determine_missing_parameters()`
        must be called before using the Restraint object (default is None).
    restrained_ligand_atom : int, optional
        The index of the ligand atom to restrain. This can temporarily
        be left undefined, but in this case `determine_missing_parameters()`
        must be called before using the Restraint object (default is None).

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

    Create a completely defined restraint.

    >>> restraint = FlatBottom(spring_constant=0.6*unit.kilocalorie_per_mole/unit.angstroms**2,
    ...                        well_radius=5.2*unit.nanometers, restrained_receptor_atom=1644,
    ...                        restrained_ligand_atom=2609)

    Automatically identify parameters and apply the Restraint. When trying
    to impose a restraint with undefined parameters, RestraintParameterError
    is raised.

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

        # Store parameters.
        self.spring_constant = K
        self.well_radius = r0


# =============================================================================================
# Base class for orientation-dependent receptor-ligand restraints.
# =============================================================================================


class OrientationDependentRestraint(ReceptorLigandRestraint):
    """
    Impose an orientation-dependent restraint between ligand and protein to prevent ligand from
    changing to a different binding mode.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    """

    def __init__(self, topology, state, system, positions, receptor_atoms, ligand_atoms):
        """
        Initialize a receptor-ligand restraint class.

        Parameters
        ----------
        topology : openmm.app.Topology
            Reference topology for the complex
        state : thermodynamics.ThermodynamicState
           The thermodynamic state specifying temperature, pressure, etc. to which restraints are to be added
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
           Reference positions to use for imposing restraints
        receptor_atoms : list of int
            A complete list of receptor atoms
        ligand_atoms : list of int
            A complete list of ligand atoms

        """
        super(OrientationDependentRestraint, self).__init__(topology, state, system, positions, receptor_atoms, ligand_atoms)


class Boresch(OrientationDependentRestraint):
    """
    Impose Boresch-style orientational restraints on protein-ligand system to restraint ligand binding mode.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    Notes
    -----
    Currently, all equilibrium values are measured from the initial structure, while spring constants are set to
    20 kcal/(mol A**2) or 20 kcal/(mol rad**2) as in Ref [1].

    Future iterations of this feature will introduce the ability to extract equilibrium parameters and spring constants
    from a short simulation.

    WARNING
    -------
    Symmetry corrections for symmetric ligands are not automatically applied.
    See Ref [1] and [2] for more information on correcting for ligand symmetry.

    References
    ----------
    [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839
    [2] Mobley DL, Chodera JD, and Dill KA. J Chem Phys 125:084902, 2006.
        https://dx.doi.org/10.1063%2F1.2221683

    """

    def __init__(self, topology, state, system, positions, receptor_atoms, ligand_atoms):
        """
        Initialize a receptor-ligand restraint class.

        Parameters
        ----------
        state : thermodynamics.ThermodynamicState
            The thermodynamic state specifying temperature, pressure, etc. to which restraints are to be added
        system : simtk.openmm.System
            OpenMM representation of fully interacting system
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
            Reference positions to use for imposing restraints
        receptor_atoms : list of int
            A complete list of receptor atoms
        ligand_atoms : list of int
            A complete list of ligand atoms

        """
        super(Boresch, self).__init__(topology, state, system, positions, receptor_atoms, ligand_atoms)

        self._automatic_parameter_selection(positions, receptor_atoms, ligand_atoms)

    def _automatic_parameter_selection(self, positions, receptor_atoms, ligand_atoms):
        """
        Determine parameters and restrained atoms automatically, rejecting choices where standard state correction will be incorrectly computed.

        Parameters
        ----------
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
            Reference positions to use for imposing restraints
        receptor_atoms : list of int
            A complete list of receptor atoms
        ligand_atoms : list of int
            A complete list of ligand atoms
        """
        NSIGMA = 4
        temperature = 300 * unit.kelvin
        kT = kB * temperature
        attempt = 0
        MAX_ATTEMPTS = 100
        reject = True
        logger.debug('Automatically selecting restraint atoms and parameters:')
        while reject and attempt < MAX_ATTEMPTS:
            logger.debug('Attempt %d / %d at automatically selecting atoms and restraint parameters...' % (attempt, MAX_ATTEMPTS))

            # Select atoms to be used in restraint.
            self._restraint_atoms = self._select_restraint_atoms(positions, receptor_atoms, ligand_atoms)

            # Determine restraint parameters
            self._determine_restraint_parameters()

            # Terminate if we satisfy criteria
            reject = False
            for name in ['A', 'B']:
                theta0 = self._parameters['theta_' + name + '0']
                K = self._parameters['K_theta' + name]
                sigma = unit.sqrt(NSIGMA * kT / (K/2.))
                if (theta0 < sigma) or (theta0 > (np.pi*unit.radians - sigma)):
                    logger.debug('Reject because theta_' + name + '0 is too close to 0 or pi for standard state correction to be accurate.')
                    reject = True

            r0 = self._parameters['r_aA0']
            K = self._parameters['K_r']
            sigma = unit.sqrt(NSIGMA * kT / (K/2.))
            if (r0 < sigma):
                logger.debug('Reject because r_aA0 is too close to 0 for standard state correction to be accurate.')
                reject = True

            attempt += 1

    def _is_collinear(self, positions, atoms, threshold=0.9):
        """Report whether any sequential vectors in a sequence of atoms are collinear to within a given dot product threshold.

        Parameters
        ----------
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
            Reference positions to use for imposing restraints
        atoms : array-like, dtype:int, length:natoms
            A seris of indices of atoms
        threshold : float, optional, default=0.9
            Atoms are not collinear if their sequential vector separation dot products are less than THRESHOLD

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

    def _select_restraint_atoms(self, positions, receptor_atoms, ligand_atoms):
        """
        Select atoms to be used in restraint.

        Parameters
        ----------
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
           Reference positions to use for imposing restraints
        receptor_atoms : list of int
            List of atom indices belonging to the receptor
        ligand_atoms : list of int
            List of atom indices belonging to the ligand

        Returns
        -------
        restraint_atoms : list of int
            List of six atom indices used in the restraint.
            restraint_atoms[0:3] belong to the receptor
            restraint_atoms[4:6] belong to the ligand

        Notes
        -----
        The current algorithm simply selects random subsets of receptor and ligand atoms
        and rejects those that are too close to collinear. Future updates can further refine
        this algorithm.

        """

        md_top = md.Topology.from_openmm(self._topology)
        t = md.Trajectory(self._positions / unit.nanometers, md_top)
        # Determine heavy atoms. Using sets since lists should be unique anyways
        heavy_atoms = set(md_top.select('not element H'))
        # Intersect heavy atoms with receptor/ligand atoms (s1&s2 is intersect)
        heavy_ligand_atoms = set(ligand_atoms) & heavy_atoms
        heavy_receptor_atoms = set(receptor_atoms) & heavy_atoms
        if (len(heavy_receptor_atoms) < 3) or (len(heavy_ligand_atoms) < 3):
            raise ValueError('There must be at least three heavy atoms in receptor_atoms (# heavy %d) and ligand_atoms '
                             '(# heavy %d).' % (len(heavy_receptor_atoms), len(heavy_ligand_atoms)))
        # Find valid pairs of ligand/receptor atoms within a cutoff
        max_distance = 4 * unit.angstrom/unit.nanometer
        min_distance = 1 * unit.angstrom/unit.nanometer
        # TODO: Cast itertools generator to np array more efficiently
        all_pairs = np.array(list(itertools.product(heavy_receptor_atoms, heavy_ligand_atoms)))
        distances = md.geometry.compute_distances(t, all_pairs)[0]
        index_of_in_range_atoms = np.where(np.logical_and(distances > min_distance, distances <= max_distance))[0]
        if len(index_of_in_range_atoms) == 0:
            error_string = 'There are no heavy ligand atoms within the range of [{},{}] nm heavy receptor atoms!\n'
            error_string += 'Please Check your input files or try another restraint class'
            raise ValueError(error_string.format(min_distance, max_distance))
        # Iterate until we have found a set of non-collinear atoms.
        accepted = False
        while not accepted:
            # Select a receptor/ligand atom in range of each other
            raA_atoms = all_pairs[random.sample(list(index_of_in_range_atoms), 1)[0]].tolist()
            # Cast to set for easy comparison operations
            raA_set = set(raA_atoms)
            # Select two additional random atoms from the receptor and ligand
            restraint_atoms = random.sample(heavy_receptor_atoms-raA_set, 2) + raA_atoms + random.sample(heavy_ligand_atoms-raA_set, 2)
            # Reject collinear sets of atoms.
            accepted = not self._is_collinear(positions, restraint_atoms)
        # Cast to Python ints to avoid type issues when passing to OpenMM
        restraint_atoms = [int(i) for i in restraint_atoms]

        return restraint_atoms

    def _determine_restraint_parameters(self):
        """
        Determine restraint parameters.

        Attributes set
        --------------
        [r_aA0, theta_A0, theta_B0]
        [K_r, K_thetaA, K_thetaB, K_phiA, K_phiB, K_phiC]
        See Ref [1] for definitions

        Notes
        -----
        Currently, all equilibrium values are measured from the initial structure, while spring constants are set to
        20 kcal/(mol A**2) or 20 kcal/(mol rad**2) as in [1].

        Future iterations of this feature will introduce the ability to extract equilibrium parameters and spring constants
        from a short simulation.

        References
        ----------
        [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839

        """
        self._parameters = dict()

        # Set spring constants uniformly, as in Ref [1] Table 1 caption.
        self._parameters['K_r'] = 20.0 * unit.kilocalories_per_mole / unit.angstrom**2
        for name in ['K_thetaA', 'K_thetaB', 'K_phiA', 'K_phiB', 'K_phiC']:
            self._parameters[name] = 20.0 * unit.kilocalories_per_mole / unit.radian**2

        # Measure equilibrium geometries from static reference structure
        t = md.Trajectory(self._positions / unit.nanometers, self._topology)

        distances = md.geometry.compute_distances(t, [self._restraint_atoms[2:4]], periodic=False)
        self._parameters['r_aA0'] = distances[0][0] * unit.nanometers

        angles = md.geometry.compute_angles(t, [self._restraint_atoms[i:(i+3)] for i in range(1,3)], periodic=False)
        for (name, angle) in zip(['theta_A0', 'theta_B0'], angles[0]):
            self._parameters[name] = angle * unit.radians

        dihedrals = md.geometry.compute_dihedrals(t, [self._restraint_atoms[i:(i+4)] for i in range(3)], periodic=False)
        for (name, angle) in zip(['phi_A0', 'phi_B0', 'phi_C0'], dihedrals[0]):
            self._parameters[name] = angle * unit.radians

        # Write restraint parameters
        msg = 'restraint parameters:\n'
        for name in ['K_r', 'r_aA0', 'K_thetaA', 'theta_A0', 'K_thetaB', 'theta_B0', 'K_phiA', 'phi_A0', 'K_phiB', 'phi_B0', 'K_phiC', 'phi_C0']:
            msg += '%24s : %s\n' % (name, str(self._parameters[name]))
        logger.debug(msg)

    def get_restraint_force(self):
        """
        Create a new copy of the receptor-ligand restraint force.

        Returns
        -------
        force : simtk.openmm.CustomCompoundBondForce
           A restraint force object

        WARNING
        -------
        Because the domain of the `dihedral()` function in `CustomCompoundBondForce` is unknown, it is not currently clear whether the dihedral restraints will work.
        We should replace those restraints with a negative log von Mises distribution: https://en.wikipedia.org/wiki/Von_Mises_distribution
        where the von Mises parameter kappa would be computed from the desired standard deviation (kappa ~ sigma**(-2))
        and the standard state correction would need to be modified

        Notes
        -----
        There may be a better way to implement this restraint.

        """
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
        for name in self._parameters:
            energy_function += '%s = %f; ' % (name, self._parameters[name].value_in_unit_system(unit.md_unit_system))
        # Create the force
        nparticles = 6 # number of particles involved in restraint: p1 ... p6
        force = openmm.CustomCompoundBondForce(nparticles, energy_function)
        force.addGlobalParameter('lambda_restraints', 1.0)
        force.addBond(self._restraint_atoms, [])
        is_periodic = self._system.usesPeriodicBoundaryConditions()
        try:  # This was added in OpenMM 7.1
            force.setUsesPeriodicBoundaryConditions(is_periodic)
        except AttributeError:
            pass
        return force

    def get_standard_state_correction(self):
        """
        Compute the standard state correction for the arbitrary restraint energy function.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT)

        Notes
        -----
        Uses analytical approach from [1], but this approach is known to be inexact.
        This approach breaks down when the equilibrium restraint angles are near the limits of their domains and when
        equilibrium distance is near 0.

        """

        DeltaG = self.get_standard_state_correction_static('analytical', kT=self.kT, parameters=self._parameters)
        return DeltaG

    @staticmethod
    def get_standard_state_correction_static(method='numerical', **kwargs):
        """
        Compute the standard state correction for the arbitrary restraint energy function.

        Returns
        -------
        DeltaG : float
            Computed standard-state correction in dimensionless units (kT)

        Notes
        -----
        'analytical' uses analytical approach from [1], but this approach is known to be inexact.
        'numerical' uses numerical integral to the partition function contriubtions for r and theta, analytical for phi

        """

        class Bunch(object):
            """Make a dict accessible via an object accessor"""
            def __init__(self, adict):
                self.__dict__.update(adict)

        def strip(passed_unit):
            """Cast the passed_unit into md unit system for integrand lambda functions"""
            return passed_unit.value_in_unit_system(unit.md_unit_system)

        all_valid_keys = ['kT', 'parameters']
        for key in all_valid_keys:
            if key not in kwargs:
                raise KeyError('Missing {} from input arguments!'.format(key))
        kT = kwargs['kT']
        pi = np.pi
        p = Bunch(kwargs['parameters'])
        DeltaG = 0

        # Multiply by unit.radian**5 to remove the expected unit value
        # radians is a soft unit in this case, it cancels in the math, but not in the equations here.
        if method is 'analytical':
            # Eq 32 of Ref [1]
            DeltaG += -np.log( \
                (8. * pi ** 2 * V0) / (p.r_aA0 ** 2 * unit.sin(p.theta_A0) * unit.sin(p.theta_B0)) \
                * unit.sqrt(p.K_r * p.K_thetaA * p.K_thetaB * p.K_phiA * p.K_phiB * p.K_phiC) / (2 * pi * kT) ** 3 \
                * unit.radian**5)
        elif method is 'numerical':
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
            DeltaG += -np.log(8 * pi**2 * V0 / ExpDeltaG)
        else:
            raise ValueError('"method" must be "analytical" or "numerical"')

        return DeltaG


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(Harmonic, globals())
