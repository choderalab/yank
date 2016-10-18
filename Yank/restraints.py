#!/usr/local/bin/env python
#
#=============================================================================================
# FILE DOCSTRING
#=============================================================================================

"""

Automated selection and imposition of receptor-ligand restraints for absolute alchemical binding
free energy calculations, along with computation of the standard state correction.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import copy
import math
import time
import random

import numpy as np
import scipy.integrate

from simtk import openmm, unit

import mdtraj as md

import abc

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA # Boltzmann constant
V0 = 1660.0*unit.angstroms**3 # standard state volume

#=============================================================================================
# Dispatch appropriate restraint type from registered restraint classes
#=============================================================================================

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

    def is_abstract(cls):
        if not hasattr(cls, "__abstractmethods__"):
            return False # an ordinary class
        elif len(cls.__abstractmethods__) == 0:
            return False # a concrete implementation of an abstract class
        else:
            return True # an abstract class

    # Build an index of all names, ensuring there are no name collisions.
    available_restraints = dict()
    for cls in get_all_subclasses(ReceptorLigandRestraint):
        classname = cls.__name__
        if is_abstract(cls):
            # Skip abstract base classes
            pass
        elif (classname in available_restraints):
            raise Exception("More than one restraint subclass has the name '%s'." % classname)
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

def createRestraints(restraint_type, topology, state, system, positions, receptor_atoms, ligand_atoms):
    """
    Initialize a receptor-ligand restraint class matching the specified restraint type name.

    Parameters
    ----------
    restraint_type : str
        Restraint type name matching a register (imported) subclass of ReceptorLigandRestraint.
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
    available_restraints = available_restraint_classes()
    if not (restraint_type in available_restraints):
        raise Exception("Restraint type '%s' unknown. Options are: %s" % (restraint_type, str(available_restraints.keys())))
    cls = available_restraints[restraint_type]
    return cls(topology, state, system, positions, receptor_atoms, ligand_atoms)

#=============================================================================================
# Base class for receptor-ligand restraints.
#=============================================================================================

ABC = abc.ABCMeta('ABC', (object,), {}) # compatible with Python 2 *and* 3

class ReceptorLigandRestraint(ABC):
    """
    Impose a single restraint between ligand and protein to prevent ligand from drifting too far
    from protein in implicit solvent calculations.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    NOTES

    Creating a subclass requires the following:
    * Perform any necessary processing in subclass `__init__` after calling base class `__init__`
    * Override getRestraintForce() to return a new `Force` instance imposing the restraint
    * Override getStandardStateCorrection() to return standard state correction

    Public class fields
    -------------------
    topology : openmm.app.Topology
        Reference topology for the complex
    state : simtk.openmm.State
        The State object used during initialization
    system : simtk.openmm.System
        The System object used during initialization
    positions : simtk.unit.Quantity of dimension (nparticles,3) with units compatible with angstroms
        The system positions used during initialization to define the restraint
        TODO: Should this be optional?
    receptor_atoms : list of int
        List of atoms declared to be part of the receptor
    ligand_atoms : list of int
        List of atoms declared to be part of the ligand to be restrained to the receptor
    temperature : simtk.unit.Quantity
        The temperature specified during initialization.
    kT : simtk.unit.Quantity
        Thermal specified during initialization.
    beta : simtk.unit.Quantity
        The inverse temperature specified during initialization.

    TODO: Should these all really be public data fields?

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
        self.topology = topology
        self.state = state
        self.system = system
        self.positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)
        self.receptor_atoms = list(receptor_atoms)
        self.ligand_atoms = list(ligand_atoms)

        # Perform some sanity checks.
        natoms = system.getNumParticles()
        if np.any(np.array(receptor_atoms) >= natoms):
            raise Exception("Receptor atom indices fall outside [0,natoms), where natoms = %d.  receptor_atoms = %s" % (natoms, str(receptor_atoms)))
        if np.any(np.array(ligand_atoms) >= natoms):
            raise Exception("Ligand atom indices fall outside [0,natoms), where natoms = %d.  ligand_atoms = %s" % (natoms, str(ligand_atoms)))

        self.temperature = state.temperature
        self.kT = kB * self.temperature # thermal energy
        self.beta = 1.0 / self.kT # inverse temperature

    def getRestrainedSystemCopy(self):
        """
        Returns a copy of the restrained system.

        Returns
        -------
        system : simtk.openmm.System
           A copy of the restrained system

        """
        system = copy.deepcopy(self.system)
        force = self.getRestraintForce()
        system.addForce(force)

        return system

    @abc.abstractmethod
    def getRestraintForce(self):
        """
        Returns a new Force object that imposes the receptor-ligand restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        raise NotImplementedError('getRestraintForce not implemented')

    @abc.abstractmethod
    def getStandardStateCorrection(self):
        """
        Return the standard state correction.

        Returns
        -------
        correction : float
           The standard-state correction, in kT

        """
        raise NotImplementedError('getStandardStateCorrection not implemented')

#=============================================================================================
# Base class for radially-symmetric receptor-ligand restraints.
#=============================================================================================

class RadiallySymmetricRestraint(ReceptorLigandRestraint):
    """
    Impose a single radially-symmetric restraint between ligand and protein to prevent ligand from drifting too far
    from protein in implicit solvent calculations. The restraint is implemented as a `CustomBondForce`.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    NOTES

    To create a subclass that uses a different restraint energy function, follow these steps:

    * Redefine class variable 'energy_function' with the energy function of choice.
    * Redefine class variable 'bond_parameter_names' to list the parameters in 'energy_function'
      that must be computed for each system.
    * Redefine the _determineBondParameters() member function to compute these parameters and
      return them in a list in the same order as 'bond_parameter_names'.

    """
    energy_function = ''  # energy function to use in computation of restraint
    bond_parameter_names = []  # list of bond parameters that appear in energy function above

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
        super(RadiallySymmetricRestraint, self).__init__(topology, state, system, positions, receptor_atoms, ligand_atoms)

        # Determine atoms closet to centroids on ligand and receptor.
        self.restrained_receptor_atom = self._closestAtomToCentroid(self.positions, self.receptor_atoms)
        self.restrained_ligand_atom = self._closestAtomToCentroid(self.positions, self.ligand_atoms)

        if not (self.restrained_receptor_atom in set(receptor_atoms)):
            raise Exception("Restrained receptor atom (%d) not in set of receptor atoms (%s)" % (self.restrained_receptor_atom, str(receptor_atoms)))
        if not (self.restrained_ligand_atom in set(ligand_atoms)):
            raise Exception("Restrained ligand atom (%d) not in set of ligand atoms (%s)" % (self.restrained_ligand_atom, str(ligand_atoms)))

        logger.debug("restrained receptor atom: %d" % self.restrained_receptor_atom)
        logger.debug("restrained ligand atom: %d" % self.restrained_ligand_atom)

        # Determine radius of gyration of receptor.
        self.radius_of_gyration = self._computeRadiusOfGyration(self.positions[self.receptor_atoms,:])

        # Determine parameters
        self.bond_parameters = self._determineBondParameters()

    @abc.abstractmethod
    def _determineBondParameters(self):
        """
        Determine bond parameters for CustomBondForce between protein and ligand.

        Returns
        -------
        parameters : list
           List of parameters for CustomBondForce

        Notes
        -----
        The spring constant is selected to give 1 kT at one standard deviation of receptor atoms about the receptor
        restrained atom.

        """
        pass

    def _computeRadiusOfGyration(self, positions):
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

        unit = positions.unit

        # Get dimensionless receptor positions.
        x = positions / unit

        # Get dimensionless restrained atom coordinate.
        xref = x.mean(0)
        xref = np.reshape(xref, (1,3)) # (1,3) array

        # Compute distances from restrained atom.
        natoms = x.shape[0]
        distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1)) # distances[i] is the distance from the centroid to particle i

        # Compute std dev of distances from restrained atom.
        radius_of_gyration = distances.std() * unit

        return radius_of_gyration

    def _createRestraintForce(self, particle1, particle2):
        """
        Create a new restraint force between specified atoms.

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
        force = openmm.CustomBondForce(self.energy_function)
        force.addGlobalParameter('lambda_restraints', 1.0)
        for parameter in self.bond_parameter_names:
            force.addPerBondParameter(parameter)
        try:
            force.addBond(particle1, particle2, self.bond_parameters)
        except Exception as e:
            print('particle1: %s' % str(particle1))
            print('particle2: %s' % str(particle1))
            print('bond_parameters: %s' % str(self.bond_parameters))
            raise(e)
        return force

    def getRestraintForce(self):
        """
        Returns a new Force object that imposes the receptor-ligand restraint.

        Returns
        -------
        force : simtk.openmm.CustomBondForce
           The created restraint force.

        """
        return self._createRestraintForce(self.restrained_receptor_atom, self.restrained_ligand_atom)

    def getStandardStateCorrection(self):
        """
        Return the standard state correction.

        Returns
        -------
        correction : float
           The standard-state correction, in kT

        """
        initial_time = time.time()

        r_min = 0 * unit.nanometers
        r_max = 100 * unit.nanometers # TODO: Use maximum distance between atoms?

        # Create a System object containing two particles connected by the reference force
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)
        system.addParticle(1.0 * unit.amu)
        force = self._createRestraintForce(0, 1)
        system.addForce(force)

        # Create a Reference context to evaluate energies on the CPU.
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, integrator, platform)

        # Set default positions.
        positions = unit.Quantity(np.zeros([2,3]), unit.nanometers)
        context.setPositions(positions)

        # Create a function to compute integrand as a function of interparticle separation.
        beta = self.beta

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

        (shell_volume, shell_volume_error) = scipy.integrate.quad(lambda r : integrand(r), r_min / unit.nanometers, r_max / unit.nanometers) * unit.nanometers**3 # integrate shell volume
        logger.debug("shell_volume = %f nm^3" % (shell_volume / unit.nanometers**3))

        # Compute standard-state volume for a single molecule in a box of size (1 L) / (avogadros number)
        liter = 1000.0 * unit.centimeters**3  # one liter
        box_volume = liter / (unit.AVOGADRO_CONSTANT_NA*unit.mole) # standard state volume
        logger.debug("box_volume = %f nm^3" % (box_volume / unit.nanometers**3))

        # Compute standard state correction for releasing shell restraints into standard-state box (in units of kT).
        DeltaG = - math.log(box_volume / shell_volume)
        logger.debug("Standard state correction: %.3f kT" % DeltaG)

        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("restraints: _computeStandardStateCorrection: %.3f s elapsed" % elapsed_time)

        # Return standard state correction (in kT).
        return DeltaG

    def _closestAtomToCentroid(self, positions, indices=None, masses=None):
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
            positions = positions[indices,:]

        # Get dimensionless positions.
        x_unit = positions.unit
        x = positions / x_unit

        # Determine number of atoms.
        natoms = x.shape[0]

        # Compute (natoms,1) array of normalized weights.
        w = np.ones([natoms, 1])
        if masses:
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

#=============================================================================================
# Harmonic protein-ligand restraint.
#=============================================================================================

class Harmonic(RadiallySymmetricRestraint):
    """
    Impose a single restraint between ligand and protein to prevent ligand from drifting too far
    from protein in implicit solvent calculations.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    NOTE

    These restraints should not be used with explicit solvent calculations, since the CustomBondForce
    does not respect periodic boundary conditions, and the analytical correction term does not include
    truncation due to a finite simulation box.

    EXAMPLE

    >>> # Create a test system.
    >>> from openmmtools import testsystems
    >>> system_container = testsystems.LysozymeImplicit()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> # Identify receptor and ligand atoms.
    >>> receptor_atoms = range(0,2603)
    >>> ligand_atoms = range(2603,2621)
    >>> # Construct a reference thermodynamic state.
    >>> from repex import ThermodynamicState
    >>> temperature = 298.0 * unit.kelvin
    >>> state = ThermodynamicState(temperature=temperature)
    >>> # Create restraints.
    >>> restraints = Harmonic(state, system, positions, receptor_atoms, ligand_atoms)
    >>> # Get standard state correction.
    >>> correction = restraints.getStandardStateCorrection()
    >>> # Get radius of gyration of receptor.
    >>> rg = restraints.getReceptorRadiusOfGyration()

    """

    energy_function = 'lambda_restraints * (K/2)*r^2'  # harmonic restraint
    bond_parameter_names = ['K']  # list of bond parameters that appear in energy function above

    def _determineBondParameters(self):
        """
        Determine bond parameters for CustomBondForce between protein and ligand.

        RETURNS

        parameters (list) - list of parameters for CustomBondForce

        NOTE

        K, the spring constant, is determined from the radius of gyration

        """

        unit = self.positions.unit

        # Get dimensionless receptor positions.
        x = self.positions[self.receptor_atoms,:] / unit

        # Get dimensionless restrained atom coordinate.
        xref = self.positions[self.restrained_receptor_atom,:] / unit # (3,) array
        xref = np.reshape(xref, (1,3)) # (1,3) array

        # Compute distances from restrained atom.
        natoms = x.shape[0]
        # distances[i] is the distance from the centroid to particle i
        distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1))

        # Compute std dev of distances from restrained atom.
        sigma = distances.std() * unit

        # Compute corresponding spring constant.
        K = self.kT / sigma**2

        # Assemble parameters.
        bond_parameters = [K]

        return bond_parameters

#=============================================================================================
# Flat-bottom protein-ligand restraint.
#=============================================================================================

class FlatBottom(RadiallySymmetricRestraint):
    """
    An alternative choice to receptor-ligand restraints that uses a flat potential inside most of the protein volume
    with harmonic restraining walls outside of this.

    Examples
    --------

    >>> # Create a test system.
    >>> from openmmtools import testsystems
    >>> system_container = testsystems.LysozymeImplicit()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> # Identify receptor and ligand atoms.
    >>> receptor_atoms = range(0,2603)
    >>> ligand_atoms = range(2603,2621)
    >>> # Construct a reference thermodynamic state.
    >>> from repex import ThermodynamicState
    >>> temperature = 298.0 * unit.kelvin
    >>> state = ThermodynamicState(temperature=temperature)
    >>> # Create restraints.
    >>> restraints = FlatBottom(state, system, positions, receptor_atoms, ligand_atoms)
    >>> # Get standard state correction.
    >>> correction = restraints.getStandardStateCorrection()
    >>> # Get radius of gyration of receptor.
    >>> rg = restraints.getReceptorRadiusOfGyration()

    """

    energy_function = 'lambda_restraints * step(r-r0) * (K/2)*(r-r0)^2'  # flat-bottom restraint
    bond_parameter_names = ['K', 'r0']  # list of bond parameters that appear in energy function above

    def _determineBondParameters(self):
        """
        Determine bond parameters for CustomBondForce between protein and ligand.

        RETURNS

        parameters (list) - list of parameters for CustomBondForce

        NOTE

        r0, the distance at which the harmonic restraint is imposed, is set at twice the robust estimate of the standard deviation (from mean absolute deviation) plus 5 A
        K, the spring constant, is set to 5.92 kcal/mol/A**2

        """

        x_unit = self.positions.unit

        # Get dimensionless receptor positions.
        x = self.positions[self.receptor_atoms, :] / x_unit

        # Determine number of atoms.
        natoms = x.shape[0]

        if (natoms > 3):
            # Check that restrained receptor atom is in expected range.
            if (self.restrained_receptor_atom > natoms):
                raise Exception('Receptor atom %d was selected for restraint, but system only has %d atoms.' % (self.restrained_receptor_atom, natoms))

            # Compute median absolute distance to central atom.
            # (Working in non-unit-bearing floats for speed.)
            xref = np.reshape(x[self.restrained_receptor_atom,:], (1,3)) # (1,3) array
            distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1)) # distances[i] is the distance from the centroid to particle i
            median_absolute_distance = np.median(abs(distances))

            # Convert back to unit-bearing quantity.
            median_absolute_distance *= x_unit

            # Convert to estimator of standard deviation for normal distribution.
            sigma = 1.4826 * median_absolute_distance

            # Calculate r0, which is a multiple of sigma plus 5 A.
            r0 = 2*sigma + 5.0 * unit.angstroms
        else:
            DEFAULT_DISTANCE = 15.0 * unit.angstroms
            logger.warning("Receptor only contains %d atoms; using default distance of %s" % (natoms, str(DEFAULT_DISTANCE)))
            r0 = DEFAULT_DISTANCE

        logger.debug("restraint distance r0 = %.1f A" % (r0 / unit.angstroms))

        # Set spring constant/
        #K = (2.0 * 0.0083144621 * 5.0 * 298.0 * 100) * unit.kilojoules_per_mole/unit.nanometers**2
        K = 0.6 * unit.kilocalories_per_mole / unit.angstroms**2
        logger.debug("K = %.1f kcal/mol/A^2" % (K / (unit.kilocalories_per_mole / unit.angstroms**2)))

        # Assemble parameter vector.
        bond_parameters = [K, r0]

        return bond_parameters


#=============================================================================================
# Base class for orientation-dependent receptor-ligand restraints.
#=============================================================================================

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

        # Select atoms to be used in restraint.
        self.restraint_atoms = self._select_restraint_atoms(positions, receptor_atoms, ligand_atoms)

        # Determine restraint parameters
        self._determine_restraint_parameters(state, system, positions)

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
        if (len(receptor_atoms) < 3) or (len(ligand_atoms) < 3):
            raise Exception('Both receptor_atoms (len %d) and ligand_atoms (len %d) must contain at least three atoms.' % (len(receptor_atoms), len(ligand_atoms)))

        # Iterate until we have found a set of non-collinear atoms.
        accepted = False
        while (not accepted):
            # Select three random atoms from the receptor and ligand
            restraint_atoms = random.sample(receptor_atoms, 3) + random.sample(ligand_atoms, 3)
            # Reject collinear sets of atoms.
            accepted = not self._is_collinear(positions, restraint_atoms)

        return restraint_atoms

    def _determine_restraint_parameters(self, state, system, positions):
        """
        Determine restraint parameters.

        Parameters
        ----------
        state : thermodynamics.ThermodynamicState
            The thermodynamic state specifying temperature, pressure, etc. to which restraints are to be added
        system : simtk.openmm.System
            OpenMM representation of fully interacting system
        positions : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
            Reference positions to use for imposing restraints

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
        # Set spring constants uniformly, as in Ref [1] Table 1 caption.
        self.K_r = 20.0 * unit.kilocalories_per_mole / unit.angstrom**2
        self.K_thetaA = self.K_thetaB = self.K_phiA = self.K_phiB = self.K_phiC = 20.0 * unit.kilocalories_per_mole / unit.radian**2

        # Measure equilibrium geometries from static reference structure
        t = md.Trajectory(self.positions / unit.nanometers, self.topology)
        distances = md.geometry.compute_distances(t, [self.restraint_atoms[2:4]], periodic=False)
        [self.r_aA0] = distances[0] * unit.nanometers
        angles = md.geometry.compute_angles(t, [self.restraint_atoms[1:4], self.restraint_atoms[2:5]], periodic=False)
        [self.theta_A0, self.theta_B0] = [angles[0][0] * unit.radians, angles[0][1] * unit.radians]
        dihedrals = md.geometry.compute_dihedrals(t, [self.restraint_atoms[0:4], self.restraint_atoms[1:5], self.restraint_atoms[2:6]], periodic=False)
        [self.phi_A0, self.phi_B0, self.phi_C0] = [dihedrals[0][0] * unit.radians, dihedrals[0][1] * unit.radians, dihedrals[0][2] * unit.radians]

    def getRestraintForce(self):
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
        constants = [
            'K_r', 'r_aA0', 'K_thetaA', 'theta_A0', 'K_thetaB', 'theta_B0',
            'K_phiA', 'phi_A0', 'K_phiB', 'phi_B0', 'K_phiC', 'phi_C0']
        for constant in constants:
            energy_function += '%s = %f; ' % (constant, getattr(self, constant).value_in_unit_system(unit.md_unit_system))
        # Create the force
        nparticles = 6 # number of particles involved in restraint: p1 ... p6
        force = openmm.CustomCompoundBondForce(nparticles, energy_function)
        force.addGlobalParameter('lambda_restraints', 1.0)
        force.addBond(self.restraint_atoms, [])
        return force

    def getStandardStateCorrection(self):
        """
        Compute the standard state correction for the arbitrary restraint energy function.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT)

        Notes
        -----
        Uses analytical approach from [1], but this approach is known to be inexact.

        """
        # Retrieve constants for convenience.
        [r_aA0, theta_A0, theta_B0] = [self.r_aA0, self.theta_A0, self.theta_B0]
        [K_r, K_thetaA, K_thetaB, K_phiA, K_phiB, K_phiC] = [self.K_r, self.K_thetaA, self.K_thetaB, self.K_phiA, self.K_phiB, self.K_phiC]
        kT = self.kT
        pi = np.pi

        # Eq 32 of Ref [1]
        DeltaG = -np.log( \
            (8. * pi**2 * V0) / (r_aA0**2 * unit.sin(theta_A0) * unit.sin(theta_B0)) \
            * unit.sqrt(K_r * K_thetaA * K_thetaB * K_phiA * K_phiB * K_phiC) / (2 * pi * kT)**3 \
            )
        # Return standard state correction (in kT).
        return DeltaG
