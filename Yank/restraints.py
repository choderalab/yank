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

import numpy as np
import scipy.integrate

import simtk.unit as units
import simtk.openmm as openmm

from abc import ABCMeta, abstractmethod

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# Base class for receptor-ligand restraints.
#=============================================================================================


class ReceptorLigandRestraint(object):
    """
    Impose a single restraint between ligand and protein to prevent ligand from drifting too far
    from protein in implicit solvent calculations.

    This restraint strength is controlled by a global context parameter called 'lambda_restraints'.

    NOTES

    To create a subclass that uses a different restraint energy function, follow these steps:

    * Redefine class variable 'energy_function' with the energy function of choice.
    * Redefine class variable 'bond_parameter_names' to list the parameters in 'energy_function'
      that must be computed for each system.
    * Redefine the _determineBondParameters() member function to compute these parameters and
      return them in a list in the same order as 'bond_parameter_names'.

    """

    __metaclass__ = ABCMeta

    energy_function = ''  # energy function to use in computation of restraint
    bond_parameter_names = []  # list of bond parameters that appear in energy function above

    def __init__(self, state, system, coordinates, receptor_atoms, ligand_atoms):
        """
        Initialize a receptor-ligand restraint class.

        Parameters
        ----------
        state : thermodynamics.ThermodynamicState
           The thermodynamic state specifying temperature, pressure, etc. to which restraints are to be added
        coordinates : simtk.unit.Quantity of natoms x 3 with units compatible with nanometers
           Reference coordinates to use for imposing restraints
        receptor_atoms : list of int
            A complete list of receptor atoms
        ligand_atoms : list of int
            A complete list of ligand atoms

        """

        self.state = state
        self.system = system
        self.coordinates = units.Quantity(np.array(coordinates / coordinates.unit), coordinates.unit)
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

        # Determine atoms closet to centroids on ligand and receptor.
        self.restrained_receptor_atom = self._closestAtomToCentroid(self.coordinates, self.receptor_atoms)
        self.restrained_ligand_atom = self._closestAtomToCentroid(self.coordinates, self.ligand_atoms)

        if not (self.restrained_receptor_atom in set(receptor_atoms)):
            raise Exception("Restrained receptor atom (%d) not in set of receptor atoms (%s)" % (self.restrained_receptor_atom, str(receptor_atoms)))
        if not (self.restrained_ligand_atom in set(ligand_atoms)):
            raise Exception("Restrained ligand atom (%d) not in set of ligand atoms (%s)" % (self.restrained_ligand_atom, str(ligand_atoms)))

        logger.debug("restrained receptor atom: %d" % self.restrained_receptor_atom)
        logger.debug("restrained ligand atom: %d" % self.restrained_ligand_atom)

        # Determine radius of gyration of receptor.
        self.radius_of_gyration = self._computeRadiusOfGyration(self.coordinates[self.receptor_atoms,:])

        # Determine parameters
        self.bond_parameters = self._determineBondParameters()

        # Determine standard state correction.
        self.standard_state_correction = self._computeStandardStateCorrection()

        return

    @abstractmethod
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

    def _computeRadiusOfGyration(self, coordinates):
        """
        Compute the radius of gyration of the specified coordinate set.

        Parameters
        ----------
        coordinates : simtk.unit.Quantity with units compatible with angstrom
           The coordinate set (natoms x 3) for which the radius of gyration is to be computed.

        Returns
        -------
        radius_of_gyration : simtk.unit.Quantity with units compatible with angstrom
           The radius of gyration

        """

        unit = coordinates.unit

        # Get dimensionless receptor coordinates.
        x = coordinates / unit

        # Get dimensionless restrained atom coordinate.
        xref = x.mean(0)
        xref = np.reshape(xref, (1,3)) # (1,3) array

        # Compute distances from restrained atom.
        natoms = x.shape[0]
        distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1)) # distances[i] is the distance from the centroid to particle i

        # Compute std dev of distances from restrained atom.
        radius_of_gyration = distances.std() * unit

        return radius_of_gyration

    def _createRestraintForce(self, particle1, particle2, mm=None):
        """
        Create a new copy of the receptor-ligand restraint force.

        Parameters
        ----------
        particle1 : int
           Index of first particle for which restraint is to be applied.
        particle2 : int
           Index of second particle for which restraint is to be applied
        mm : simtk.openmm compliant interface, optional, default=None
           If specified, use an alternative OpenMM API implementation.
           Otherwise, use simtk.openmm.

        Returns
        -------
        force : simtk.openmm.CustomBondForce
           A restraint force object

        """

        if mm is None: mm = openmm

        force = openmm.CustomBondForce(self.energy_function)
        force.addGlobalParameter('lambda_restraints', 1.0)
        for parameter in self.bond_parameter_names:
            force.addPerBondParameter(parameter)
        force.addBond(particle1, particle2, self.bond_parameters)

        return force

    def _computeStandardStateCorrection(self):
        """
        Compute the standard state correction for the arbitrary restraint energy function.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT)

        Notes
        -----
        Equivalent to the free energy of releasing restraints and confining into a box of standard state size.

        """

        initial_time = time.time()

        r_min = 0 * units.nanometers
        r_max = 100 * units.nanometers # TODO: Use maximum distance between atoms?

        # Create a System object containing two particles connected by the reference force
        system = openmm.System()
        system.addParticle(1.0 * units.amu)
        system.addParticle(1.0 * units.amu)
        force = self._createRestraintForce(0, 1)
        system.addForce(force)

        # Create a Reference context to evaluate energies on the CPU.
        integrator = openmm.VerletIntegrator(1.0 * units.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, integrator, platform)

        # Set default positions.
        positions = units.Quantity(np.zeros([2,3]), units.nanometers)
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
            positions[1, 0] = r * units.nanometers
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            potential = state.getPotentialEnergy()
            dI = 4.0 * math.pi * r**2 * math.exp(-beta * potential)
            return dI

        (shell_volume, shell_volume_error) = scipy.integrate.quad(lambda r : integrand(r), r_min / units.nanometers, r_max / units.nanometers) * units.nanometers**3 # integrate shell volume
        logger.debug("shell_volume = %f nm^3" % (shell_volume / units.nanometers**3))

        # Compute standard-state volume for a single molecule in a box of size (1 L) / (avogadros number)
        liter = 1000.0 * units.centimeters**3  # one liter
        box_volume = liter / (units.AVOGADRO_CONSTANT_NA*units.mole) # standard state volume
        logger.debug("box_volume = %f nm^3" % (box_volume / units.nanometers**3))

        # Compute standard state correction for releasing shell restraints into standard-state box (in units of kT).
        DeltaG = - math.log(box_volume / shell_volume)
        logger.debug("Standard state correction: %.3f kT" % DeltaG)
        
        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("restraints: _computeStandardStateCorrection: %.3f s elapsed" % elapsed_time)

        # Return standard state correction (in kT).
        return DeltaG

    def getRestraintForce(self, mm=None):
        """
        Returns a new Force object that imposes the receptor-ligand restraint.
        
        Parameters
        ----------
        mm : simtk.openmm compliant interface, optional, default=None
           If specified, use an alternative OpenMM API implementation.
           Otherwise, use simtk.openmm.

        Returns
        -------
        force : simtk.openmm.HarmonicBondForce
           The created restraint force.

        """

        return self._createRestraintForce(self.restrained_receptor_atom, self.restrained_ligand_atom, mm=mm)

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
    
    def getStandardStateCorrection(self):
        """
        Return the standard state correction.

        Returns
        -------
        correction : float
           The standard-state correction, in kT

        """
        return self.standard_state_correction

    def getReceptorRadiusOfGyration(self):
        """
        Returns the radius of gyration of the receptor.

        Returns
        -------
        radius_of_gyration : simtk.unit.Quantity with units compatible with angstrom
            Radius of gyration of the receptor

        """
        return self.radius_of_gyration

    def _closestAtomToCentroid(self, coordinates, indices=None, masses=None):
        """
        Identify the closest atom to the centroid of the given coordinate set.

        Parameters
        ----------
        coordinates : units.Quantity of natoms x 3 with units compatible with nanometers
           Coordinates of object to identify atom closes to centroid
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
            coordinates = coordinates[indices,:]

        # Get dimensionless coordinates.
        x_unit = coordinates.unit
        x = coordinates / x_unit
        
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


class HarmonicReceptorLigandRestraint(ReceptorLigandRestraint):
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
    >>> temperature = 298.0 * units.kelvin
    >>> state = ThermodynamicState(temperature=temperature)
    >>> # Create restraints.
    >>> restraints = HarmonicReceptorLigandRestraint(state, system, positions, receptor_atoms, ligand_atoms)
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

        unit = self.coordinates.unit
        
        # Get dimensionless receptor coordinates.
        x = self.coordinates[self.receptor_atoms,:] / unit
        
        # Get dimensionless restrained atom coordinate.
        xref = self.coordinates[self.restrained_receptor_atom,:] / unit # (3,) array
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


class FlatBottomReceptorLigandRestraint(ReceptorLigandRestraint):
    """
    An alternative choice to receptor-ligand restraints that uses a flat potential inside most of the protein volume
    with harmonic restraining walls outside of this.

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
    >>> temperature = 298.0 * units.kelvin
    >>> state = ThermodynamicState(temperature=temperature)
    >>> # Create restraints.
    >>> restraints = FlatBottomReceptorLigandRestraint(state, system, positions, receptor_atoms, ligand_atoms)
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

        x_unit = self.coordinates.unit

        # Get dimensionless receptor coordinates.
        x = self.coordinates[self.receptor_atoms, :] / x_unit

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
            r0 = 2*sigma + 5.0 * units.angstroms
        else:
            DEFAULT_DISTANCE = 15.0 * units.angstroms
            logger.warning("Receptor only contains %d atoms; using default distance of %s" % (natoms, str(DEFAULT_DISTANCE)))
            r0 = DEFAULT_DISTANCE

        logger.debug("restraint distance r0 = %.1f A" % (r0 / units.angstroms))

        # Set spring constant/
        #K = (2.0 * 0.0083144621 * 5.0 * 298.0 * 100) * units.kilojoules_per_mole/units.nanometers**2
        K = 0.6 * units.kilocalories_per_mole / units.angstroms**2
        logger.debug("K = %.1f kcal/mol/A^2" % (K / (units.kilocalories_per_mole / units.angstroms**2)))

        # Assemble parameter vector.
        bond_parameters = [K, r0]

        return bond_parameters
