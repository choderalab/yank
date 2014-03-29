#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Alchemical factory for free energy calculations that operates directly on OpenMM swig System objects.

DESCRIPTION

This module contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

The code in this module operates directly on OpenMM Swig-wrapped System objects for efficiency.

EXAMPLES

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

* Automatic optimization of alchemical states?
* Can we store serialized form of Force objects so that we can save time in reconstituting
  Force objects when we make copies?  We can even manipulate the XML representation directly.
* Allow protocols to automatically be resized to arbitrary number of states, to 
  allow number of states to be enlarged to be an integral multiple of number of GPUs.
* Add GBVI support to AlchemicalFactory.
* Add analytical dispersion correction to softcore Lennard-Jones, or find some other
  way to deal with it (such as simply omitting it from lambda < 1 states).
* Deep copy Force objects that don't need to be modified instead of using explicit 
  handling routines to copy data.  Eventually replace with removeForce once implemented?
* Can alchemically-modified System objects share unmodified Force objects to avoid overhead
  of duplicating Forces that are not modified?

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import numpy as np
import copy
import time

import simtk.openmm as openmm

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# AlchemicalState
#=============================================================================================

class AlchemicalState(object):
    """
    Alchemical state description.
        
    These parameters describe the parameters that affect computation of the energy.

    Attributes
    ----------
    relativeRestraints : float
        Scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near protein).     
    ligandElectrostatics : float
        Scaling factor for ligand charges, intrinsic Born radii, and surface area term.
    ligandSterics : float
        Scaling factor for ligand sterics (Lennard-Jones and Halgren) interactions.
    ligandTorsions : float
        Scaling factor for ligand non-ring torsions.
    annihilateElectrostatics : bool
        If True, electrostatics should be annihilated, rather than decoupled.
    annihilateSterics : bool
        If True, sterics (Lennard-Jones or Halgren potential) will be annihilated, rather than decoupled.

    TODO
    ----
    * Rework these structure members into something more general and flexible?
    * Add receptor modulation back in?
    """
        
    def __init__(self, relativeRestraints=0.0, ligandElectrostatics=1.0, ligandSterics=1.0, ligandTorsions=1.0, annihilateElectrostatics=True, annihilateSterics=False):
        """
        Create an Alchemical state.

        Parameters
        ----------
        relativeRestraints : float, optional, default = 0.0
            Scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near protein).     
        ligandElectrostatics : float, optional, default = 1.0
            Scaling factor for ligand charges, intrinsic Born radii, and surface area term.
        ligandSterics : float, optional, default = 1.0
            Scaling factor for ligand sterics (Lennard-Jones or Halgren) interactions.
        ligandTorsions : float, optional, default = 1.0
            Scaling factor for ligand non-ring torsions.
        annihilateElectrostatics : bool, optional, default = True
            If True, electrostatics should be annihilated, rather than decoupled.
        annihilateSterics : bool, optional, default = False
            If True, sterics (Lennard-Jones or Halgren potential) will be annihilated, rather than decoupled.

        Examples
        --------

        Create a fully-interacting, unrestrained alchemical state.
        
        >>> alchemical_state = AlchemicalState(relativeRestraints=0.0, ligandElectrostatics=1.0, ligandSterics=1.0, ligandTorsions=1.0)
        >>> # This is equivalent to
        >>> alchemical_state = AlchemicalState()


        Annihilate electrostatics.
        
        >>> alchemical_state = AlchemicalState(annihilateElectrostatics=True, ligandElectrostatics=0.0)
        
        """

        self.relativeRestraints = relativeRestraints
        self.ligandElectrostatics = ligandElectrostatics
        self.ligandSterics = ligandSterics
        self.ligandTorsions = ligandTorsions
        self.annihilateElectrostatics = annihilateElectrostatics
        self.annihilateSterics = annihilateSterics

        return

#=============================================================================================
# AbsoluteAlchemicalFactory
#=============================================================================================

class AbsoluteAlchemicalFactory(object):
    """
    Factory for generating OpenMM System objects that have been alchemically perturbed for absolute binding free energy calculation.

    Examples
    --------
    
    Create alchemical intermediates for default alchemical protocol for p-xylene in T4 lysozyme L99A in GBSA.

    >>> # Create a reference system.
    >>> from repex import testsystems
    >>> complex = testsystems.LysozymeImplicit()
    >>> [reference_system, positions] = [complex.system, complex.positions]
    >>> # Create a factory to produce alchemical intermediates.
    >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
    >>> ligand_atoms = range(2603,2621) # p-xylene
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
    >>> protocol = factory.defaultComplexProtocolImplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    Create alchemical intermediates for default alchemical protocol for one water in a water box.
    
    >>> # Create a reference system.
    >>> from repex import testsystems
    >>> waterbox = testsystems.WaterBox()
    >>> [reference_system, positions] = [waterbox.system, waterbox.positions]
    >>> # Create a factory to produce alchemical intermediates.
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
    >>> # Get the default protocol for 'denihilating' in solvent.
    >>> protocol = factory.defaultSolventProtocolExplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    """    

    # Factory initialization.
    def __init__(self, reference_system, ligand_atoms=[]):
        """
        Initialize absolute alchemical intermediate factory with reference system.

        The reference system will not be modified when alchemical intermediates are generated.

        Parmeters
        ---------
        reference_system : simtk.openmm.System
            The reference system that is to be alchemically modified.
        ligand_atoms : list of int, optional, default = []
            List of atoms to be designated as 'ligand' for alchemical modification; everything else in system is considered the 'environment'.

        """

        # Store serialized form of reference system.
        self.reference_system = copy.deepcopy(reference_system)

        # Store copy of atom sets.
        self.ligand_atoms = copy.deepcopy(ligand_atoms)
        
        # Store atom sets
        self.ligand_atomset = set(self.ligand_atoms)

        return

    @classmethod
    def defaultComplexProtocolImplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein in implicit solvent.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            The list of alchemical states defining the protocol.
        
        Notes
        -----
        The unrestrained, fully interacting system is always listed first.

        Examples
        --------

        >>> from repex import testsystems
        >>> alanine_dipeptide = testsystems.AlanineDipeptideImplicit()
        >>> [reference_system, positions] = [alanine_dipeptide.system, alanine_dipeptide.positions]
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> protocol = factory.defaultComplexProtocolImplicit()

        """

        alchemical_states = list()

        # Protocol used for SAMPL4.
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.975, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.95, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.90, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.80, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.70, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.60, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.50, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.40, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.30, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.20, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.10, 1.00, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.99999, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.99, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.98, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.97, 1.)) #
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.96, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.95, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.925, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.90, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.85, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.80, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.75, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.70, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.675, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.65, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.60, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.55, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.40, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.30, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.20, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.10, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.05, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.025, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultComplexProtocolExplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein in explicit solvent.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            The list of alchemical states defining the protocol.
        
        Notes
        -----
        The unrestrained, fully interacting system is always listed first.

        TODO
        ----
        * Update this with optimized set of alchemical states.

        Examples
        --------

        >>> from repex import testsystems
        >>> waterbox = testsystems.WaterBox()
        >>> [reference_system, positions] = [waterbox.system, waterbox.positions]
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> protocol = factory.defaultComplexProtocolExplicit()

        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.95, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.90, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.80, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.70, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.60, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.40, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.30, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.20, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.10, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultSolventProtocolImplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in imlicit solvent.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            The list of alchemical states defining the protocol.
        
        Notes
        -----
        The unrestrained, fully interacting system is always listed first.

        TODO
        ----
        * Update this with optimized set of alchemical states.

        Examples
        --------
        >>> from repex import testsystems
        >>> alanine_dipeptide = testsystems.AlanineDipeptideImplicit()
        >>> [reference_system, positions] = [alanine_dipeptide.system, alanine_dipeptide.positions]
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> protocol = factory.defaultSolventProtocolImplicit()

        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.75, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.50, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.25, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultVacuumProtocol(cls):
        """
        Return the default protocol for 'denihilating' a ligand in vacuum.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            The list of alchemical states defining the protocol.
        
        Notes
        -----
        The unrestrained, fully interacting system is always listed first.

        Examples
        --------

        >>> from repex import testsystems
        >>> alanine_dipeptide = testsystems.AlanineDipeptideVacuum()
        >>> [reference_system, positions] = [alanine_dipeptide.system, alanine_dipeptide.positions]
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> protocol = factory.defaultVacuumProtocol()

        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultSolventProtocolExplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in explicit solvent.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            The list of alchemical states defining the protocol.
        
        Notes
        -----
        The unrestrained, fully interacting system is always listed first.

        TODO
        ----
        * Update this with optimized set of alchemical states.

        Examples
        --------

        >>> from repex import testsystems
        >>> waterbox = testsystems.WaterBox()
        >>> [reference_system, positions] = [waterbox.system, waterbox.positions]
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> protocol = factory.defaultSolventProtocolExplicit()
        
        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.95, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.90, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.80, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.70, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.60, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.40, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.30, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.20, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.10, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def _alchemicallyModifyLennardJones(cls, system, nonbonded_force, alchemical_atom_indices, alchemical_state, alpha=0.50, a=1, b=1, c=6):
        """
        Alchemically modify the Lennard-Jones force terms.
        
        Parameters
        ----------
        system : simtk.openmm.System
            System to modify.
        nonbonded_force : simtk.openmm.NonbondedForce
            The NonbondedForce to modify (will be changed).
        alchemical_atom_indices : list of int
            Atom indices to be alchemically modified.
        alchemical_state : AlchemicalState
            The alchemical state specification to be used in modifying Lennard-Jones terms.
        alpha : float, optional, default = 0.5
            Alchemical softcore parameter.
        a, b, c : float, optional, default a=1, b=1, c=6
            Parameters describing softcore force.

        Details
        -------
        
        First, a CustomNonbondedForce is created to provide interactions between the alchemically-modified atoms and the unmodified remainder of the system.
        This interaction uses a softcore Lennard-Jones function that should reduce to standard Lennard-Jones at lambda = 1.
        
        """

        # Create CustomNonbondedForce to handle softcore interactions between alchemically-modified system and rest of system.
        # Create atom groups.
        natoms = system.getNumParticles()
        atomset1 = set(alchemical_atom_indices) # only alchemically-modified atoms
        atomset2 = set(range(system.getNumParticles())) - atomset1 # all atoms minus intra-alchemical region

        # Create alchemically modified nonbonded force.
        energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
        energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        energy_expression += "lambda = lennard_jones_lambda;" # lambda
        energy_expression += "alpha = %f;" % alpha
        energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)            
        custom_nonbonded_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction()) 
        custom_nonbonded_force.setCutoffDistance( nonbonded_force.getCutoffDistance() )
        custom_nonbonded_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance()) 
        custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
        custom_nonbonded_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandSterics);
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon

        # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
        # Only add custom nonbonded force if interacting groups are both nonzero in size, or else we will get a segfault during energy evaluation.
        # TODO: Remove this restriction once segfault has been fixed.
        if (len(atomset1) != 0) and (len(atomset2) != 0):
            custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)
            system.addForce(custom_nonbonded_force)

        # Create CustomBondedForce to handle intramolecular softcore exceptions if alchemically annihilating ligand.
        if alchemical_state.annihilateSterics:
            energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
            energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
            energy_expression += "alpha = %f;" % alpha
            energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)
            energy_expression += "lambda = lennard_jones_lambda;"
            custom_bond_force = openmm.CustomBondForce(energy_expression)            
            custom_bond_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandSterics);
            custom_bond_force.addPerBondParameter("sigma") # Lennard-Jones sigma
            custom_bond_force.addPerBondParameter("epsilon") # Lennard-Jones epsilon
            system.addForce(custom_bond_force)

        # Decoupling requires another force term.
        if not alchemical_state.annihilateSterics:
            # Add a second CustomNonbondedForce to restore "intra-alchemical" interactions to full strength.
            energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);" 
            energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
            energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
            custom_nonbonded_force2 = openmm.CustomNonbondedForce(energy_expression)            
            custom_nonbonded_force2.setCutoffDistance( nonbonded_force.getCutoffDistance() )
            custom_nonbonded_force2.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction()) 
            custom_nonbonded_force2.setSwitchingDistance(nonbonded_force.getSwitchingDistance()) 
            custom_nonbonded_force2.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
            custom_nonbonded_force2.addPerParticleParameter("sigma") # Lennard-Jones sigma
            custom_nonbonded_force2.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
            system.addForce(custom_nonbonded_force2)
            # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
            atomset1 = set(alchemical_atom_indices) # only alchemically-modified atoms
            atomset2 = set(alchemical_atom_indices) # only alchemically-modified atoms
            custom_nonbonded_force2.addInteractionGroup(atomset1, atomset2)

        # Copy Lennard-Jones particle parameters.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)            
            # CustomNonbondedForce will handle interactions with non-alchemical particles.
            # Add corresponding particle to softcore interactions.
            if particle_index in alchemical_atom_indices:
                # Turn off Lennard-Jones contribution from alchemically-modified particles.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon*0.0) 
            # Add contribution back to custom force.
            custom_nonbonded_force.addParticle([sigma, epsilon])            
            if not alchemical_state.annihilateSterics:
                if particle_index in alchemical_atom_indices:
                    custom_nonbonded_force2.addParticle([sigma, epsilon])
                else:
                    custom_nonbonded_force2.addParticle([sigma, 0*epsilon])

        # Create an exclusion for each exception in the reference NonbondedForce, assuming that NonbondedForce will handle them.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            custom_nonbonded_force.addExclusion(iatom, jatom)
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.addExclusion(iatom, jatom)

            # If annihilating Lennard-Jones, move intramolecular interactions to custom_bond_force.
            if alchemical_state.annihilateSterics and (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices):
                # Remove Lennard-Jones exception.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon * 0.0)
                # Add special CustomBondForce term to handle alchemically-modified Lennard-Jones exception.
                custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME, openmm.NonbondedForce.CutoffPeriodic]:
            # Convert Ewald and PME to CutoffPeriodic.
            custom_nonbonded_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
        else:
            custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.setNonbondedMethod(nonbonded_force.getNonbondedMethod() )

        return 

    @classmethod
    def _alchemicallyModifyAmoebaVdwForce(cls, system, reference_force, alchemical_atom_indices, alchemical_state):
        """
        Alchemically modify AmoebaVdwForce term.
        
        Parameters
        ----------
        cls : class
           This class.
        system : simtk.openmm.System
           The System object to use as a template for generating alchemical intermediates.
        reference_force : simtk.openmm.AmoebaVdwForce
           The AmoebaVdwForce term to create a softcore variant of.
           
        WARNING
        -------
        This version currently ignores exclusions, so should only be used on annihilating single-atom systems.

        """

        # Create a softcore force to add back modulated vdW interactions with alchemically-modified atoms.
        # Softcore Halgren potential from Eq. 3 of
        # Shi, Y., Jiao, D., Schnieders, M.J., and Ren, P. (2009). Trypsin-ligand binding free energy calculation with AMOEBA. Conf Proc IEEE Eng Med Biol Soc 2009, 2328-2331.
        energy_expression = 'lambda^5 * epsilon * (1.07^7 / (0.7*(1-lambda)^2+(rho+0.07)^7)) * (1.12 / (0.7*(1-lambda)^2 + rho^7 + 0.12) - 2);'
        energy_expression += 'epsilon = 4*epsilon1*epsilon2 / (sqrt(epsilon1) + sqrt(epsilon2))^2;'
        energy_expression += 'rho = r / R0;'
        energy_expression += 'R0 = (R01^3 + R02^3) / (R01^2 + R02^2);'
        energy_expression += 'lambda = vdw_lambda * (ligand1*(1-ligand2) + ligand2*(1-ligand1)) + ligand1*ligand2;'
        energy_expression += 'vdw_lambda = %f;' % vdw_lambda

        softcore_force = openmm.CustomNonbondedForce(energy_expression)
        softcore_force.addPerParticleParameter('epsilon')
        softcore_force.addPerParticleParameter('R0')
        softcore_force.addPerParticleParameter('ligand')

        for particle_index in range(system.getNumParticles()):
            # Retrieve parameters from vdW force.
            [parentIndex, sigma, epsilon, reductionFactor] = reference_force.getParticleParameters(particle_index)   
            # Add parameters to CustomNonbondedForce.
            if particle_index in alchemical_atom_indices:
                softcore_force.addParticle([epsilon, sigma, 1])
            else:
                softcore_force.addParticle([epsilon, sigma, 0])

            # Deal with exclusions.
            # TODO: Can we skip exclusions that aren't in the alchemically-modified system?
            excluded_atoms = force.getParticleExclusions(particle_index)
            for jatom in excluded_atoms:
                if (particle_index < jatom):
                    softcore_force.addExclusion(particle_index, jatom)

        # Make sure periodic boundary conditions are treated the same way.
        if reference_force.getPBC():
            softcore_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
        else:
            softcore_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffNonperiodic )
        softcore_force.setCutoffDistance( force.getCutoff() )

        # Add the softcore force.
        system.addForce( softcore_force )

        # Turn off vdW interactions for alchemically-modified atoms.
        for particle_index in alchemical_atom_indices:
            # Retrieve parameters.
            [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
            epsilon = 1.0e-6 * epsilon # TODO: Can we use zero?
            force.setParticleParameters(particle_index, parentIndex, sigma, epsilon, reductionFactor)

        return 

    @classmethod
    def _createCustomSoftcoreGBOBC(cls, reference_force, particle_lambdas, sasa_model='ACE'):
        """
        Create a softcore OBC GBSA force using CustomGBForce.

        Parameters
        ----------
        reference_force : simtk.openmm.GBSAOBCForce
            Reference force to use for template (will not be modified).
        particle_lambdas : list (or numpy array) of float
            particle_lambdas[i] is the alchemical lambda for particle i, with 1.0 being fully interacting and 0.0 noninteracting
        sasa_model : str, optional, default='ACE'
            Solvent accessible surface area model.

        Returns
        -------
        custom : simtk.openmm.CustomGBForce
            Custom GB force object.
        
        """

        custom = openmm.CustomGBForce()

        # Add per-particle parameters.
        custom.addPerParticleParameter("q");
        custom.addPerParticleParameter("radius");
        custom.addPerParticleParameter("scale");
        custom.addPerParticleParameter("lambda");
        
        # Set nonbonded method.
        custom.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom.setCutoffDistance(reference_force.getCutoffDistance())

        # Add global parameters.
        custom.addGlobalParameter("solventDielectric", reference_force.getSolventDielectric())
        custom.addGlobalParameter("soluteDielectric", reference_force.getSoluteDielectric())
        custom.addGlobalParameter("offset", 0.009)

        custom.addComputedValue("I",  "lambda2*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
                                "U=r+sr2;"
                                "L=max(or1, D);"
                                "D=abs(r-sr2);"
                                "sr2 = scale2*or2;"
                                "or1 = radius1-offset; or2 = radius2-offset", openmm.CustomGBForce.ParticlePairNoExclusions)

        custom.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", openmm.CustomGBForce.SingleParticle)

        custom.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", openmm.CustomGBForce.SingleParticle)
        if sasa_model == 'ACE':
            custom.addEnergyTerm("lambda*28.3919551*(radius+0.14)^2*(radius/B)^6", openmm.CustomGBForce.SingleParticle)

        custom.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                             "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", openmm.CustomGBForce.ParticlePairNoExclusions);

        # Add particle parameters.
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
            lambda_factor = float(particle_lambdas[particle_index])
            # Set particle parameters.
            # Note that charges must be scaled by lambda_factor in this representation.
            parameters = [charge * lambda_factor, radius, scaling_factor, lambda_factor]
            custom.addParticle(parameters)

        return custom


    def createPerturbedSystem(self, alchemical_state, mm=None):
        """
        Create a perturbed copy of the system given the specified alchemical state.

        Parameters
        ----------
        alchemical_state : AlchemicalState
            The alchemical state to create from the reference system.

        TODO
        ----
        * This could be streamlined if it was possible to modify System or Force objects.
        * isinstance(mm.NonbondedForce) and related expressions won't work if reference system was created with a different OpenMM implementation. 
          Use class names instead.

        Examples
        --------

        Create alchemical intermediates for 'denihilating' one water in a water box.
        
        >>> # Create a reference system.
        >>> from repex import testsystems
        >>> waterbox = testsystems.WaterBox()
        >>> [reference_system, positions] = [waterbox.system, waterbox.positions]
        >>> # Create a factory to produce alchemical intermediates.
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.00, 1.00, 1.00, 1.00)
        >>> # Create the perturbed system.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
        
        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> from repex import testsystems
        >>> complex = testsystems.LysozymeImplicit()
        >>> [reference_system, positions] = [complex.system, complex.positions]
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.00, 1.00, 1.00, 1.)
        >>> # Create the perturbed systems using this protocol.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)

        NOTES

        If lambda = 1.0 is specified for some force terms, they will not be replaced with modified forms.        

        """

        # Record timing statistics.
        initial_time = time.time()
        logger.debug("Creating alchemically modified intermediate...")

        reference_system = self.reference_system

        # Create new deep copy reference system to modify.
        system = openmm.System()
        
        # Set periodic box vectors.
        [a,b,c] = reference_system.getDefaultPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(a,b,c)
        
        # Add atoms.
        for atom_index in range(reference_system.getNumParticles()):
            mass = reference_system.getParticleMass(atom_index)
            system.addParticle(mass)

        # Add constraints
        for constraint_index in range(reference_system.getNumConstraints()):
            [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
            system.addConstraint(iatom, jatom, r0)    

        # Modify forces as appropriate, copying other forces without modification.
        nforces = reference_system.getNumForces()
        for force_index in range(nforces):
            reference_force = reference_system.getForce(force_index)

            if isinstance(reference_force, openmm.PeriodicTorsionForce):
                # PeriodicTorsionForce
                force = openmm.PeriodicTorsionForce()
                for torsion_index in range(reference_force.getNumTorsions()):
                    # Retrieve parmaeters.
                    [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                    # Scale torsion barrier of alchemically-modified system.
                    if set([particle1,particle2,particle3,particle4]).issubset(self.ligand_atomset):
                        k *= alchemical_state.ligandTorsions
                    force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
                system.addForce(force)

            elif isinstance(reference_force, openmm.AmoebaMultipoleForce):
                # AmoebaMultipoleForce
                force = openmm.AmoebaMultipoleForce
                for particle_index in self.ligand_atomset:
                    # Retrieve parameters.
                    [charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, dampingFactor, polarizibility] = reference_force.getMultipoleParameters(particle_index)
                    # Modify parameters.
                    charge = charge * coulomb_lambda
                    molecularDipole = coulomb_lambda * numpy.array(molecularDipole)
                    molecularQuadrupole = coulomb_lambda * numpy.array(molecularQuadrupole)
                    polarizibility *= coulomb_lambda                
                    force.addParticle(charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, dampingFactor, polarizibility)
                system.addForce(force)

            elif isinstance(reference_force, openmm.AmoebaVdwForce):
                
                # Modify AmoebaVdwForce.
                self._alchemicallyModifyAmoebaVdwForce(system, force, self.ligand_atoms, alchemical_state)                

            elif isinstance(reference_force, openmm.NonbondedForce):

                # Copy NonbondedForce.
                force = copy.deepcopy(reference_force)
                system.addForce(force)
                
                # Modify electrostatics.
                if alchemical_state.ligandElectrostatics != 1.0:
                    for particle_index in range(force.getNumParticles()):
                        # Retrieve parameters.
                        [charge, sigma, epsilon] = force.getParticleParameters(particle_index)
                        # Alchemically modify charges.
                        if particle_index in self.ligand_atomset:
                            charge *= alchemical_state.ligandElectrostatics
                        # Set modified particle parameters.
                        force.setParticleParameters(particle_index, charge, sigma, epsilon)
                    for exception_index in range(force.getNumExceptions()):
                        # Retrieve parameters.
                        [iatom, jatom, chargeprod, sigma, epsilon] = force.getExceptionParameters(exception_index)
                        # Alchemically modify chargeprod.
                        if (iatom in self.ligand_atomset) and (jatom in self.ligand_atomset):
                            if alchemical_state.annihilateElectrostatics:
                                chargeprod *= alchemical_state.ligandElectrostatics**2
                        # Set modified exception parameters.
                        force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)

                # Modify Lennard-Jones if required.
                if alchemical_state.ligandSterics != 1.0:
                    # Create softcore Lennard-Jones interactions by modifying NonbondedForce and adding CustomNonbondedForce.                
                    self._alchemicallyModifyLennardJones(system, force, self.ligand_atoms, alchemical_state)                
                    
            elif isinstance(reference_force, openmm.GBSAOBCForce) and (alchemical_state.ligandElectrostatics != 1.0):

                # Create a CustomNonbondedForce to implement softcore interactions.
                particle_lambdas = np.ones([system.getNumParticles()], np.float32)
                particle_lambdas[self.ligand_atoms] = alchemical_state.ligandElectrostatics
                custom_force = AbsoluteAlchemicalFactory._createCustomSoftcoreGBOBC(reference_force, particle_lambdas)
                system.addForce(custom_force)
                    
#            elif isinstance(reference_force, openmm.CustomExternalForce):
#
#                force = openmm.CustomExternalForce( reference_force.getEnergyFunction() )
#                for parameter_index in range(reference_force.getNumGlobalParameters()):
#                    name = reference_force.getGlobalParameterName(parameter_index)
#                    default_value = reference_force.getGlobalParameterDefaultValue(parameter_index)
#                    force.addGlobalParameter(name, default_value)
#                for parameter_index in range(reference_force.getNumPerParticleParameters()):
#                    name = reference_force.getPerParticleParameterName(parameter_index)
#                    force.addPerParticleParameter(name)
#                for index in range(reference_force.getNumParticles()):
#                    [particle_index, parameters] = reference_force.getParticleParameters(index)
#                    force.addParticle(particle_index, parameters)
#                system.addForce(force)

#            elif isinstance(reference_force, openmm.CustomBondForce):                                
#
#                force = openmm.CustomBondForce( reference_force.getEnergyFunction() )
#                for parameter_index in range(reference_force.getNumGlobalParameters()):
#                    name = reference_force.getGlobalParameterName(parameter_index)
#                    default_value = reference_force.getGlobalParameterDefaultValue(parameter_index)
#                    force.addGlobalParameter(name, default_value)
#                for parameter_index in range(reference_force.getNumPerBondParameters()):
#                    name = reference_force.getPerBondParameterName(parameter_index)
#                    force.addPerBondParameter(name)
#                for index in range(reference_force.getNumBonds()):
#                    [particle1, particle2, parameters] = reference_force.getBondParameters(index)
#                    force.addBond(particle1, particle2, parameters)
#                system.addForce(force)                    

            else:                
                # Copy force without modification.
                # TODO: Can speed this up by storing serialized versions of reference forces.
                force = copy.deepcopy(reference_force)
                system.addForce(force)
                
        # Record timing statistics.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("Elapsed time %.3f s." % (elapsed_time))
        
        return system

    def createPerturbedSystems(self, alchemical_states, mpicomm=None):
        """
        Create a list of perturbed copies of the system given a specified set of alchemical states.

        Parameters
        ----------
        states : list of AlchemicalState
            List of alchemical states to generate.
        mpicomm : mpi4py communicator, optional, default=None
            If given, will create perturbed system objects in parallel and reduce results to all nodes.

        Returns
        -------        
        systems : list of simtk.openmm.System
            List of alchemically-modified System objects.  The cached reference system will be unmodified.

        TODO
        ----
        Remove MPI code path if there is no performance improvement.

        Examples
        --------
            
        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> from repex import testsystems
        >>> complex = testsystems.LysozymeImplicit()
        >>> [reference_system, positions] = [complex.system, complex.positions]
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
        >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
        >>> protocol = factory.defaultComplexProtocolImplicit()
        >>> # Create the perturbed systems using this protocol.
        >>> systems = factory.createPerturbedSystems(protocol)        
        
        """

        if mpicomm:
            initial_time = time.time()

            nstates = len(alchemical_states)
            # Divide up alchemical state creation.
            local_systems = list()
            for state_index in range(mpicomm.rank, nstates, mpicomm.size):
                alchemical_state = alchemical_states[state_index]
                logger.debug("node %d / %d : Creating alchemical system %d / %d..." % (mpicomm.rank, mpicomm.size, state_index, len(alchemical_states)))
                system = self.createPerturbedSystem(alchemical_state)
                local_systems.append(system)
            # Collect System objects from all processors.
            if (mpicomm.rank == 0):
                logger.debug("Collecting alchemically-modified System objects from all processes...")
            gathered_systems = mpicomm.allgather(local_systems)
            systems = list()
            for state_index in range(nstates):
                source_rank = state_index % mpicomm.size # rank of process that created the system
                source_index = int(state_index / mpicomm.size) # local list index of system on process
                systems.append( gathered_systems[source_rank][source_index] )
            mpicomm.barrier()

            # Report timing.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            if (mpicomm.rank == 0):
                logger.debug("createPerturbedSystems: Elapsed time %.3f s." % elapsed_time)
        else:
            initial_time = time.time()

            systems = list()
            for (state_index, alchemical_state) in enumerate(alchemical_states):            
                logger.debug("Creating alchemical system %d / %d..." % (state_index, len(alchemical_states)))
                system = self.createPerturbedSystem(alchemical_state)
                systems.append(system)

            # Report timing.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            logger.debug("createPerturbedSystems: Elapsed time %.3f s." % elapsed_time)

        return systems
    
    def _is_restraint(self, valence_atoms):
        """
        Determine whether specified valence term connects the ligand with its environment.

        Parameters
        ----------
        valence_atoms : list of int
            Atom indices involved in valence term (bond, angle or torsion).
            
        Returns
        -------
        is_restraint : bool
            True if the set of atoms includes at least one ligand atom and at least one non-ligand atom; False otherwise

        Examples
        --------
        
        Various tests for a simple system.
        
        >>> # Create a reference system.
        >>> from repex import testsystems
        >>> alanine_dipeptide = testsystems.AlanineDipeptideImplicit()
        >>> [reference_system, positions] = [alanine_dipeptide.system, alanine_dipeptide.positions]
        >>> # Create a factory.
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> factory._is_restraint([0,1,2])
        False
        >>> factory._is_restraint([1,2,3])
        True
        >>> factory._is_restraint([3,4])
        False
        >>> factory._is_restraint([2,3,4,5])
        True

        """

        valence_atomset = set(valence_atoms)
        intersection = set.intersection(valence_atomset, self.ligand_atomset)
        if (len(intersection) >= 1) and (len(intersection) < len(valence_atomset)):
            return True

        return False        
