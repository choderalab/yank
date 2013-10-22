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

import os
import numpy
import copy
import time

import simtk.openmm as openmm

from sets import Set

#=============================================================================================
# AlchemicalState
#=============================================================================================

class AlchemicalState(object):
    """
    Alchemical state description.
    
    These parameters describe the parameters that affect computation of the energy.

    TODO

    * Rework these structure members into something more general and flexible?
    * Add receptor modulation back in?
    
    """
        
    def __init__(self, relativeRestraints, ligandElectrostatics, ligandLennardJones, ligandTorsions, annihilateElectrostatics=True, annihilateLennardJones=False):
        """
        Create an Alchemical state

        relativeRestraints (float)        # scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near pocket)
        ligandElectrostatics (float)      # scaling factor for ligand charges, intrinsic Born radii, and surface area term
        ligandLennardJones (float)        # scaling factor for ligand Lennard-Jones well depth and radius
        ligandTorsions (float)            # scaling factor for ligand non-ring torsions

        """

        self.annihilateElectrostatics = annihilateElectrostatics
        self.annihilateLennardJones = annihilateLennardJones

        self.relativeRestraints = relativeRestraints
        self.ligandElectrostatics = ligandElectrostatics
        self.ligandLennardJones = ligandLennardJones
        self.ligandTorsions = ligandTorsions

        return

#=============================================================================================
# AbsoluteAlchemicalFactory
#=============================================================================================

class AbsoluteAlchemicalFactory(object):
    """
    Factory for generating OpenMM System objects that have been alchemically perturbed for absolute binding free energy calculation.

    EXAMPLES
    
    Create alchemical intermediates for 'denihilating' one water in a water box.
    
    >>> # Create a reference system.
    >>> import testsystems
    >>> [reference_system, coordinates] = testsystems.WaterBox()
    >>> # Create a factory to produce alchemical intermediates.
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
    >>> # Get the default protocol for 'denihilating' in solvent.
    >>> protocol = factory.defaultSolventProtocolExplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.

    >>> # Create a reference system.
    >>> import testsystems
    >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
    >>> # Create a factory to produce alchemical intermediates.
    >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
    >>> ligand_atoms = range(2603,2621) # p-xylene
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
    >>> protocol = factory.defaultComplexProtocolImplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    """    

    # Factory initialization.
    def __init__(self, reference_system, ligand_atoms=[]):
        """
        Initialize absolute alchemical intermediate factory with reference system.

        ARGUMENTS

        reference_system (System) - reference system containing receptor and ligand
        ligand_atoms (list) - list of atoms to be designated 'ligand' -- everything else in system is considered the 'environment'
        
        """

        # Store serialized form of reference system.
        self.reference_system = copy.deepcopy(reference_system)

        # Store copy of atom sets.
        self.ligand_atoms = copy.deepcopy(ligand_atoms)
        
        # Store atom sets
        self.ligand_atomset = Set(self.ligand_atoms)

        return

    @classmethod
    def defaultComplexProtocolImplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # 0
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # 1
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # 2 fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.975, 1.00, 1.)) # 3
        alchemical_states.append(AlchemicalState(0.00, 0.95, 1.00, 1.)) # 4
        alchemical_states.append(AlchemicalState(0.00, 0.90, 1.00, 1.)) # 5
        alchemical_states.append(AlchemicalState(0.00, 0.80, 1.00, 1.)) # 6
        alchemical_states.append(AlchemicalState(0.00, 0.70, 1.00, 1.)) # 7
        alchemical_states.append(AlchemicalState(0.00, 0.60, 1.00, 1.)) # 8
        alchemical_states.append(AlchemicalState(0.00, 0.50, 1.00, 1.)) # 9
        alchemical_states.append(AlchemicalState(0.00, 0.40, 1.00, 1.)) # 10
        alchemical_states.append(AlchemicalState(0.00, 0.30, 1.00, 1.)) # 11
        alchemical_states.append(AlchemicalState(0.00, 0.20, 1.00, 1.)) # 12
        alchemical_states.append(AlchemicalState(0.00, 0.10, 1.00, 1.)) # 13
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # 14 discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.99, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.98, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.97, 1.)) #
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.96, 1.)) # 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.95, 1.)) # 15
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.925, 1.)) # 16
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.90, 1.)) # 17
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.85, 1.)) # 18
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.80, 1.)) # 19
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.75, 1.)) # 20
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.70, 1.)) # 21
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.675, 1.)) # 22
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.65, 1.)) # 23
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.60, 1.)) # 24
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.55, 1.)) # 25
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 1.)) # 26
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.40, 1.)) # 27
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.30, 1.)) # 28
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.20, 1.)) # 29
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.10, 1.)) # 30
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.05, 1.)) # 31
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.025, 1.)) # 32
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # 33 discharged, LJ annihilated
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # 34
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # 35 
        
        return alchemical_states

    @classmethod
    def defaultComplexProtocolExplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

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
        Return the default protocol for ligand in solvent.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

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
        Return the default protocol for ligand in solvent.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultSolventProtocolExplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

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
    def _alchemicallyModifyLennardJones(cls, system, nonbonded_force, alchemical_atom_indices, alchemical_state, alpha=0.50, a=1, b=1, c=1, mm=None):
        """
        Create a softcore version of the given reference force that controls only interactions between alchemically modified atoms and
        the rest of the system.

        ARGUMENTS

        system (simtk.openmm.System) - system to modify
        nonbonded_force (implements NonbondedForce API) - the NonbondedForce to modify
        alchemical_atom_indices (list of int) - atom indices to be alchemically modified
        alchemical_state (AlchemicalState)

        OPTIONAL ARGUMENTS

        annihilate (boolean) - if True, will annihilate alchemically-modified self-interactions; if False, will decouple
        alpha (float) - softcore parameter
        a, b, c (float) - parameters describing softcore force        
        mm (simtk.openmm implementation) - OpenMM implementation to use
        
        """

        if mm is None:
            mm = openmm

        # Create CustomNonbondedForce to handle softcore interactions between alchemically-modified system and rest of system.

        energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
        energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma

        if alchemical_state.annihilateLennardJones:
            # Annihilate interactions within alchemically-modified subsystem and between subsystem and environment.
            energy_expression += "lambda = lennard_jones_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);" 
        else:
            # Decouple interactions between alchemically-modified subsystem and environment only.
            energy_expression += "lambda = lennard_jones_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1)) + alchemical1*alchemical2;"
        energy_expression += "alpha = %f;" % alpha
        energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
        custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)            
        custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
        custom_nonbonded_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandLennardJones);
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
        custom_nonbonded_force.addPerParticleParameter("alchemical") # alchemical flag: 1 if this particle is alchemically modified, 0 otherwise
        system.addForce(custom_nonbonded_force)

        # Create CustomBondedForce to handle softcore exceptions if alchemically annihilating ligand.

        if alchemical_state.annihilateLennardJones:
            energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
            energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
            energy_expression += "alpha = %f;" % alpha
            energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)
            energy_expression += "lambda = lennard_jones_lambda;"
            custom_bond_force = mm.CustomBondForce(energy_expression)            
            custom_bond_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandLennardJones);
            custom_bond_force.addPerBondParameter("sigma") # Lennard-Jones sigma
            custom_bond_force.addPerBondParameter("epsilon") # Lennard-Jones epsilon
            system.addForce(custom_bond_force)

        # Copy Lennard-Jones particle parameters.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Add corresponding particle to softcore interactions.
            if particle_index in alchemical_atom_indices:
                # Turn off Lennard-Jones contribution from alchemically-modified particles.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon*0.0) 
                # Add contribution back to custom force.
                custom_nonbonded_force.addParticle([sigma, epsilon, 1])
            else:
                custom_nonbonded_force.addParticle([sigma, epsilon, 0])

        # Create an exclusion for each exception in the reference NonbondedForce, assuming that NonbondedForce will handle them.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            custom_nonbonded_force.addExclusion(iatom, jatom)

            # If annihilating Lennard-Jones, move intramolecular interactions to custom_bond_force.
            if alchemical_state.annihilateLennardJones and (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices):
                # Remove Lennard-Jones exception.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon * 0.0)
                # Add special CustomBondForce term to handle alchemically-modified Lennard-Jones exception.
                custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [mm.NonbondedForce.Ewald, mm.NonbondedForce.PME]:
            # Convert Ewald and PME to CutoffPeriodic.
            custom_nonbonded_force.setNonbondedMethod( mm.CustomNonbondedForce.CutoffPeriodic )
        else:
            custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )
        custom_nonbonded_force.setCutoffDistance( nonbonded_force.getCutoffDistance() )

        return 

    @classmethod
    def _alchemicallyModifyLennardJonesGroup(cls, system, nonbonded_force, alchemical_atom_indices, alchemical_state, alpha=0.50, a=1, b=1, c=1, mm=None):
        """
        Create a softcore version of the given reference force that controls only interactions between alchemically modified atoms and
        the rest of the system.
        
        This version uses the new group-based restriction capabilities of CustomNonbondedForce.

        ARGUMENTS

        system (simtk.openmm.System) - system to modify
        nonbonded_force (implements NonbondedForce API) - the NonbondedForce to modify
        alchemical_atom_indices (list of int) - atom indices to be alchemically modified
        alchemical_state (AlchemicalState)

        OPTIONAL ARGUMENTS

        annihilate (boolean) - if True, will annihilate alchemically-modified self-interactions; if False, will decouple
        alpha (float) - softcore parameter
        a, b, c (float) - parameters describing softcore force        
        mm (simtk.openmm implementation) - OpenMM implementation to use
        
        """

        if mm is None:
            mm = openmm

        # Create CustomNonbondedForce to handle softcore interactions between alchemically-modified system and rest of system.

        energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
        energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        energy_expression += "lambda = lennard_jones_lambda;" # lambda

        # Create atom groups.
        natoms = system.getNumParticles()
        atomset1 = Set(alchemical_atom_indices) # only alchemically-modified atoms
        atomset2 = Set(range(system.getNumParticles())) - atomset1 # all atoms minus intra-alchemical region

        energy_expression += "alpha = %f;" % alpha
        energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
        custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)            
        custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
        custom_nonbonded_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandLennardJones);
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
        system.addForce(custom_nonbonded_force)

        # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
        custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)

        # Create CustomBondedForce to handle softcore exceptions if alchemically annihilating ligand.
        if alchemical_state.annihilateLennardJones:
            energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
            energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
            energy_expression += "alpha = %f;" % alpha
            energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)
            energy_expression += "lambda = lennard_jones_lambda;"
            custom_bond_force = mm.CustomBondForce(energy_expression)            
            custom_bond_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandLennardJones);
            custom_bond_force.addPerBondParameter("sigma") # Lennard-Jones sigma
            custom_bond_force.addPerBondParameter("epsilon") # Lennard-Jones epsilon
            system.addForce(custom_bond_force)
        else:
            # Add a second CustomNonbondedForce to handle "intra-alchemical" interactions at full strength.
            energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);" 
            energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
            energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
            custom_nonbonded_force2 = mm.CustomNonbondedForce(energy_expression)            
            custom_nonbonded_force2.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
            custom_nonbonded_force2.addPerParticleParameter("sigma") # Lennard-Jones sigma
            custom_nonbonded_force2.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
            system.addForce(custom_nonbonded_force2)
            # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
            atomset1 = Set(alchemical_atom_indices) # only alchemically-modified atoms
            atomset2 = Set(alchemical_atom_indices) # only alchemically-modified atoms
            custom_nonbonded_force2.addInteractionGroup(atomset1, atomset2)

        # Copy Lennard-Jones particle parameters.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Add corresponding particle to softcore interactions.
            if particle_index in alchemical_atom_indices:
                # Turn off Lennard-Jones contribution from alchemically-modified particles.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon*0.0) 
            # Add contribution back to custom force.
            custom_nonbonded_force.addParticle([sigma, epsilon])            
            if not alchemical_state.annihilateLennardJones:
                custom_nonbonded_force2.addParticle([sigma, epsilon])

        # Create an exclusion for each exception in the reference NonbondedForce, assuming that NonbondedForce will handle them.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            custom_nonbonded_force.addExclusion(iatom, jatom)
            if not alchemical_state.annihilateLennardJones:
                custom_nonbonded_force2.addExclusion(iatom, jatom)

            # If annihilating Lennard-Jones, move intramolecular interactions to custom_bond_force.
            if alchemical_state.annihilateLennardJones and (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices):
                # Remove Lennard-Jones exception.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon * 0.0)
                # Add special CustomBondForce term to handle alchemically-modified Lennard-Jones exception.
                custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [mm.NonbondedForce.Ewald, mm.NonbondedForce.PME]:
            # Convert Ewald and PME to CutoffPeriodic.
            custom_nonbonded_force.setNonbondedMethod( mm.CustomNonbondedForce.CutoffPeriodic )
            if not alchemical_state.annihilateLennardJones:
                custom_nonbonded_force2.setNonbondedMethod( mm.CustomNonbondedForce.CutoffPeriodic )
        else:
            custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )
            if not alchemical_state.annihilateLennardJones:
                custom_nonbonded_force2.setNonbondedMethod(nonbonded_force.getNonbondedMethod() )

        custom_nonbonded_force.setCutoffDistance( nonbonded_force.getCutoffDistance() )
        if not alchemical_state.annihilateLennardJones:
            custom_nonbonded_force2.setCutoffDistance( nonbonded_force.getCutoffDistance() )

        return 

    @classmethod
    def _createCustomSoftcoreGBOBC(cls, reference_force, particle_lambdas, sasa_model='ACE', mm=None):
        """
        Create a softcore OBC GB force using CustomGBForce.

        ARGUMENTS

        reference_force (simtk.openmm.GBSAOBCForce) - reference force to use for template
        particle_lambdas (list or numpy array) - particle_lambdas[i] is the alchemical lambda for particle i, with 1.0 being fully interacting and 0.0 noninteracting

        OPTIONAL ARGUMENTS

        sasa_model (string) - solvent accessible surface area model (default: 'ACE')
        mm (simtk.openmm API) - (default: simtk.openmm)

        RETURNS

        custom (openmm.CustomGBForce) - custom GB force object
        
        """

        if mm is None:
            mm = openmm    
            
        custom = mm.CustomGBForce()

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
                                "or1 = radius1-offset; or2 = radius2-offset", mm.CustomGBForce.ParticlePairNoExclusions)

        custom.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", mm.CustomGBForce.SingleParticle)

        custom.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", mm.CustomGBForce.SingleParticle)
        if sasa_model == 'ACE':
            custom.addEnergyTerm("lambda*28.3919551*(radius+0.14)^2*(radius/B)^6", mm.CustomGBForce.SingleParticle)

        custom.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                             "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", mm.CustomGBForce.ParticlePairNoExclusions);

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

    def createPerturbedSystem(self, alchemical_state, mm=None, verbose=False):
        """
        Create a perturbed copy of the system given the specified alchemical state.

        ARGUMENTS

        alchemical_state (AlchemicalState) - the alchemical state to create from the reference system

        TODO

        * Start from a deep copy of the system, rather than building copy through Python interface.
        * isinstance(mm.NonbondedForce) and related expressions won't work if reference system was created with a different OpenMM implemnetation.

        EXAMPLES

        Create alchemical intermediates for 'denihilating' one water in a water box.
        
        >>> # Create a reference system.
        >>> import testsystems
        >>> [reference_system, coordinates] = testsystems.WaterBox()
        >>> # Create a factory to produce alchemical intermediates.
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.00, 1.00, 1.00, 1.)
        >>> # Create the perturbed system.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
        >>> # Compare energies.
        >>> import simtk.openmm as openmm
        >>> import simtk.unit as units
        >>> timestep = 1.0 * units.femtosecond
        >>> reference_integrator = openmm.VerletIntegrator(timestep)
        >>> reference_context = openmm.Context(reference_system, reference_integrator)
        >>> reference_state = reference_context.getState(getEnergy=True)
        >>> reference_potential = reference_state.getPotentialEnergy()
        >>> alchemical_integrator = openmm.VerletIntegrator(timestep)
        >>> alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
        >>> alchemical_state = alchemical_context.getState(getEnergy=True)
        >>> alchemical_potential = alchemical_state.getPotentialEnergy()
        >>> delta = alchemical_potential - reference_potential 
        >>> print delta
        0.0 kJ/mol
        
        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> import testsystems
        >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
        >>> # Compute reference potential.
        >>> timestep = 1.0 * units.femtosecond
        >>> reference_integrator = openmm.VerletIntegrator(timestep)
        >>> reference_context = openmm.Context(reference_system, reference_integrator)
        >>> reference_context.setPositions(coordinates)
        >>> reference_state = reference_context.getState(getEnergy=True)
        >>> reference_potential = reference_state.getPotentialEnergy()
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.00, 1.00, 1.00, 1.)
        >>> # Create the perturbed systems using this protocol.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
        >>> # Compare energies.        
        >>> alchemical_integrator = openmm.VerletIntegrator(timestep)
        >>> alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
        >>> alchemical_context.setPositions(coordinates)
        >>> alchemical_state = alchemical_context.getState(getEnergy=True)
        >>> alchemical_potential = alchemical_state.getPotentialEnergy()
        >>> delta = alchemical_potential - reference_potential 
        >>> print delta
        0.0 kJ/mol

        NOTES

        If lambda = 1.0 is specified for some force terms, they will not be replaced with modified forms.        

        """

        # Record timing statistics.
        initial_time = time.time()
        if verbose: print "Creating alchemically modified intermediate..."

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
                if alchemical_state.ligandLennardJones != 1.0:
                    # Create softcore Lennard-Jones interactions by modifying NonbondedForce and adding CustomNonbondedForce.                
                    #self._alchemicallyModifyLennardJones(system, force, self.ligand_atoms, alchemical_state)                
                    self._alchemicallyModifyLennardJonesGroup(system, force, self.ligand_atoms, alchemical_state)                

            elif isinstance(reference_force, openmm.GBSAOBCForce) and (alchemical_state.ligandElectrostatics != 1.0):

                # Create a CustomNonbondedForce to implement softcore interactions.
                particle_lambdas = numpy.ones([system.getNumParticles()], numpy.float32)
                particle_lambdas[self.ligand_atoms] = alchemical_state.ligandElectrostatics
                custom_force = AbsoluteAlchemicalFactory._createCustomSoftcoreGBOBC(reference_force, particle_lambdas)
                system.addForce(custom_force)
                    
            elif isinstance(reference_force, openmm.CustomExternalForce):

                force = openmm.CustomExternalForce( reference_force.getEnergyFunction() )
                for parameter_index in range(reference_force.getNumGlobalParameters()):
                    name = reference_force.getGlobalParameterName(parameter_index)
                    default_value = reference_force.getGlobalParameterDefaultValue(parameter_index)
                    force.addGlobalParameter(name, default_value)
                for parameter_index in range(reference_force.getNumPerParticleParameters()):
                    name = reference_force.getPerParticleParameterName(parameter_index)
                    force.addPerParticleParameter(name)
                for index in range(reference_force.getNumParticles()):
                    [particle_index, parameters] = reference_force.getParticleParameters(index)
                    force.addParticle(particle_index, parameters)
                system.addForce(force)

            elif isinstance(reference_force, openmm.CustomBondForce):                                

                force = openmm.CustomBondForce( reference_force.getEnergyFunction() )
                for parameter_index in range(reference_force.getNumGlobalParameters()):
                    name = reference_force.getGlobalParameterName(parameter_index)
                    default_value = reference_force.getGlobalParameterDefaultValue(parameter_index)
                    force.addGlobalParameter(name, default_value)
                for parameter_index in range(reference_force.getNumPerBondParameters()):
                    name = reference_force.getPerBondParameterName(parameter_index)
                    force.addPerBondParameter(name)
                for index in range(reference_force.getNumBonds()):
                    [particle1, particle2, parameters] = reference_force.getBondParameters(index)
                    force.addBond(particle1, particle2, parameters)
                system.addForce(force)                    

            else:                

                # Copy force without modification.
                # TODO: Can speed this up by storing serialized versions of reference forces.
                force = copy.deepcopy(reference_force)
                system.addForce(force)

        # Record timing statistics.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        if verbose: print "Elapsed time %.3f s." % (elapsed_time)
        
        return system

    def createPerturbedSystems(self, alchemical_states, verbose=False, mpicomm=None):
        """
        Create a list of perturbed copies of the system given a specified set of alchemical states.

        ARGUMENTS

        states (list of AlchemicalState) - list of alchemical states to generate
        
        OPTIONAL ARGUMENTS

        verbose (boolean) - if True, will print verbose output
        mpicomm (mpi4py communicator) - if given, will create perturbed system objects in parallel (default: None)

        RETURNS
        
        systems (list of simtk.openmm.System) - list of alchemically-modified System objects

        EXAMPLES

        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> import testsystems
        >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
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
            nstates = len(alchemical_states)
            # Divide up alchemical state creation.
            local_systems = list()
            for state_index in range(mpicomm.rank, nstates, mpicomm.size):
                alchemical_state = alchemical_states[state_index]
                if verbose: print "node %d / %d : Creating alchemical system %d / %d..." % (mpicomm.rank, mpicomm.size, state_index, len(alchemical_states))
                system = self.createPerturbedSystem(alchemical_state, verbose=verbose)
                local_systems.append(system)
            # Collect System objects from all processors.
            if verbose and (mpicomm.rank == 0): print "Collecting alchemically-modified System objects from all processes..."
            import time
            initial_time = time.time()
            gathered_systems = mpicomm.allgather(local_systems)
            systems = list()
            for state_index in range(nstates):
                source_rank = state_index % mpicomm.size # rank of process that created the system
                source_index = int(state_index / mpicomm.size) # local list index of system on process
                systems.append( gathered_systems[source_rank][source_index] )
            mpicomm.barrier()
            final_time = time.time()
            elapsed_time = final_time - initial_time
            if verbose and (mpicomm.rank == 0): print "Elapsed time %.3f s." % elapsed_time
        else:
            systems = list()
            for (state_index, alchemical_state) in enumerate(alchemical_states):            
                if verbose: print "Creating alchemical system %d / %d..." % (state_index, len(alchemical_states))
                system = self.createPerturbedSystem(alchemical_state, verbose=verbose)
                systems.append(system)

        return systems
    
    def _is_restraint(self, valence_atoms):
        """
        Determine whether specified valence term connects the ligand with its environment.

        ARGUMENTS
        
        valence_atoms (list of int) - atom indices involved in valence term (bond, angle or torsion)

        RETURNS

        True if the set of atoms includes at least one ligand atom and at least one non-ligand atom; False otherwise

        EXAMPLES
        
        Various tests.
        
        >>> # Create a reference system.
        >>> import testsystems
        >>> [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
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

        valence_atomset = Set(valence_atoms)
        intersection = Set.intersection(valence_atomset, self.ligand_atomset)
        if (len(intersection) >= 1) and (len(intersection) < len(valence_atomset)):
            return True

        return False        

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == "__main__":
    # Run doctests.
    import doctest
    doctest.testmod(verbose=False)

    
