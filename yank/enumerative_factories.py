#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Enumerative factories for generating chemically or alchemically-modified System objects for free energy calculations.

DESCRIPTION

This module contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

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

* Add GBVI support to AlchemicalFactory.
* Add analytical dispersion correction to softcore Lennard-Jones.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import numpy
import copy
import time

import simtk.openmm

import pyopenmm

from sets import Set

#=============================================================================================
# EnumerativeFactory
#=============================================================================================

class EnumerativeFactory(object):
    """
    Abstract base class for enumerative factories.
    
    """
    pass

#=============================================================================================
# AlchemicalState
#=============================================================================================

class AlchemicalState(object):
    """
    Alchemical state description.
    
    These parameters describe the parameters that affect computation of the energy.

    TODO

    * Rework these structure members into something more general and flexible?
    
    """
        
    def __init__(self, relativeDistanceRestraint, relativeRestraints, ligandElectrostatics, ligandLennardJones, ligandTorsions, receptorElectrostatics, receptorLennardJones, receptorTorsions):
        """
        Create an Alchemical state

        relativeDistanceRestraint (float) # scaling factor for receptor-ligand relative distance restraint (needed for standard state correction)
        relativeRestraints (float)        # scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near pocket)
        ligandElectrostatics (float)      # scaling factor for ligand charges, intrinsic Born radii, and surface area term
        ligandLennardJones (float)        # scaling factor for ligand Lennard-Jones well depth and radius
        ligandTorsions (float)            # scaling factor for ligand non-ring torsions
        receptorElectrostatics (float)    # scaling factor for tagged receptor charges
        receptorLennardJones (float)      # scaling factor for tagged receptor Lennard-Jones well depth and radius
        receptorTorsions (float)          # scaling factor for tagged receptor torsions

        """

        self.annihilateElectrostatics = True
        self.annihilateLennardJones = False

        self.relativeDistanceRestraint = relativeDistanceRestraint
        self.relativeRestraints = relativeRestraints
        self.ligandElectrostatics = ligandElectrostatics
        self.ligandLennardJones = ligandLennardJones
        self.ligandTorsions = ligandTorsions
        self.receptorElectrostatics = receptorElectrostatics
        self.receptorLennardJones = receptorLennardJones
        self.receptorTorsions = receptorTorsions

        return

#=============================================================================================
# AbsoluteAlchemicalFactory
#=============================================================================================

class AbsoluteAlchemicalFactory(EnumerativeFactory):
    """
    Factory for generating OpenMM System objects that have been alchemically perturbed for absolute binding free energy calculation.

    EXAMPLES
    
    Create alchemical intermediates for 'denihilating' one water in a water box.
    
    >>> # Create a reference system.
    >>> from simtk.pyopenmm.extras import testsystems
    >>> [reference_system, coordinates] = testsystems.WaterBox()
    >>> # Create a factory to produce alchemical intermediates.
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
    >>> # Get the default protocol for 'denihilating' in solvent.
    >>> protocol = factory.defaultSolventProtocolExplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.

    >>> # Create a reference system.
    >>> from simtk.pyopenmm.extras import testsystems
    >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
    >>> # Create a factory to produce alchemical intermediates.
    >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
    >>> ligand_atoms = range(2603,2621) # p-xylene
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms, receptor_atoms=receptor_atoms)
    >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
    >>> protocol = factory.defaultComplexProtocolImplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    """    

    # Factory initialization.
    def __init__(self, reference_system, receptor_atoms=[], ligand_atoms=[]):
        """
        Initialize absolute alchemical intermediate factory with reference system.

        ARGUMENTS

        reference_system (System) - reference system containing receptor and ligand
        ligand_atoms (list) - list of atoms to be designated 'ligand' -- everything else in system is considered the 'environment'
        receptor_atoms (list) - list of atoms to be considered in softening specific 'receptor' degrees of freedom -- shouldn't be the whole receptor, but a subset of atoms in binding site
        
        """

        # Create pyopenmm System object.
        self.reference_system = pyopenmm.System(reference_system)

        # Store copy of atom sets.
        self.receptor_atoms = copy.deepcopy(receptor_atoms)
        self.ligand_atoms = copy.deepcopy(ligand_atoms)
        
        # Store atom sets
        self.ligand_atomset = Set(self.ligand_atoms)
        self.receptor_atomset = Set(self.receptor_atoms)

        # Make sure intersection of ligand and receptor atomsets is null.
        intersection = Set.intersection(self.ligand_atomset, self.receptor_atomset)
        if (len(intersection) > 0):
            raise ParameterException("receptor and ligand atomsets must not overlap.")
        
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

        alchemical_states.append(AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)) # no restraints, fully interacting
        alchemical_states.append(AlchemicalState(1.0, 1.00, 1.00, 1.00, 1., 1., 1., 1.)) # full restraints, fully interacting
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 1.00, 1., 1., 1., 1.)) # full restraints, discharged
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.95, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.90, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.80, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.70, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.60, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.50, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.40, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.30, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.20, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.10, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.00, 1., 1., 1., 1.)) # full restraints, discharged, LJ annihilated
        alchemical_states.append(AlchemicalState(1.0, 0.50, 0.00, 0.00, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 0.10, 0.00, 0.00, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 0.00, 0.00, 0.00, 1., 1., 1., 1.)) # only distance restraint, discharged and annihilated
        
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

        alchemical_states.append(AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)) # no restraints, fully interacting
        alchemical_states.append(AlchemicalState(1.0, 1.00, 1.00, 1.00, 1., 1., 1., 1.)) # full restraints, fully interacting
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 1.00, 1., 1., 1., 1.)) # full restraints, discharged
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.95, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.90, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.80, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.70, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.60, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.50, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.40, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.30, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.20, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.10, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.00, 1., 1., 1., 1.)) # full restraints, discharged, LJ annihilated
        alchemical_states.append(AlchemicalState(1.0, 0.50, 0.00, 0.00, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 0.10, 0.00, 0.00, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 0.00, 0.00, 0.00, 1., 1., 1., 1.)) # only distance restraint, discharged and annihilated
        
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

        alchemical_states.append(AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.0, 0.00, 0.00, 1.00, 1., 1., 1., 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.0, 0.00, 0.00, 0.00, 1., 1., 1., 1.)) # discharged, LJ annihilated
        
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

        alchemical_states.append(AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.0, 0.00, 0.00, 1.00, 1., 1., 1., 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.0, 0.00, 0.00, 0.00, 1., 1., 1., 1.)) # discharged, LJ annihilated
        
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

        alchemical_states.append(AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)) # no restraints, fully interacting
        alchemical_states.append(AlchemicalState(1.0, 1.00, 1.00, 1.00, 1., 1., 1., 1.)) # full restraints, fully interacting
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 1.00, 1., 1., 1., 1.)) # full restraints, discharged
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.95, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.90, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.80, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.70, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.60, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.50, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.40, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.30, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.20, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.10, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 1.00, 0.00, 0.00, 1., 1., 1., 1.)) # full restraints, discharged, LJ annihilated
        alchemical_states.append(AlchemicalState(1.0, 0.50, 0.00, 0.00, 1., 1., 1., 1.))
        alchemical_states.append(AlchemicalState(1.0, 0.10, 0.00, 0.00, 1., 1., 1., 1.)) 
        alchemical_states.append(AlchemicalState(1.0, 0.00, 0.00, 0.00, 1., 1., 1., 1.)) # only distance restraint, discharged and annihilated
        
        return alchemical_states

    @classmethod
    def _createSoftcoreForce(cls, reference_force, particle_lambdas, alpha=0.25, a=1., b=1., c=12.):
        """
        Create a softcore version of the given reference force.

        ARGUMENTS

        reference_force (implements NonbondedForce API) - the reference force to create a softcore version of
        particle_lambdas (numpy array) - particle_lambdas[particle_index] is the softcore lambda value of particle 'particle_index'

        OPTIONAL ARGUMENTS

        alpha (float) - softcore parameter
        a, b, c (float) - parameters describing softcore force        

        RETURNS

        force - CustomNonbondedForce that implements softcore Lennard-Jones
        
        """

        # Create CustomNonbondedForce to handle softcore interactions.

        energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
        energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        energy_expression += "lambda = lambda1*lambda2;" # mixing rule for lambda
        
        force = pyopenmm.CustomNonbondedForce(energy_expression)            

        force.addGlobalParameter("alpha", alpha);
        force.addGlobalParameter("a", a)
        force.addGlobalParameter("b", b)
        force.addGlobalParameter("c", c)

        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon")
        force.addPerParticleParameter("lambda")

        # Copy Lennard-Jones particle parameters.
        import simtk.unit as units
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
            # Alchemically modify parameters.
            parameters = [sigma, epsilon, float(particle_lambdas[particle_index])]
            force.addParticle(parameters)

        # Create an exclusion for each exception in the reference NonbondedForce, assuming that NonbondedForce will handle them.
        for exception_index in range(reference_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            force.addExclusion(iatom, jatom)

        # Set periodicity and cutoff parameters.
        if reference_force.getNonbondedMethod() in [pyopenmm.NonbondedForce.Ewald, pyopenmm.NonbondedForce.PME]:
            # Convert Ewald and PME to CutoffPeriodic.
            print "Converting Ewald or PME to CutoffPeriodic for Lennard-Jones..."
            force.setNonbondedMethod( pyopenmm.CustomNonbondedForce.CutoffPeriodic )
        else:
            force.setNonbondedMethod( reference_force.getNonbondedMethod() )
        force.setCutoffDistance( reference_force.getCutoffDistance() )

        return force

    @classmethod
    def _createCustomSoftcoreGBOBC(cls, reference_force, particle_lambdas, igb=5, mm=pyopenmm):
        """
        Create a softcore OBC GB force using CustomGBForce.

        ARGUMENTS

        reference_force (simtk.openmm.GBSAOBCForce) - reference force to use for template
        particle_lambdas (list or numpy array) - particle_lambdas[i] is the alchemical lambda for particle i, with 1.0 being fully interacting and 0.0 noninteracting

        OPTIONAL ARGUMENTS

        igb (int) - AMBER 'igb' equivalent, either 2 or 5 (default: 5)

        RETURNS

        custom (pyopenmm.CustomGBForce) - custom GB force object
        
        """

        custom_force = mm.CustomGBForce()

        custom_force.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom_force.setCutoffDistance(reference_force.getCutoffDistance())

        custom_force.addGlobalParameter("solventDielectric", reference_force.getSolventDielectric());
        custom_force.addGlobalParameter("soluteDielectric", reference_force.getSoluteDielectric());
        custom_force.addGlobalParameter("offset", 0.009)

        custom_force.addPerParticleParameter("q");
        custom_force.addPerParticleParameter("radius");
        custom_force.addPerParticleParameter("scale");
        custom_force.addPerParticleParameter("lambda");

        
        method = 'Eastman'
        if method == 'Klein':
            # From Christoph Klein
            custom_force.addComputedValue("I",  "lambda1*lambda2*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);" 
                                          "U=r+sr2;"
                                          "L=max(or1, D);"
                                          "D=abs(r-sr2);"
                                          "sr2 = scale2*or2;"
                                          "or1 = radius1-offset; or2 = radius2-offset", mm.CustomGBForce.ParticlePairNoExclusions)
        else:
            # From Peter Eastman
            custom_force.addComputedValue("I", "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);" 
                                          "U=r+sr2;"
                                          "C=2*(1/or1-1/L)*step(sr2-r-or1);"
                                          "L=max(or1, D);"
                                          "D=abs(r-sr2);"
                                          "sr2 = scale2*or2;"
                                          "or1 = radius1-0.009; or2 = radius2-0.009", mm.CustomGBForce.ParticlePairNoExclusions);

        if igb == 2:
            custom_force.addComputedValue("B", "1/(1/or-tanh(0.8*psi+2.909125*psi^3)/radius);"
                                          "psi=I*or; or=radius-offset", mm.CustomGBForce.SingleParticle)
        elif igb == 5:
            custom_force.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                          "psi=I*or; or=radius-offset", mm.CustomGBForce.SingleParticle)            
        else:
            raise Exception("igb must be 2 or 5")

        custom_force.addEnergyTerm("lambda*28.3919551*(radius+0.14)^2*(radius/B)^6-lambda*0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", mm.CustomGBForce.SingleParticle);
        custom_force.addEnergyTerm("-138.935485*lambda1*lambda2*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                             "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", mm.CustomGBForce.ParticlePairNoExclusions);        

        # Add particle parameters.
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
            # Set particle parameters.
            parameters = [charge, radius, scaling_factor, float(particle_lambdas[particle_index])]
            custom_force.addParticle(parameters)

        return custom_force

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
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.WaterBox()
        >>> # Create a factory to produce alchemical intermediates.
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)
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
        >>> alchemical_integrator = pyopenmm.VerletIntegrator(timestep)
        >>> alchemical_context = pyopenmm.Context(alchemical_system, alchemical_integrator)
        >>> alchemical_state = alchemical_context.getState(getEnergy=True)
        >>> alchemical_potential = alchemical_state.getPotentialEnergy()
        >>> delta = alchemical_potential - reference_potential 
        >>> print delta
        
        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms, receptor_atoms=receptor_atoms)
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)
        >>> # Create the perturbed systems using this protocol.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
        >>> # Compare energies.        
        >>> timestep = 1.0 * units.femtosecond
        >>> reference_integrator = openmm.VerletIntegrator(timestep)
        >>> reference_context = openmm.Context(reference_system, reference_integrator)
        >>> reference_context.setPositions(coordinates)
        >>> reference_state = reference_context.getState(getEnergy=True)
        >>> reference_potential = reference_state.getPotentialEnergy()
        >>> alchemical_integrator = pyopenmm.VerletIntegrator(timestep)
        >>> alchemical_context = pyopenmm.Context(alchemical_system, alchemical_integrator)
        >>> alchemical_context.setPositions(coordinates)
        >>> alchemical_state = alchemical_context.getState(getEnergy=True)
        >>> alchemical_potential = alchemical_state.getPotentialEnergy()
        >>> delta = alchemical_potential - reference_potential 
        >>> print delta

        NOTES

        If lambda = 1.0 is specified for some force terms, they will not be replaced with modified forms.        

        """

        # Record timing statistics.
        initial_time = time.time()
        if verbose: print "Creating alchemically modified intermediate..."

        # Create deep copy of pure Python system.
        system = copy.deepcopy(self.reference_system)
        deepcopy_time = time.time()
        
        # Modify forces as appropriate.
        forces = [force for force in system.forces] # TODO: More elegant way to only deal with original forces, so we don't modify new forces we create?
        for force in forces:
            if isinstance(force, pyopenmm.PeriodicTorsionForce):
                # PeriodicTorsionForce
                for torsion_index in range(force.getNumTorsions()):
                    # Retrieve parmaeters.
                    [particle1, particle2, particle3, particle4, periodicity, phase, k] = force.getTorsionParameters(torsion_index)
                    # TODO: Deal with torsion annihilation?                    
                    if set([particle1,particle2,particle3,particle4]).issubset(self.ligand_atomset):
                        force.setTorsionParameters(torsion_index, particle1, particle2, particle3, particle4, periodicity, phase, k*alchemical_state.ligandTorsions)
            elif isinstance(force, pyopenmm.NonbondedForce):
                # Don't modify if all lambdas are 1.
                if (alchemical_state.ligandLennardJones==1.0) and (alchemical_state.receptorLennardJones==1.0) and (alchemical_state.ligandElectrostatics==1.0) and (alchemical_state.receptorElectrostatics==1.0):
                    continue
                
                # Create a softcore version of the NonbondedForce by modifying NonbondedForce and adding CustomNonbondedForce.                
                # Create a CustomNonbondedForce to implement softcore interactions.
                particle_lambdas = numpy.ones([system.nparticles], numpy.float32)
                particle_lambdas[self.ligand_atoms] = alchemical_state.ligandLennardJones
                particle_lambdas[self.receptor_atoms] = alchemical_state.receptorLennardJones
                custom_force = AbsoluteAlchemicalFactory._createSoftcoreForce(force, particle_lambdas)
                system.addForce(custom_force)
                # TODO: We need to add back in the analytical dispersion correction.
                # NonbondedForce will handle charges and exception interactions.
                for particle_index in range(force.getNumParticles()):
                    # Retrieve parameters.
                    [charge, sigma, epsilon] = force.getParticleParameters(particle_index)
                    # Remove Lennard-Jones interactions, which will be handled by CustomNonbondedForce.
                    epsilon *= 0.0
                    # Alchemically modify charges.
                    if particle_index in self.ligand_atomset:
                        charge *= alchemical_state.ligandElectrostatics
                    # Set modified particle parameters.
                    force.setParticleParameters(particle_index, charge, sigma, epsilon)
                for exception_index in range(force.getNumExceptions()):
                    # Retrieve parameters.
                    [iatom, jatom, chargeprod, sigma, epsilon] = force.getExceptionParameters(exception_index)
                    # Alchemically modify epsilon and chargeprod.
                    # Note that exceptions are handled by NonbondedForce and not CustomNonbondedForce.
                    if (iatom in self.ligand_atomset) and (jatom in self.ligand_atomset):
                        if alchemical_state.annihilateLennardJones:
                            epsilon *= alchemical_state.ligandLennardJones
                        if alchemical_state.annihilateElectrostatics:
                            chargeprod *= alchemical_state.ligandElectrostatics
                    # Set modified exception parameters.
                    force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)
            elif isinstance(force, pyopenmm.GBSAOBCForce):
                # Don't modify if all lambdas are 1.
                if (alchemical_state.ligandElectrostatics==1.0) and (alchemical_state.receptorElectrostatics==1.0):
                    continue

                # Create a CustomNonbondedForce to implement softcore interactions.
                particle_lambdas = numpy.ones([system.nparticles], numpy.float32)
                particle_lambdas[self.ligand_atoms] = alchemical_state.ligandElectrostatics
                particle_lambdas[self.receptor_atoms] = alchemical_state.receptorElectrostatics
                custom_force = AbsoluteAlchemicalFactory._createCustomSoftcoreGBOBC(force, particle_lambdas, igb=5)
                system.addForce(custom_force)
                system.removeForce(force)
            else:
                # Don't modify unrecognized forces.
                pass
        alchemical_time = time.time()
        
        # Return Swig version of system.
        swig_system = system.asSwig()
        asswig_time = time.time()
        
        # Record timing statistics.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        if verbose: print "Elapsed time %.3f s (deepcopy %.3f s, alchemical modifications %.3f s, asSwig %.3f s)" % (elapsed_time, deepcopy_time - initial_time, alchemical_time - deepcopy_time, asswig_time - alchemical_time)
        
        return swig_system

    def createPerturbedSystems(self, alchemical_states, verbose=False):
        """
        Create a list of perturbed copies of the system given a specified set of alchemical states.

        ARGUMENTS

        states (list of AlchemicalState) - list of alchemical states to generate
        
        RETURNS
        
        systems (list of simtk.openmm.System) - list of alchemically-modified System objects

        EXAMPLES

        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms, receptor_atoms=receptor_atoms)
        >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
        >>> protocol = factory.defaultComplexProtocolImplicit()
        >>> # Create the perturbed systems using this protocol.
        >>> systems = factory.createPerturbedSystems(protocol)        
        
        """

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
        >>> from simtk.pyopenmm.extras import testsystems
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
# MAIN AND UNIT TESTS
#=============================================================================================

def testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms):
    """
    Compare energies of reference system and fully-interacting alchemically modified system.

    TODO: We need to change testing strategy if we use lambda = 1.0, since createPerturbedSystem returns original Force objects in this case.

    """

    # Create a factory to produce alchemical intermediates.
    print "Creating alchemical factory..."
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)

    # Create an alchemically-perturbed state corresponding to fully-interacting.
    #alchemical_state = AlchemicalState(0.0, 0.00, 1.00, 1.00, 1., 1., 1., 1.)
    lambda_value = 1.0 - 1.0e-6
    alchemical_state = AlchemicalState(0.0, 0.00, lambda_value, lambda_value, lambda_value, lambda_value, lambda_value, lambda_value)

    # Create the perturbed system.
    print "Creating alchemically-modified state..."
    alchemical_system = factory.createPerturbedSystem(alchemical_state)    
    # Compare energies.
    import simtk.unit as units
    import simtk.openmm as openmm
    timestep = 1.0 * units.femtosecond
    print "Computing reference energies..."
    reference_integrator = openmm.VerletIntegrator(timestep)
    reference_context = openmm.Context(reference_system, reference_integrator)
    reference_context.setPositions(coordinates)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()    
    print "Computing alchemical energies..."
    alchemical_integrator = simtk.openmm.VerletIntegrator(timestep)
    alchemical_context = simtk.openmm.Context(alchemical_system, alchemical_integrator)
    alchemical_context.setPositions(coordinates)
    alchemical_state = alchemical_context.getState(getEnergy=True)
    alchemical_potential = alchemical_state.getPotentialEnergy()
    delta = alchemical_potential - reference_potential 
    print reference_potential
    print alchemical_potential

    # TODO: Also try values very close to lambda = 1 to test how close nearest intermediate can be.
    
    return delta

if __name__ == "__main__":
    # TODO: Uncomment doctests once performance speed is improved.
    #import doctest
    #doctest.testmod()

    # Create a reference system.    
    from simtk.pyopenmm.extras import testsystems

    # TODO: Automate this with loop.

    print "Creating Lennard-Jones cluster system..."
    [reference_system, coordinates] = testsystems.LennardJonesCluster()
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(1,2) # second atom
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating Lennard-Jones fluid system..."
    [reference_system, coordinates] = testsystems.LennardJonesFluid()
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(1,2) # second atom
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating alanine dipeptide implicit system..."
    [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
    ligand_atoms = range(0,4) # methyl group
    receptor_atoms = range(4,22) # rest of system
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating alanine dipeptide explicit system..."
    [reference_system, coordinates] = testsystems.AlanineDipeptideExplicit()
    ligand_atoms = range(0,22) # alanine residue
    receptor_atoms = range(22,25) # one water
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating T4 lysozyme system..."
    [reference_system, coordinates] = testsystems.LysozymeImplicit()
    receptor_atoms = range(0,2603) # T4 lysozyme L99A
    ligand_atoms = range(2603,2621) # p-xylene
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)    
    print ""

