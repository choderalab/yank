#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Run Metropolis MC simulation of simple dimer interacting only by harmonic bond.

DESCRIPTION

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

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import sys
import math
import copy
import time

import numpy

import simtk
import simtk.unit as units
import simtk.openmm as openmm

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def norm(n01):
    return n01.unit * numpy.sqrt(numpy.dot(n01/n01.unit, n01/n01.unit))

#=============================================================================================
# ALCHEMICAL MODIFICATIONS
#=============================================================================================

def build_alchemically_modified_system(reference_system, receptor_atoms, ligand_atoms, annihilate=True):
    """
    Build alchemically-modified system where ligand is decoupled or annihilated.
    
    """

    # Create new system.
    system = openmm.System()

    # Add atoms.
    for atom_index in range(reference_system.getNumParticles()):
        mass = reference_system.getParticleMass(atom_index)
        system.addParticle(mass)

    # Add constraints
    for constraint_index in range(reference_system.getNumConstraints()):
        [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
        system.addConstraint(iatom, jatom, r0)    

    # Perturb force terms.
    for force_index in range(reference_system.getNumForces()):
        reference_force = reference_system.getForce(force_index)
        # Dispatch forces
        if isinstance(reference_force, openmm.HarmonicBondForce):
            # HarmonicBondForce
            force = openmm.HarmonicBondForce()
            for bond_index in range(reference_force.getNumBonds()):
                # Retrieve parameters.
                [iatom, jatom, r0, K] = reference_force.getBondParameters(bond_index)
                # Annihilate if directed.
                if annihilate and (iatom in ligand_atoms) and (jatom in ligand_atoms):
                    K *= 0.0                    
                # Add bond parameters.
                force.addBond(iatom, jatom, r0, K)
            # Add force to new system.
            system.addForce(force)
        elif isinstance(reference_force, openmm.HarmonicAngleForce):
            # HarmonicAngleForce
            force = openmm.HarmonicAngleForce()
            for angle_index in range(reference_force.getNumAngles()):
                # Retrieve parameters.
                [iatom, jatom, katom, theta0, Ktheta] = reference_force.getAngleParameters(angle_index)
                # Annihilate if directed:
                if annihilate and (iatom in ligand_atoms) and (jatom in ligand_atoms) and (katom in ligand_atoms):
                    Ktheta *= 0.0                                    
                # Add parameters.
                force.addAngle(iatom, jatom, katom, theta0, Ktheta)
            # Add force to system.                
            system.addForce(force)
        elif isinstance(reference_force, openmm.PeriodicTorsionForce):
            # PeriodicTorsionForce
            force = openmm.PeriodicTorsionForce()
            for torsion_index in range(reference_force.getNumTorsions()):
                # Retrieve parmaeters.
                [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                # Annihilate if directed:
                if annihilate and (particle1 in ligand_atoms) and (particle2 in ligand_atoms) and (particle3 in ligand_atoms) and (particle4 in ligand_atoms):
                    k *= 0.0                                                    
                # Add parameters.
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
            # Add force to system.
            system.addForce(force)            
        elif isinstance(reference_force, openmm.NonbondedForce):
            # NonbondedForce
            force = openmm.NonbondedSoftcoreForce()
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
                # Alchemically modify parameters.
                alchemical_lambda = 1.0
                if particle_index in ligand_atoms:
                    alchemical_lambda = 0.0
                    charge *= 0.0
                    if annihilate:
                        epsilon *= 0.0
                # Add modified particle parameters.
                force.addParticle(charge, sigma, epsilon, alchemical_lambda)
            for exception_index in range(reference_force.getNumExceptions()):
                # Retrieve parameters.
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
                # TODO: Alchemically modify parameters.
                if (iatom in ligand_atoms) and (jatom in ligand_atoms):
                    chargeprod *= 0.0
                    if annihilate:
                        epsilon *= 0.0
                # Add modified exception parameters.
                force.addException(iatom, jatom, chargeprod, sigma, epsilon)
            # Set parameters.
            force.setNonbondedMethod( reference_force.getNonbondedMethod() )
            force.setCutoffDistance( reference_force.getCutoffDistance() )
            force.setReactionFieldDielectric( reference_force.getReactionFieldDielectric() )
            force.setEwaldErrorTolerance( reference_force.getEwaldErrorTolerance() )
            # Add force to new system.
            system.addForce(force)
        elif isinstance(reference_force, openmm.GBSAOBCForce):
            # GBSAOBCForce
            force = openmm.GBSAOBCSoftcoreForce()
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
                # Alchemically modify parameters.
                nonpolar_scaling_factor = 1.0
                if particle_index in ligand_atoms:                    
                    charge *= 0.0
                    #radius *= 0.0
                    #scaling_factor *= 0.0
                    nonpolar_scaling_factor = 0.0
                    pass
                # Add parameters.
                force.addParticle(charge, radius, scaling_factor, nonpolar_scaling_factor)
            force.setSolventDielectric( reference_force.getSolventDielectric() )
            force.setSoluteDielectric( reference_force.getSoluteDielectric() )
            #force.setCutoffDistance( reference_force.getCutoffDistance() )
            #force.setNonbondedMethod( reference_force.getNonbondedMethod() )
            # Add force to new system.
            system.addForce(force)
        else:
            # Don't add unrecognized forces.
            pass

    return system

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # Create system
    system = openmm.System()

    timestep = 1.0 * units.femtoseconds

    # Set whether to use GB model or not.
    gbmodel = 'OBC' # use OBC GBSA (gives finite error)
    #gbmodel = 'none' # no solvent model (gives zero error)
    
    # Load amber system for complex.
    prmtop_filename = 'complex.prmtop'
    crd_filename = 'complex.crd'
    import simtk.pyopenmm.amber.amber_file_parser as amber
    complex_system = amber.readAmberSystem(prmtop_filename, mm=openmm, gbmodel=gbmodel, shake='h-bonds')
    complex_coordinates = amber.readAmberCoordinates(crd_filename)

    # Load amber system for receptor only.
    prmtop_filename = 'receptor.prmtop'
    crd_filename = 'receptor.crd'
    receptor_system = amber.readAmberSystem(prmtop_filename, mm=openmm, gbmodel=gbmodel, shake='h-bonds')
    receptor_coordinates = amber.readAmberCoordinates(crd_filename)

    # Create alchemically-modified system where ligand is turned off.
    receptor_atoms = range(0,2603)
    ligand_atoms = range(2603,2621)
    alchemically_modified_complex_system = build_alchemically_modified_system(complex_system, receptor_atoms, ligand_atoms)

    for platform_name in ['Cuda', 'Reference']:
        # Select platform to use.
        platform = openmm.Platform.getPlatformByName(platform_name)
        print "platform %s" % platform_name

        # Compute complex energy.
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(alchemically_modified_complex_system, integrator, platform)
        context.setPositions(complex_coordinates)
        state = context.getState(getEnergy=True)
        complex_potential = state.getPotentialEnergy()
        del integrator, context

        # Compute receptor energy.
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(receptor_system, integrator, platform)
        context.setPositions(complex_coordinates[receptor_atoms,:])
        state = context.getState(getEnergy=True)
        receptor_potential = state.getPotentialEnergy()
        del integrator, context

        # Show results.
        print "receptor : %16.8f kJ/mol" % (receptor_potential / units.kilojoules_per_mole)
        print "complex  : %16.8f kJ/mol" % (complex_potential / units.kilojoules_per_mole)
        print "ERROR      %16.8f kJ/mol" % ((complex_potential - receptor_potential) / units.kilojoules_per_mole)
        print "\n"

    
