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

#=============================================================================================
# ALCHEMICAL MODIFICATIONS
#=============================================================================================

def createCustomSoftcoreGBOBC(solventDielectric, soluteDielectric, igb):

    custom = openmm.CustomGBForce()

    custom.addPerParticleParameter("q");
    custom.addPerParticleParameter("radius");
    custom.addPerParticleParameter("scale");
    custom.addPerParticleParameter("lambda");
    custom.addGlobalParameter("solventDielectric", solventDielectric);
    custom.addGlobalParameter("soluteDielectric", soluteDielectric);
    custom.addGlobalParameter("offset", 0.009)
    custom.addComputedValue("I",  "lambda1*lambda2*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
                                  "U=r+sr2;"
                                  "L=max(or1, D);"
                                  "D=abs(r-sr2);"
                                  "sr2 = scale2*or2;"
                                  "or1 = radius1-offset; or2 = radius2-offset", openmm.CustomGBForce.ParticlePairNoExclusions)

#    custom.addComputedValue("I", "lambda1*lambda2*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
#                            "U=r+sr2;"
#                            "C=2*(1/or1-1/L)*step(sr2-r-or1);"
#                            "L=max(or1, D);"
#                            "D=abs(r-sr2);"
#                            "sr2 = scale2*or2;"
#                            "or1 = radius1-offset; or2 = radius2-offset", openmm.CustomGBForce.ParticlePairNoExclusions);
    

    if igb == 2:
        custom.addComputedValue("B", "1/(1/or-tanh(0.8*psi+2.909125*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", openmm.CustomGBForce.SingleParticle)

    elif igb == 5:
        custom.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", openmm.CustomGBForce.SingleParticle)

    else:
        print "ERROR: Incorrect igb# input for 'createCustomGBOBC'"
        print "Exiting..."
        sys.exit()

#    custom.addEnergyTerm("-0.5*138.935485*lambda*(1/soluteDielectric-1/solventDielectric)*q^2/B", openmm.CustomGBForce.SingleParticle)
#    custom.addEnergyTerm("-138.935485*lambda1*lambda2*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
#                           "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", openmm.CustomGBForce.ParticlePair)

    custom.addEnergyTerm("lambda*28.3919551*(radius+0.14)^2*(radius/B)^6-lambda*0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", openmm.CustomGBForce.SingleParticle);
    custom.addEnergyTerm("-138.935485*lambda1*lambda2*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                          "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", openmm.CustomGBForce.ParticlePairNoExclusions);
        

    return custom

def build_softcore_system(reference_system, receptor_atoms, ligand_atoms, valence_lambda, coulomb_lambda, vdw_lambda, annihilate=False):
    """
    Build alchemically-modified system where ligand is decoupled or annihilated using *SoftcoreForce classes.
    
    """

    # Create new system.
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
                    K *= valence_lambda
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
                    Ktheta *= valence_lambda
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
                    k *= valence_lambda
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
                if particle_index in ligand_atoms:
                    charge *= coulomb_lambda
                    epsilon *= vdw_lambda
                    # Add modified particle parameters.
                    force.addParticle(charge, sigma, epsilon, vdw_lambda)
                else:
                    # Add unmodified particle parameters.
                    force.addParticle(charge, sigma, epsilon, 1.0)
            for exception_index in range(reference_force.getNumExceptions()):
                # Retrieve parameters.
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
                # Alchemically modify epsilon and chargeprod.
                if (iatom in ligand_atoms) and (jatom in ligand_atoms):
                    if annihilate:
                        epsilon *= vdw_lambda 
                        chargeprod *= coulomb_lambda
                    # Add modified exception parameters.
                    force.addException(iatom, jatom, chargeprod, sigma, epsilon, vdw_lambda)
                else:
                    # Add unmodified exception parameters.
                    force.addException(iatom, jatom, chargeprod, sigma, epsilon, 1.0)                    
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

            force.setSolventDielectric( reference_force.getSolventDielectric() )
            force.setSoluteDielectric( reference_force.getSoluteDielectric() )

            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
                # Alchemically modify parameters.
                if particle_index in ligand_atoms:
                    # Scale charge and contribution to GB integrals.
                    force.addParticle(charge*coulomb_lambda, radius, scaling_factor, coulomb_lambda)
                else:
                    # Don't modulate GB.
                    force.addParticle(charge, radius, scaling_factor, 1.0)

            # Add force to new system.
            system.addForce(force)
        else:
            # Don't add unrecognized forces.
            pass

    return system

def build_custom_system(reference_system, receptor_atoms, ligand_atoms, valence_lambda, coulomb_lambda, vdw_lambda, annihilate=False):
    """
    Build alchemically-modified system where ligand is decoupled or annihilated using Custom*Force classes.
    
    """

    # Create new system.
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
                    K *= valence_lambda
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
                    Ktheta *= valence_lambda
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
                    k *= valence_lambda
                # Add parameters.
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
            # Add force to system.
            system.addForce(force)            
        elif isinstance(reference_force, openmm.NonbondedForce):
            # NonbondedForce will handle charges and exception interactions.
            force = openmm.NonbondedForce()
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
                # Remove Lennard-Jones interactions, which will be handled by CustomNonbondedForce.
                epsilon *= 0.0
                # Alchemically modify charges.
                if particle_index in ligand_atoms:
                    charge *= coulomb_lambda
                # Add modified particle parameters.
                force.addParticle(charge, sigma, epsilon)
            for exception_index in range(reference_force.getNumExceptions()):
                # Retrieve parameters.
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
                # Alchemically modify epsilon and chargeprod.
                # Note that exceptions are handled by NonbondedForce and not CustomNonbondedForce.
                if (iatom in ligand_atoms) and (jatom in ligand_atoms):
                    if annihilate:
                        epsilon *= vdw_lambda 
                        chargeprod *= coulomb_lambda
                # Add modified exception parameters.
                force.addException(iatom, jatom, chargeprod, sigma, epsilon)
            # Set parameters.
            force.setNonbondedMethod( reference_force.getNonbondedMethod() )
            force.setCutoffDistance( reference_force.getCutoffDistance() )
            force.setReactionFieldDielectric( reference_force.getReactionFieldDielectric() )
            force.setEwaldErrorTolerance( reference_force.getEwaldErrorTolerance() )
            # Add force to new system.
            system.addForce(force)

            # CustomNonbondedForce
            # Softcore potential.
            energy_expression = "4*epsilon*lambda*x*(x-1.0);"
            energy_expression += "x = 1.0/(alpha*(1.0-lambda) + (r/sigma)^6);"
            energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
            energy_expression += "sigma = 0.5*(sigma1 + sigma2);"
            energy_expression += "lambda = lambda1*lambda2;"

            force = openmm.CustomNonbondedForce(energy_expression)            
            alpha = 0.5 # softcore parameter
            force.addGlobalParameter("alpha", alpha);
            force.addPerParticleParameter("sigma")
            force.addPerParticleParameter("epsilon")
            force.addPerParticleParameter("lambda"); 
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
                # Alchemically modify parameters.
                if particle_index in ligand_atoms:
                    force.addParticle([sigma, epsilon, vdw_lambda])
                else:
                    force.addParticle([sigma, epsilon, 1.0])
            for exception_index in range(reference_force.getNumExceptions()):
                # Retrieve parameters.
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
                # All exceptions are handled by NonbondedForce, so we exclude all these here.
                force.addExclusion(iatom, jatom)
            if reference_force.getNonbondedMethod() in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
                force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
            else:
                force.setNonbondedMethod( reference_force.getNonbondedMethod() )
            force.setCutoffDistance( reference_force.getCutoffDistance() )
            system.addForce(force)
            
        elif isinstance(reference_force, openmm.GBSAOBCForce):
            # GBSAOBCForce
            solvent_dielectric = reference_force.getSolventDielectric()
            solute_dielectric = reference_force.getSoluteDielectric()
            force = createCustomSoftcoreGBOBC(solvent_dielectric, solute_dielectric, igb=5)
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
                # Alchemically modify parameters.
                if particle_index in ligand_atoms:
                    # Scale charge and contribution to GB integrals.
                    force.addParticle([charge*coulomb_lambda, radius, scaling_factor, coulomb_lambda])
                else:
                    # Don't modulate GB.
                    force.addParticle([charge, radius, scaling_factor, 1.0])

            # Add force to new system.
            system.addForce(force)
        else:
            # Don't add unrecognized forces.
            pass

    return system

def write_file(filename, contents):
    outfile = open(filename, 'w')
    outfile.write(contents)
    outfile.close()
    return

def write_coordinates(filename, coordinates):
    [natoms,dim] = coordinates.shape
    outfile = open(filename, 'w')
    outfile.write('%12d\n' % natoms)
    for atom_index in range(natoms):
        outfile.write('%16.8f %16.8f %16.8f\n' % (coordinates[atom_index,0] / units.nanometers, coordinates[atom_index,1] / units.nanometers, coordinates[atom_index,2] / units.nanometers))
    outfile.close()
    return

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # Create system
    system = openmm.System()

    temperature = 300 * units.kelvin
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    beta = 1.0 / kT
    
    gbmodel = 'OBC'
    #gbmodel = None
    
    # Load amber system.
    receptor_prmtop_filename = 'receptor.prmtop'
    complex_prmtop_filename = 'complex.prmtop'
    ligand_prmtop_filename = 'ligand.prmtop'    
    complex_crd_filename = 'complex.crd'
    import simtk.pyopenmm.amber.amber_file_parser as amber
    receptor_system = amber.readAmberSystem(receptor_prmtop_filename, mm=openmm, gbmodel=gbmodel, nonbondedCutoff=None, nonbondedMethod='no-cutoff', shake='h-bonds')
    complex_system = amber.readAmberSystem(complex_prmtop_filename, mm=openmm, gbmodel=gbmodel, nonbondedCutoff=None, nonbondedMethod='no-cutoff', shake='h-bonds')
    ligand_system = amber.readAmberSystem(ligand_prmtop_filename, mm=openmm, gbmodel=gbmodel, nonbondedCutoff=None, nonbondedMethod='no-cutoff', shake='h-bonds')
    coordinates = amber.readAmberCoordinates(complex_crd_filename)

    # Create alchemically-modified system.
    receptor_atoms = range(0,receptor_system.getNumParticles())
    ligand_atoms = range(receptor_system.getNumParticles(), complex_system.getNumParticles())

    valence_lambda = 1.0
    coulomb_lambda = 0.0
    vdw_lambda = 0.0
    system = build_softcore_system(complex_system, receptor_atoms, ligand_atoms, valence_lambda, coulomb_lambda, vdw_lambda)
    #system = build_custom_system(complex_system, receptor_atoms, ligand_atoms, valence_lambda, coulomb_lambda, vdw_lambda)    
    
    # Add harmonic potential.
    iatom = receptor_atoms[0]
    jatom = ligand_atoms[0]
    sigma = 3.816371 * units.angstroms
    r0 = 0.0 * units.angstroms
    K = 1.0 / (beta * sigma**2)
    print "sigma = %.3f A, r0 = %.3f A, K = %.16f kcal/mol/A**2" % (sigma / units.angstroms, r0 / units.angstroms, K / (units.kilocalories_per_mole / units.angstrom**2))

    add_restraint = True
    if add_restraint:
        add_new = False
        if add_new:
            # Add new HarmonicBondForce
            print "Adding restraint as new HarmonicBondForce."
            force = openmm.HarmonicBondForce()
            force.addBond(iatom, jatom, r0, K)
            system.addForce(force)
        else:
            # Add to existing HarmonicBondForce
            print "Appending restraint to existing HarmonicBondForce."
            for force_index in range(system.getNumForces()):
                force = system.getForce(force_index)
                if isinstance(force, openmm.HarmonicBondForce):
                    # We found it!
                    force.addBond(iatom, jatom, r0, K)
                    break
            
    # Select platform.
    #platform = openmm.Platform.getPlatformByName("Reference")
    platform = openmm.Platform.getPlatformByName("Cuda")
    #platform = openmm.Platform.getPlatformByName("OpenCL")
    
    # Create integrator.
    timestep = 2.0 * units.femtoseconds
    friction_coefficient = 1.0 / units.picosecond
    integrator = openmm.LangevinIntegrator(temperature, friction_coefficient, timestep)
    #integrator = openmm.BrownianIntegrator(temperature, friction_coefficient, timestep)
    #integrator = openmm.VerletIntegrator(timestep)

    # Create context.
    context = openmm.Context(system, integrator, platform)
    context.setPositions(coordinates)

    # Get initial energy.
    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()

    # Run simulation
    distance_filename = 'distances.txt'
    niterations = 10000
    nsteps = 1000
    outfile = open(distance_filename, 'w')
    
    for iteration in range(niterations):
        if numpy.mod(iteration,100)==0: print iteration
        integrator.step(nsteps)
        state = context.getState(getEnergy=True,getPositions=True)
        coordinates = state.getPositions(asNumpy=True)
        potential = state.getPotentialEnergy()
        
        # Compute properties.
        dr = coordinates[jatom,:] - coordinates[iatom,:]
        r = norm(dr)
        outfile.write('%16.8f %16.8f %16.8f %16.8f %16.8f\n' % (r / units.angstroms, potential / units.kilocalories_per_mole, dr[0] / units.angstroms, dr[1] / units.angstroms, dr[2] / units.angstroms))
        outfile.flush()

        if not add_restraint:
            del context, integrator
            
            # Compare receptor energies.
            print "Comparing energies with receptor alone..."
            # Create context for receptor.
            reference_platform = openmm.Platform.getPlatformByName("Cuda")
            receptor_integrator = openmm.VerletIntegrator(timestep)
            receptor_context = openmm.Context(receptor_system, receptor_integrator, reference_platform)
            # Compute energy.
            receptor_context.setPositions(coordinates[receptor_atoms,:])
            state = receptor_context.getState(getEnergy=True)
            receptor_potential = state.getPotentialEnergy()
            print "Total potential = %.5f kcal/mol, receptor potential = %.5f kcal/mol, error = %.5f kcal/mol" % (potential / units.kilocalories_per_mole, receptor_potential / units.kilocalories_per_mole, (receptor_potential - potential) / units.kilocalories_per_mole)

            del receptor_context, receptor_integrator
            integrator = openmm.LangevinIntegrator(temperature, friction_coefficient, timestep)
            context = openmm.Context(system, integrator, platform)
            context.setPositions(coordinates)            
    
    outfile.close()
