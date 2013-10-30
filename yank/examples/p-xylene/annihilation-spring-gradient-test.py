#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Compare energy and gradient for standard Force terms, SoftcoreForce terms, and CustomForce terms
implementing alchemical annihilation and decoupling.

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
            #system.addForce(force)
            
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
            #system.addForce(force)
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

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    timestep = 2.0 * units.femtoseconds

    verbose = False

    # Set whether to use GB model or not.
    gbmodel = 'OBC' # use OBC GBSA (gives finite error)
    #gbmodel = None # no solvent model (gives zero error)
    
    # PARAMETERS
    complex_prmtop_filename = 'complex.prmtop'
    complex_crd_filename = 'complex.crd'
#    complex_crd_filename = 'complex-minimized.crd'    
    receptor_prmtop_filename = 'receptor.prmtop'
    receptor_crd_filename = 'receptor.crd'
#    receptor_crd_filename = 'receptor-minimized.crd'
    ligand_prmtop_filename = 'ligand.prmtop'
    ligand_crd_filename = 'ligand.crd'
#    ligand_crd_filename = 'ligand-minimized.crd'
    platform_names = ['Reference', 'Cuda', 'OpenCL']
    platform_names = ['Cuda', 'OpenCL']    

    # Create force type variants of systems.
    force_types = ['standard', 'softcore lambda=1', 'softcore lambda=0', 'custom lambda=1', 'custom lambda=0'] # Different force types
    system_types = ['complex', 'receptor', 'ligand']
    systems = dict()
    for system_type in system_types:
        systems[system_type] = dict()        
    # Load standard systems.
    import simtk.pyopenmm.amber.amber_file_parser as amber    
    systems['complex']['standard'] = amber.readAmberSystem(complex_prmtop_filename, mm=openmm, gbmodel=gbmodel, nonbondedCutoff=None, nonbondedMethod='no-cutoff', shake='h-bonds')
    systems['receptor']['standard'] = amber.readAmberSystem(receptor_prmtop_filename, mm=openmm, gbmodel=gbmodel, nonbondedCutoff=None, nonbondedMethod='no-cutoff', shake='h-bonds')
    systems['ligand']['standard'] = amber.readAmberSystem(ligand_prmtop_filename, mm=openmm, gbmodel=gbmodel, nonbondedCutoff=None, nonbondedMethod='no-cutoff', shake='h-bonds')
                
    # Determine number of particles.
    nparticles = dict()
    for system_type in system_types:
        nparticles[system_type] = systems[system_type]['standard'].getNumParticles()
    nparticles['ligand'] = nparticles['complex'] - nparticles['receptor']

    receptor_atoms = range(0,nparticles['receptor'])
    ligand_atoms = range(nparticles['receptor'], nparticles['complex'])

    # Add harmonic protein-ligand restraint.
    add_restraint = True
    iatom = receptor_atoms[0]
    jatom = ligand_atoms[0]
    temperature = 300 * units.kelvin
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    beta = 1.0 / kT
    sigma = 3.816371 * units.angstroms
    r0 = 0.0 * units.angstroms
    K = 1.0 / (beta * sigma**2)
    print "sigma = %.3f A, r0 = %.3f A, K = %.16f kcal/mol/A**2" % (sigma / units.angstroms, r0 / units.angstroms, K / (units.kilocalories_per_mole / units.angstrom**2))
    if add_restraint:
        add_new = False
        if add_new:
            # Add new HarmonicBondForce
            print "Adding restraint as new HarmonicBondForce."
            force = openmm.HarmonicBondForce()
            force.addBond(iatom, jatom, r0, K)
            systems['complex']['standard'].addForce(force)
        else:
            # Add to existing HarmonicBondForce
            print "Appending restraint to existing HarmonicBondForce."
            for force_index in range(systems['complex']['standard'].getNumForces()):
                force = systems['complex']['standard'].getForce(force_index)
                if isinstance(force, openmm.HarmonicBondForce):
                    # We found it!
                    force.addBond(iatom, jatom, r0, K)
                    break
    
    # Create softcore force type variants.
    for system_type in system_types:
        if system_type == 'complex':
            receptor_atoms = range(0,nparticles['receptor'])
            ligand_atoms = range(nparticles['receptor'], nparticles['receptor'] + nparticles['ligand'])
        elif system_type == 'receptor':
            receptor_atoms = range(0,nparticles['receptor'])
            ligand_atoms = list()
        elif system_type == 'ligand':
            receptor_atoms = list()
            ligand_atoms = range(0,nparticles['ligand'])

        systems[system_type]['softcore lambda=1'] = build_softcore_system(systems[system_type]['standard'], receptor_atoms, ligand_atoms, 1.0, 1.0, 1.0)
        systems[system_type]['softcore lambda=0'] = build_softcore_system(systems[system_type]['standard'], receptor_atoms, ligand_atoms, 1.0, 0.0, 0.0)
        systems[system_type]['custom lambda=1'] = build_custom_system(systems[system_type]['standard'], receptor_atoms, ligand_atoms, 1.0, 1.0, 1.0)
        systems[system_type]['custom lambda=0'] = build_custom_system(systems[system_type]['standard'], receptor_atoms, ligand_atoms, 1.0, 0.0, 0.0)

    receptor_atoms = range(0,nparticles['receptor'])
    ligand_atoms = range(nparticles['receptor'], nparticles['complex'])

    # Read coordinates.
    coordinates = dict()
    coordinates['complex'] = amber.readAmberCoordinates(complex_crd_filename)
    # Translate ligand to be inside of receptor.
    #coordinates['complex'][ligand_atoms,:] -= 3.0 * units.angstroms
    # End translate
    coordinates['receptor'] = coordinates['complex'][receptor_atoms,:]
    coordinates['ligand'] = coordinates['complex'][ligand_atoms,:]

#    # Write out .xml files.
#    write_file('complex.xml', simtk.openmm.XmlSerializer.serializeSystem(systems['complex']['standard']))
#    write_file('receptor.xml', simtk.openmm.XmlSerializer.serializeSystem(systems['receptor']['standard']))    
#    write_file('ligand.xml', simtk.openmm.XmlSerializer.serializeSystem(systems['ligand']['standard']))
#
#    # Write out coordinates.
#    write_coordinates('complex.xyz', coordinates[:,:])    
#    write_coordinates('receptor.xyz', coordinates[receptor_atoms,:])
#    write_coordinates('ligand.xyz', coordinates[ligand_atoms,:])

    niterations = 100
    for iteration in range(niterations):

        #==============================================================================================
        # Compute potential energy and force for all combinations.
        #==============================================================================================

        potential = dict()
        forces = dict()
        for platform_name in platform_names:
            potential[platform_name] = dict()
            forces[platform_name] = dict()        

            # Select platform to use.
            platform = openmm.Platform.getPlatformByName(platform_name)

            for system_type in system_types:
                potential[platform_name][system_type] = dict()
                forces[platform_name][system_type] = dict()            

                for force_type in force_types:
                    potential[platform_name][system_type][force_type] = dict()
                    forces[platform_name][system_type][force_type] = dict()                                        

                    # Compute full complex energy.
                    if verbose: print "Computing potential for system '%(system_type)s' force type '%(force_type)s' on platform '%(platform_name)s'" % vars()
                    try:                
                        integrator = openmm.VerletIntegrator(timestep)
                        context = openmm.Context(systems[system_type][force_type], integrator, platform)
                        context.setPositions(coordinates[system_type])
                        state = context.getState(getEnergy=True,getForces=True)
                        potential[platform_name][system_type][force_type] = state.getPotentialEnergy()
                        forces[platform_name][system_type][force_type] = state.getForces(asNumpy=True)
                        if verbose: print state.getPotentialEnergy()
                        del integrator, context
                    except Exception as e:
                        if verbose: print " *** %s" % str(e)                    
                        potential[platform_name][system_type][force_type] = None
                        forces[platform_name][system_type][force_type] = None

            # Test gradient for complex.
            try:
                integrator = openmm.VerletIntegrator(timestep)
                system_type = 'complex'
                force_type = 'softcore lambda=0'
                delta = 1.0e-5
                print "Force (analytical above, finite-difference below) with delta = %f angstroms for platform %s" % (delta, platform_name)
                context = openmm.Context(systems[system_type][force_type], integrator, platform)                
                force_units = units.kilojoules_per_mole / units.nanometer
                import copy
                for iatom in ligand_atoms:
                    print "atom %d" % iatom                
                    for k in range(3):
                        print "%14.8f" % (forces[platform_name][system_type][force_type][iatom,k] / force_units),
                    print ""
                    for k in range(3):
                        x0 = copy.deepcopy(coordinates[system_type])
                        dx = delta * abs(x0[iatom,k])
                        x0[iatom,k] += dx
                        context.setPositions(x0)  
                        state = context.getState(getEnergy=True)
                        Ep = state.getPotentialEnergy()
                        x0 = copy.deepcopy(coordinates[system_type])                        
                        x0[iatom,k] -= dx
                        context.setPositions(x0)                      
                        state = context.getState(getEnergy=True)
                        Em = state.getPotentialEnergy()                    
                        dEdx = (Ep - Em) / (dx + dx)
                        print "%14.8f" % (-dEdx / force_units),
                    print ""
                del integrator, context                                
            except Exception as e:
                print e
                pass


        #==============================================================================================
        # Show energies.
        #==============================================================================================

        print "%24s : " % "Platforms:",
        for platform_name in platform_names: print "%24s       " % (platform_name),
        print ""

        # fully interacting
        print "-----------------------------"
        print "Interacting ligand in complex"
        for force_type in ['standard', 'softcore lambda=1', 'custom lambda=1']:
            print "%24s : " % (force_type),
            for platform_name in platform_names:
                energy = potential[platform_name]['complex'][force_type]
                if energy is None:
                    print "%24s       " % "N/A",                
                else:
                    if (abs(energy) > 1e10 * units.kilojoules_per_mole):                    
                        print "%24.8e kJ/mol" % (energy / units.kilojoules_per_mole),
                    else:
                        print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),                    
            print ""

        # non-interacting complex
        print "-----------------------------"
        print "Annihilated ligand in complex"

        for force_type in ['softcore lambda=0', 'custom lambda=0']:
            print "%24s : " % (force_type),
            for platform_name in platform_names:
                energy = potential[platform_name]['complex'][force_type]
                if energy is None:
                    print "%24s       " % "N/A",
                else:
                    if (energy > 1e10 * units.kilojoules_per_mole):                    
                        print "%24.8e kJ/mol" % (energy / units.kilojoules_per_mole),
                    else:
                        print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),                    
            print ""

        print "-----------------------------"
        print "Receptor only + ligand only"

        for force_type in force_types:
            print "%24s : " % (force_type),
            for platform_name in platform_names:
                energy = potential[platform_name]['receptor'][force_type] 
                if energy is None:
                    print "%24s       " % "N/A",                
                else:
                    energy += potential[platform_name]['ligand'][force_type] 
                    if (energy > 1e10 * units.kilojoules_per_mole):                    
                        print "%24.8e kJ/mol" % (energy / units.kilojoules_per_mole),
                    else:
                        print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),                    
            print ""
        print "-----------------------------"
        print "ERROR (complex - (receptor + ligand)) : should match 'SPRING CONTRIBUTION' below"

        for force_type in ['softcore lambda=0', 'custom lambda=0']:
            print "%24s : " % (force_type),
            for platform_name in platform_names:
                if (potential[platform_name]['complex'][force_type] is None) or (potential[platform_name]['receptor'][force_type] is None) or (potential[platform_name]['ligand'][force_type] is None):
                    print "%24s       " % "N/A",                
                else:
                    energy = potential[platform_name]['complex'][force_type] - (potential[platform_name]['receptor'][force_type] + potential[platform_name]['ligand'][force_type])
                    if (abs(energy) > 1e10 * units.kilojoules_per_mole):                    
                        print "%24.8e kJ/mol" % (energy / units.kilojoules_per_mole),
                    else:
                        print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),                    
            print ""
        print "-----------------------------"
        print "SPRING CONTRIBUTION"
        print ""
        r = norm(coordinates['complex'][iatom,:] - coordinates['complex'][jatom,:])
        Uspring = (K/2.0) * (r-r0)**2
        print "%22.3f A : %24.8f kJ/mol" % (r / units.angstroms, Uspring / units.kilojoules_per_mole)
        print "-----------------------------"
        print "Receptor only"

        for force_type in force_types:
            print "%24s : " % (force_type),
            for platform_name in platform_names:
                energy = potential[platform_name]['receptor'][force_type]
                if energy is None:
                    print "%24s       " % "N/A",                
                else:
                    if (energy > 1e10 * units.kilojoules_per_mole):                    
                        print "%24.8e kJ/mol" % (energy / units.kilojoules_per_mole),
                    else:
                        print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),                    
            print ""
        print "-----------------------------"
        print "Ligand only"

        for force_type in force_types:
            print "%24s : " % (force_type),
            for platform_name in platform_names:
                energy = potential[platform_name]['ligand'][force_type]
                if energy is None:
                    print "%24s       " % "N/A",                
                else:
                    if (energy > 1e10 * units.kilojoules_per_mole):                    
                        print "%24.8e kJ/mol" % (energy / units.kilojoules_per_mole),
                    else:
                        print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),                    
            print ""

        #==============================================================================================
        # Show force errors.
        #==============================================================================================

        print "%24s : " % "Platforms:",
        for platform_name in platform_names: print "%24s       " % (platform_name),
        print ""

        import numpy.linalg

        # fully interacting
        print "-----------------------------"
        print "Interacting ligand in complex (lambda = 1)"
        print "%24s : " % ("Softcore - Custom"),
        if (forces['Cuda']['complex']['softcore lambda=1'] is not None) and (forces['OpenCL']['complex']['custom lambda=1'] is not None):
            force_difference = (forces['Cuda']['complex']['softcore lambda=1'] - forces['OpenCL']['complex']['custom lambda=1'])  / (units.kilojoules_per_mole / units.nanometer)
            force_mean = 0.5 * (forces['Cuda']['complex']['softcore lambda=1'] + forces['OpenCL']['complex']['custom lambda=1'])  / (units.kilojoules_per_mole / units.nanometer)        
            force_relative_error = numpy.linalg.norm(force_difference) / numpy.linalg.norm(force_mean)
            print "%24.8e       " % (force_relative_error),                    
        else:
            print "%24s       " % "N/A",                
        print ""

        # noninteracting
        print "-----------------------------"
        print "Noninteracting ligand in complex (lambda = 0)"
        print "%24s : " % ("Softcore - Custom"),
        if (forces['Cuda']['complex']['softcore lambda=0'] is not None) and (forces['OpenCL']['complex']['custom lambda=0'] is not None):
            force_difference = (forces['Cuda']['complex']['softcore lambda=0'] - forces['OpenCL']['complex']['custom lambda=0'])  / (units.kilojoules_per_mole / units.nanometer)
            force_mean = 0.5 * (forces['Cuda']['complex']['softcore lambda=0'] + forces['OpenCL']['complex']['custom lambda=0'])  / (units.kilojoules_per_mole / units.nanometer)        
            force_relative_error = numpy.linalg.norm(force_difference) / numpy.linalg.norm(force_mean)
            print "%24.8e       " % (force_relative_error),                    
        else:
            print "%24s       " % "N/A",                
        print ""


        # fully interacting
        print "-----------------------------"
        print "Interacting receptor (lambda = 1)"
        print "%24s : " % ("Softcore - Custom"),
        if (forces['Cuda']['receptor']['softcore lambda=1'] is not None) and (forces['OpenCL']['receptor']['custom lambda=1'] is not None):
            force_difference = (forces['Cuda']['receptor']['softcore lambda=1'] - forces['OpenCL']['receptor']['custom lambda=1'])  / (units.kilojoules_per_mole / units.nanometer)
            force_mean = 0.5 * (forces['Cuda']['receptor']['softcore lambda=1'] + forces['OpenCL']['receptor']['custom lambda=1'])  / (units.kilojoules_per_mole / units.nanometer)        
            force_relative_error = numpy.linalg.norm(force_difference) / numpy.linalg.norm(force_mean)
            print "%24.8e       " % (force_relative_error),                    
        else:
            print "%24s       " % "N/A",                
        print ""

        # noninteracting
        print "-----------------------------"
        print "Noninteracting receptor (lambda = 0)"
        print "%24s : " % ("Softcore - Custom"),
        if (forces['Cuda']['receptor']['softcore lambda=0'] is not None) and (forces['OpenCL']['receptor']['custom lambda=0'] is not None):
            force_difference = (forces['Cuda']['receptor']['softcore lambda=0'] - forces['OpenCL']['receptor']['custom lambda=0'])  / (units.kilojoules_per_mole / units.nanometer)
            force_mean = 0.5 * (forces['Cuda']['receptor']['softcore lambda=0'] + forces['OpenCL']['receptor']['custom lambda=0'])  / (units.kilojoules_per_mole / units.nanometer)        
            force_relative_error = numpy.linalg.norm(force_difference) / numpy.linalg.norm(force_mean)
            print "%24.8e       " % (force_relative_error),                    
        else:
            print "%24s       " % "N/A",                
        print ""

        #==============================================================================================
        # Run MD
        #==============================================================================================

        nsteps = 500
        print "Running %d steps of MD..." % nsteps
        platform = openmm.Platform.getPlatformByName('Cuda')
        friction_coefficient = 1.0 / units.picoseconds
        integrator = openmm.LangevinIntegrator(temperature, friction_coefficient, timestep)
        context = openmm.Context(systems['complex']['softcore lambda=0'], integrator, platform)
        context.setPositions(coordinates['complex'])
        integrator.step(nsteps)
        state = context.getState(getPositions=True)
        coordinates['complex'] = state.getPositions(asNumpy=True)
        coordinates['receptor'] = coordinates['complex'][receptor_atoms,:]
        coordinates['ligand'] = coordinates['complex'][ligand_atoms,:]
        del integrator, context
        
    #==============================================================================================
    # Run timings
    #==============================================================================================

    nsteps = 500

    elapsed_times = dict() # elapsed_times[platform_name] is time for nsteps of dynamics
    system_type = 'complex'
    for platform_name in platform_names:
        # Select platform to use.
        platform = openmm.Platform.getPlatformByName(platform_name)

        elapsed_times[platform_name] = dict()
        for force_type in force_types:

            if verbose: print "Running system '%(system_type)s' force type '%(force_type)s' for %(nsteps)d steps of dynamics on platform '%(platform_name)s'..." % vars()
            try:
                integrator = openmm.VerletIntegrator(timestep)
                context = openmm.Context(systems[system_type][force_type], integrator, platform)
                context.setPositions(coordinates[system_type])
                initial_time = time.time()
                integrator.step(nsteps)
                final_time = time.time()
                elapsed_time = final_time - initial_time
                elapsed_times[platform_name][force_type] = elapsed_time
                state = context.getState(getEnergy=True)
                potential = state.getPotentialEnergy() 
                print potential
                if numpy.isnan(potential / units.kilojoules_per_mole): 
                    elapsed_times[platform_name][force_type] = None                   
                    print " *** System '%(system_type)s' with force type '%(force_type)s' on platform '%(platform_name)s' exploded." % vars()
                del integrator, context
            except Exception as e:
                if verbose: print " *** %s" % str(e)                    
                elapsed_times[platform_name][force_type] = None

    print "-----------------------------"
    print "Timings for system '%s' (%d steps)" % (system_type, nsteps)
    
    for force_type in force_types:
        print "%24s : " % (force_type),
        for platform_name in platform_names:
            elapsed_time = elapsed_times[platform_name][force_type]
            if elapsed_time is None:
                print "%24s       " % ("N/A"),
            else:
                ns_per_day =  (float(nsteps) * timestep / (elapsed_time * units.seconds)) / (units.nanoseconds / units.day)                
                print "%8.3f ms (%8.1f ns/day)  " % (elapsed_time*1000, ns_per_day),
        print ""
