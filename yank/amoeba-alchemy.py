#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Module for aiding in the production of alchemical intermediates for OpenMM System objects.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

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

    custom.addComputedValue("I",  "lambda2*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
                                  "U=r+sr2;"
                                  "L=max(or1, D);"
                                  "D=abs(r-sr2);"
                                  "sr2 = scale2*or2;"
                                  "or1 = radius1-offset; or2 = radius2-offset", openmm.CustomGBForce.ParticlePairNoExclusions)

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

    custom.addEnergyTerm("lambda*28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", openmm.CustomGBForce.SingleParticle); 
    custom.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                          "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", openmm.CustomGBForce.ParticlePairNoExclusions);
        
    return custom

def build_softcore_system(reference_system, ligand_atoms, coulomb_lambda, vdw_lambda, annihilate=True):
    """
    Build alchemically-modified system where ligand is decoupled or annihilated using *SoftcoreForce classes from freeEnergy plugin.

    ARGUMENTS

    reference_system (simtk.openmm.System) - the System object to use as a template for generating alchemical intermediates
    ligand_atoms (list of int) - list of atoms defining ligand to be alchemically modified
    coulomb_lambda (float) - alchemical parameter to decouple or annihilate ligand electrostatic interactions
    vdw_lambda (float) alchemical parameter to decouple or annihilate ligand Lennard-Jones interactions
    
    OPTIONAL ARGUMENTS
    
    annihilate (boolean) - if True, will annihilate ligand electrostatics instead of decoupling (default: True)

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
            # Copy HarmonicBondForce.
            force = openmm.HarmonicBondForce()
            for bond_index in range(reference_force.getNumBonds()):
                # Retrieve parameters.
                [iatom, jatom, r0, K] = reference_force.getBondParameters(bond_index)
                # Add bond parameters.
                force.addBond(iatom, jatom, r0, K)
            # Add force to new system.
            system.addForce(force)
        elif isinstance(reference_force, openmm.HarmonicAngleForce):
            # Copy HarmonicAngleForce.
            force = openmm.HarmonicAngleForce()
            for angle_index in range(reference_force.getNumAngles()):
                # Retrieve parameters.
                [iatom, jatom, katom, theta0, Ktheta] = reference_force.getAngleParameters(angle_index)
                # Add parameters.
                force.addAngle(iatom, jatom, katom, theta0, Ktheta)
            # Add force to system.                
            system.addForce(force)
        elif isinstance(reference_force, openmm.PeriodicTorsionForce):
            # Copy PeriodicTorsionForce.
            force = openmm.PeriodicTorsionForce()
            for torsion_index in range(reference_force.getNumTorsions()):
                # Retrieve parmaeters.
                [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                # Add parameters.
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
            # Add force to system.
            system.addForce(force)            
        elif isinstance(reference_force, openmm.NonbondedForce):
            # Alchemically modify NonbondedForce.
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
            # Add force to new system.
            system.addForce(force)
            
        elif isinstance(reference_force, openmm.GBSAOBCForce):
            # Alchemically modify GBSAOBCForce.
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
            # Raise exception that we don't recognize this force.
            raise Exception("Can't handle force type '%s'." % str(force))

    return system

def build_custom_system(reference_system, ligand_atoms, coulomb_lambda, vdw_lambda, annihilate=True):
    """
    Build alchemically-modified system where ligand is decoupled or annihilated using Custom*Force classes.

    ARGUMENTS

    reference_system (simtk.openmm.System) - the System object to use as a template for generating alchemical intermediates
    ligand_atoms (list of int) - list of atoms defining ligand to be alchemically modified
    coulomb_lambda (float) - alchemical parameter to decouple or annihilate ligand electrostatic interactions
    vdw_lambda (float) alchemical parameter to decouple or annihilate ligand Lennard-Jones interactions
    
    OPTIONAL ARGUMENTS
    
    annihilate (boolean) - if True, will annihilate ligand electrostatics instead of decoupling (default: True)
    
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
            # Copy HarmonicBondForce.
            force = openmm.HarmonicBondForce()
            for bond_index in range(reference_force.getNumBonds()):
                # Retrieve parameters.
                [iatom, jatom, r0, K] = reference_force.getBondParameters(bond_index)
                # Add bond parameters.
                force.addBond(iatom, jatom, r0, K)
            # Add force to new system.
            system.addForce(force)
        elif isinstance(reference_force, openmm.HarmonicAngleForce):
            # Copy HarmonicAngleForce.
            force = openmm.HarmonicAngleForce()
            for angle_index in range(reference_force.getNumAngles()):
                # Retrieve parameters.
                [iatom, jatom, katom, theta0, Ktheta] = reference_force.getAngleParameters(angle_index)
                # Add parameters.
                force.addAngle(iatom, jatom, katom, theta0, Ktheta)
            # Add force to system.                
            system.addForce(force)
        elif isinstance(reference_force, openmm.PeriodicTorsionForce):
            # Copy PeriodicTorsionForce.
            force = openmm.PeriodicTorsionForce()
            for torsion_index in range(reference_force.getNumTorsions()):
                # Retrieve parmaeters.
                [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
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
            # Raise exception that we don't recognize this force.
            raise Exception("Can't handle force type '%s'." % str(force))

    return system

def build_softcore_amoeba_system(reference_system, ligand_atoms, coulomb_lambda, vdw_lambda, annihilate=True):
    """
    Build alchemically-modified AMOEBA system.

    ARGUMENTS

    reference_system (simtk.openmm.System) - the System object to use as a template for generating alchemical intermediates
    ligand_atoms (list of int) - list of atoms defining ligand to be alchemically modified
    coulomb_lambda (float) - alchemical parameter to decouple or annihilate ligand electrostatic interactions
    vdw_lambda (float) alchemical parameter to decouple or annihilate ligand Lennard-Jones interactions
    
    OPTIONAL ARGUMENTS
    
    annihilate (boolean) - if True, will annihilate ligand electrostatics instead of decoupling (default: True)

    WARNING

    This version currently ignores exclusions, so should only be used on annihilating single-atom systems.

    """

    # Create a copy of the system.
    import copy
    system = copy.deepcopy(reference_system)

    # Perturb force terms.
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        # Dispatch forces
        if isinstance(force, openmm.AmoebaMultipoleForce):
            for particle_index in ligand_atoms:
                # Retrieve parameters.
                [charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, dampingFactor, polarizibility] = force.getMultipoleParameters(particle_index)
                # Modify parameters.
                charge = charge * coulomb_lambda
                molecularDipole = coulomb_lambda * numpy.array(molecularDipole)
                molecularQuadrupole = coulomb_lambda * numpy.array(molecularQuadrupole)
                polarizibility *= coulomb_lambda                
                force.setMultipoleParameters(particle_index, charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, dampingFactor, polarizibility)
        elif isinstance(force, openmm.AmoebaVdwForce):
            # Create a softcore force to add back modulated vdW interactions with alchemically-modified atoms.
            # Softcore Halgren potential from Eq. 3 of
            # Shi, Y., Jiao, D., Schnieders, M.J., and Ren, P. (2009). Trypsin-ligand binding free energy calculation with AMOEBA. Conf Proc IEEE Eng Med Biol Soc 2009, 2328-2331.
            energy_expression = 'lambda^5 * epsilon * (1.07^7 / (0.7*(1-lambda)^2+(rho+0.07)^7)) * (1.12 / (0.7*(1-lambda)^2 + rho^7 + 0.12) - 2);'
            energy_expression += 'epsilon = 4*epsilon1*epsilon2 / (sqrt(epsilon1) + sqrt(epsilon2))^2;'
            energy_expression += 'rho = r / R0;'
            energy_expression += 'R0 = (R01^3 + R02^3) / (R01^2 + R02^2);'
            energy_expression += 'lambda = vdw_lambda * (ligand1*(1-ligand2) + ligand2*(1-ligand1)) + ligand1*ligand2;'
            energy_expression += 'vdw_lambda = %f;' % vdw_lambda
#            print energy_expression


            softcore_force = openmm.CustomNonbondedForce(energy_expression)
            softcore_force.addPerParticleParameter('epsilon')
            softcore_force.addPerParticleParameter('R0')
            softcore_force.addPerParticleParameter('ligand')

            for particle_index in range(system.getNumParticles()):
                # Retrieve parameters from vdW force.
                [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)   
#                [ivIndex, classIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)   
#                print str(particle_index) + " " + str(epsilon) + " " + str(sigma)
                # Add parameters to CustomNonbondedForce.
                if particle_index in ligand_atoms:
                    softcore_force.addParticle([epsilon, sigma, 1])
                else:
                    softcore_force.addParticle([epsilon, sigma, 0])

                # Deal with exclusions.
                excluded_atoms = force.getParticleExclusions(particle_index)
                #print str(particle_index) + ' : ' + str(excluded_atoms)
                for jatom in excluded_atoms:
                    if (particle_index < jatom):
                        softcore_force.addExclusion(particle_index, jatom)

            # Make sure periodic boundary conditions are treated the same way.
#            if force.getPBC():
                softcore_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
#            else:
#                softcore_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffNonperiodic )
            softcore_force.setCutoffDistance( force.getCutoff() )

            # Add the softcore force.
            system.addForce( softcore_force )

            # Turn off vdW interactions for alchemically-modified atoms.
            for particle_index in ligand_atoms:
                # Retrieve parameters.
                [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
#                [ivIndex, classIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
                epsilon = 1.0e-6 * epsilon
#                force.setParticleParameters(particle_index, ivIndex, classIndex, sigma, epsilon, reductionFactor)
                force.setParticleParameters(particle_index, parentIndex, sigma, epsilon, reductionFactor)

            # Deal with exceptions here.
            # TODO

    return system

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # Test the alchemy code.

    import os, os.path

    # Create system
    system = openmm.System()

    timestep = 2.0 * units.femtoseconds

    # Set whether to use GB model or not.
    gbmodel = 'OBC' # use OBC GBSA (gives finite error)
    #gbmodel = 'none' # no solvent model (gives zero error)
    
    # PARAMETERS
    directory = 'benzene-toluene'
    receptor_atoms = range(0,12) # benzene
    ligand_atoms = range(12,27) # toluene
    platform_names = ['Reference', 'Cuda', 'OpenCL']

    # For p-xylene binding to T4 lysozyme L99A system.
    #directory = 'p-xylene'
    #receptor_atoms = range(0,2603) # T4 lysozyme L99A
    #ligand_atoms = range(2603,2621) # p-xylene

    complex_prmtop_filename = os.path.join(directory, 'complex.prmtop')
    complex_crd_filename = os.path.join(directory, 'complex.crd')
    receptor_prmtop_filename = os.path.join(directory, 'receptor.prmtop')
    receptor_crd_filename = os.path.join(directory, 'receptor.crd')

    # Create force type variants of systems.
    force_types = ['standard', 'softcore-full', 'softcore-half', 'softcore-annihilated', 'custom-full', 'custom-half', 'custom-annihilated'] # Different force types
    system_types = ['complex', 'receptor']
    systems = dict()
    for system_type in system_types:
        systems[system_type] = dict()        
    # Load standard systems.
    import simtk.pyopenmm.amber.amber_file_parser as amber    
    systems['complex']['standard'] = amber.readAmberSystem(complex_prmtop_filename, mm=openmm, gbmodel=gbmodel, shake='h-bonds')
    systems['receptor']['standard'] = amber.readAmberSystem(receptor_prmtop_filename, mm=openmm, gbmodel=gbmodel, shake='h-bonds')
    # Create softcore force type variants.
    for system_type in system_types:
        systems[system_type]['softcore-full'] = build_softcore_system(systems[system_type]['standard'], ligand_atoms, 1.0, 1.0)
        systems[system_type]['softcore-half'] = build_softcore_system(systems[system_type]['standard'], ligand_atoms, 0.5, 1.0)
        systems[system_type]['softcore-annihilated'] = build_softcore_system(systems[system_type]['standard'], ligand_atoms, 0.0, 0.0)
    # Create custom force type variants.
    for system_type in system_types:
        systems[system_type]['custom-full'] = build_custom_system(systems[system_type]['standard'], ligand_atoms, 1.0, 1.0)
        systems[system_type]['custom-half'] = build_custom_system(systems[system_type]['standard'], ligand_atoms, 0.5, 1.0)
        systems[system_type]['custom-annihilated'] = build_custom_system(systems[system_type]['standard'], ligand_atoms, 0.0, 0.0)

    # Read coordinates.
    coordinates = dict()
    coordinates['complex'] = amber.readAmberCoordinates(complex_crd_filename, asNumpy=True)
    coordinates['receptor'] = coordinates['complex'][receptor_atoms,:]

    # Shift ligand atoms.
    coordinates['complex'][ligand_atoms,:] += 1.8 * units.angstroms

    #==============================================================================================
    # Compute potentials.
    #==============================================================================================
        
    potential = dict()
    for platform_name in platform_names:
        potential[platform_name] = dict()

        # Select platform to use.
        platform = openmm.Platform.getPlatformByName(platform_name)

        for system_type in system_types:
            potential[platform_name][system_type] = dict()

            for force_type in force_types:
                potential[platform_name][system_type][force_type] = dict()                        

                # Compute full complex energy.
                print "Computing potential for system '%(system_type)s' force type '%(force_type)s' on platform '%(platform_name)s'" % vars()
                try:                
                    integrator = openmm.VerletIntegrator(timestep)
                    context = openmm.Context(systems[system_type][force_type], integrator, platform)
                    context.setPositions(coordinates[system_type])
                    state = context.getState(getEnergy=True)
                    potential[platform_name][system_type][force_type] = state.getPotentialEnergy()
                    del integrator, context
                except Exception as e:
                    print e
                    potential[platform_name][system_type][force_type] = None

    #==============================================================================================
    # Show results.
    #==============================================================================================

    print "%24s : " % "Platforms",
    for platform_name in platform_names: print "%31s" % (platform_name + '       '),
    print ""

    # fully interacting
    print "-----------------------------"
    print "Interacting ligand in complex"
    for force_type in ['standard', 'softcore-full', 'custom-full']:
        print "%24s : " % (force_type),
        for platform_name in platform_names:
            energy = potential[platform_name]['complex'][force_type]
            if energy is None:
                print "%24s       " % "N/A",                
            else:
                print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),
        print ""

    # non-interacting complex
    print "-----------------------------"
    print "Annihilated ligand in complex - should be same as 'receptor only' below"

    for force_type in ['softcore-annihilated', 'custom-annihilated']:
        print "%24s : " % (force_type),
        for platform_name in platform_names:
            energy = potential[platform_name]['complex'][force_type]
            if energy is None:
                print "%24s       " % "N/A",
            else:
                print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),
        print ""

    # non-interacting complex
    print "-----------------------------"
    print "Partially annihilated ligand in complex - should match between custom and softcore"

    for force_type in ['softcore-half', 'custom-half']:
        print "%24s : " % (force_type),
        for platform_name in platform_names:
            energy = potential[platform_name]['complex'][force_type]
            if energy is None:
                print "%24s       " % "N/A",
            else:
                print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),
        print ""

    print "-----------------------------"
    print "Receptor only (no ligand)"
    
    for force_type in force_types:
        print "%24s : " % (force_type),
        for platform_name in platform_names:
            energy = potential[platform_name]['receptor'][force_type]
            if energy is None:
                print "%24s       " % "N/A",                
            else:
                print "%24.8f kJ/mol" % (energy / units.kilojoules_per_mole),
        print ""

    #==============================================================================================
    # Run timings
    #==============================================================================================

    nsteps = 50

    elapsed_times = dict() # elapsed_times[platform_name] is time for nsteps of dynamics
    system_type = 'complex'
    for platform_name in platform_names:
        # Select platform to use.
        platform = openmm.Platform.getPlatformByName(platform_name)

        elapsed_times[platform_name] = dict()
        for force_type in force_types:

            print "Running system '%(system_type)s' force type '%(force_type)s' for %(nsteps)d steps of dynamics on platform '%(platform_name)s'..." % vars()
            try:
                integrator = openmm.VerletIntegrator(timestep)
                context = openmm.Context(systems[system_type][force_type], integrator, platform)
                context.setPositions(coordinates[system_type])
                initial_time = time.time()
                integrator.step(nsteps)
                final_time = time.time()
                elapsed_time = final_time - initial_time
                elapsed_times[platform_name][force_type] = elapsed_time
                del integrator, context
            except:
                elapsed_times[platform_name][force_type] = None

    print "-----------------------------"
    print "Timings for system '%s'" % (system_type)
    
    for force_type in force_types:
        print "%48s : " % (force_type),
        for platform_name in platform_names:
            elapsed_time = elapsed_times[platform_name][force_type]
            if elapsed_time is None:
                print "%24s%24s      " % ("N/A",""),
            else:
                ns_per_day =  (float(nsteps) * timestep / (elapsed_time * units.seconds)) / (units.nanoseconds / units.day)                
                print "%12.3f s (%12.6f ns/day)                  " % (elapsed_time, ns_per_day),
        print ""
