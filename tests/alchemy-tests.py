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

import numpy as np
import time

import simtk.openmm as openmm
import simtk.unit as units

import logging
logger = logging.getLogger(__name__)

from alchemy import AlchemicalState, AbsoluteAlchemicalFactory
from yanktools import testsystems

#=============================================================================================
# MAIN AND UNIT TESTS
#=============================================================================================

def compareSystemEnergies(positions, systems, descriptions, platform=None):
    # Compare energies.
    timestep = 1.0 * units.femtosecond
    
    potentials = list()
    for system in systems:
        integrator = openmm.VerletIntegrator(timestep)
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()    
        potentials.append(potential)

    logger.info("========")
    for i in range(len(systems)):
        logger.info("%32s : %24.8f kcal/mol" % (descriptions[i], potentials[i] / units.kilocalories_per_mole))
        if (i > 0):
            delta = potentials[i] - potentials[0]
            logger.info("%32s : %24.8f kcal/mol" % ('ERROR', delta / units.kilocalories_per_mole))

    return potentials

def testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms, platform_name='CUDA', annihilateElectrostatics=True, annihilateLennardJones=False):
    """
    Compare energies of reference system and fully-interacting alchemically modified system.

    ARGUMENTS
    
    reference_system (simtk.openmm.System) - the reference System object to compare with
    positions - the positions to assess energetics for
    receptor_atoms (list of int) - the list of receptor atoms 
    ligand_atoms (list of int) - the list of ligand atoms to alchemically modify

    """

    # Create a factory to produce alchemical intermediates.
    logger.info("Creating alchemical factory...")
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    logger.info("AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time)

    platform = openmm.Platform.getPlatformByName(platform_name)

    delta = 0.001
    delta = 1.0e-6

    compareSystemEnergies(positions, [reference_system, factory.createPerturbedSystem(AlchemicalState(0, 1-delta, 1, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones))], ['reference', 'partially discharged'], platform=platform)
    compareSystemEnergies(positions, [factory.createPerturbedSystem(AlchemicalState(0, delta, 1, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones)), factory.createPerturbedSystem(AlchemicalState(0, 0.0, 1, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones))], ['partially charged', 'discharged'], platform=platform)
    compareSystemEnergies(positions, [factory.createPerturbedSystem(AlchemicalState(0, 0, 1, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones)), factory.createPerturbedSystem(AlchemicalState(0, 0, 1-delta, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones))], ['discharged', 'partially decoupled'], platform=platform)
    compareSystemEnergies(positions, [factory.createPerturbedSystem(AlchemicalState(0, 0, delta, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones)), factory.createPerturbedSystem(AlchemicalState(0, 0, 0, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones))], ['partially coupled', 'decoupled'], platform=platform)

    return
    

def benchmark(reference_system, positions, receptor_atoms, ligand_atoms, platform_name='CUDA', annihilateElectrostatics=True, annihilateLennardJones=False, nsteps=500):
    """
    Benchmark performance relative to unmodified system.

    ARGUMENTS
    
    reference_system (simtk.openmm.System) - the reference System object to compare with
    positions - the positions to assess energetics for
    receptor_atoms (list of int) - the list of receptor atoms 
    ligand_atoms (list of int) - the list of ligand atoms to alchemically modify

    """

    # Create a factory to produce alchemical intermediates.
    logger.info("Creating alchemical factory...")
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    logger.info("AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time)

    # Create an alchemically-perturbed state corresponding to nearly fully-interacting.
    # NOTE: We use a lambda slightly smaller than 1.0 because the AlchemicalFactory does not use Custom*Force softcore versions if lambda = 1.0 identically.
    lambda_value = 1.0 - 1.0e-6
    alchemical_state = AlchemicalState(0.00, lambda_value, lambda_value, lambda_value)
    alchemical_state.annihilateElectrostatics = annihilateElectrostatics
    alchemical_state.annihilateLennardJones = annihilateLennardJones

    platform = openmm.Platform.getPlatformByName(platform_name)
    
    # Create the perturbed system.
    logger.info("Creating alchemically-modified state...")
    initial_time = time.time()
    alchemical_system = factory.createPerturbedSystem(alchemical_state)    
    final_time = time.time()
    elapsed_time = final_time - initial_time
    # Compare energies.
    timestep = 1.0 * units.femtosecond
    logger.info("Computing reference energies...")
    reference_integrator = openmm.VerletIntegrator(timestep)
    reference_context = openmm.Context(reference_system, reference_integrator, platform)
    reference_context.setPositions(positions)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()    
    logger.info("Computing alchemical energies...")
    alchemical_integrator = openmm.VerletIntegrator(timestep)
    alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, platform)
    alchemical_context.setPositions(positions)
    alchemical_state = alchemical_context.getState(getEnergy=True)
    alchemical_potential = alchemical_state.getPotentialEnergy()
    delta = alchemical_potential - reference_potential 

    # Make sure all kernels are compiled.
    reference_integrator.step(1)
    alchemical_integrator.step(1)

    # Time simulations.
    logger.info("Simulating reference system...")
    initial_time = time.time()
    reference_integrator.step(nsteps)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()    
    final_time = time.time()
    reference_time = final_time - initial_time
    logger.info("Simulating alchemical system...")
    initial_time = time.time()
    alchemical_integrator.step(nsteps)
    alchemical_state = alchemical_context.getState(getEnergy=True)
    alchemical_potential = alchemical_state.getPotentialEnergy()    
    final_time = time.time()
    alchemical_time = final_time - initial_time

    logger.info("TIMINGS")
    logger.info("reference system       : %12.3f s for %8d steps (%12.3f ms/step)" % (reference_time, nsteps, reference_time/nsteps*1000))
    logger.info("alchemical system      : %12.3f s for %8d steps (%12.3f ms/step)" % (alchemical_time, nsteps, alchemical_time/nsteps*1000))
    logger.info("alchemical simulation is %12.3f x slower than unperturbed system" % (alchemical_time / reference_time))

    return delta

def test_overlap():
    """
    BUGS TO REPORT:
    * Even if epsilon = 0, energy of two overlapping atoms is 'nan'.
    * Periodicity in 'nan' if dr = 0.1 even in nonperiodic system
    """

    # Create a reference system.    

    logger.info("Creating Lennard-Jones cluster system...")
    #[reference_system, positions] = testsystems.LennardJonesFluid()
    #receptor_atoms = [0]
    #ligand_atoms = [1]

    system_container = testsystems.LysozymeImplicit()
    (reference_system, positions) = system_container.system, system_container.positions
    receptor_atoms = range(0,2603) # T4 lysozyme L99A
    ligand_atoms = range(2603,2621) # p-xylene

    unit = positions.unit
    positions = units.Quantity(np.array(positions / unit), unit)

    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    alchemical_state = AlchemicalState(0.00, 0.00, 0.00, 1.0)

    # Create the perturbed system.
    logger.info("Creating alchemically-modified state...")
    alchemical_system = factory.createPerturbedSystem(alchemical_state)    
    # Compare energies.
    timestep = 1.0 * units.femtosecond
    logger.info("Computing reference energies...")
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(reference_system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    reference_potential = state.getPotentialEnergy()    
    del state, context, integrator
    logger.info(reference_potential)
    logger.info("Computing alchemical energies...")
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(alchemical_system, integrator)
    dr = 0.1 * units.angstroms # TODO: Why does 0.1 cause periodic 'nan's?
    a = receptor_atoms[-1]
    b = ligand_atoms[-1]
    delta = positions[a,:] - positions[b,:]
    for k in range(3):
        positions[ligand_atoms,k] += delta[k]
    for i in range(30):
        r = dr * i
        positions[ligand_atoms,0] += dr
          
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        alchemical_potential = state.getPotentialEnergy()    
        logger.info("%8.3f A : %f " % (r / units.angstroms, alchemical_potential / units.kilocalories_per_mole))
    del state, context, integrator

    return

def lambda_trace(reference_system, positions, receptor_atoms, ligand_atoms, platform_name='CUDA', annihilateElectrostatics=True, annihilateLennardJones=False, nsteps=50):
    """
    Compute potential energy as a function of lambda.

    """
    # Create a factory to produce alchemical intermediates.
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)

    # Get platform.
    platform = openmm.Platform.getPlatformByName(platform_name)
    
    delta = 1.0 / nsteps

    def compute_potential(system, positions, platform):
        timestep = 1.0 * units.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()    
        del integrator, context
        return potential
        
    # discharging
    outfile = open('discharging-trace.out', 'w')
    for i in range(nsteps+1):
        lambda_value = 1.0-i*delta
        alchemical_system = factory.createPerturbedSystem(AlchemicalState(0, lambda_value, 1, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones))
        potential = compute_potential(alchemical_system, positions, platform)
        line = '%12.6f %24.6f' % (lambda_value, potential / units.kilocalories_per_mole)
        outfile.write(line + '\n')
        logger.info(line)
    outfile.close()

    # decoupling
    outfile = open('decoupling-trace.out', 'w')
    for i in range(nsteps+1):
        lambda_value = 1.0-i*delta
        alchemical_system = factory.createPerturbedSystem(AlchemicalState(0, 0, lambda_value, 1, annihilateElectrostatics=annihilateElectrostatics, annihilateLennardJones=annihilateLennardJones))
        potential = compute_potential(alchemical_system, positions, platform)
        line = '%12.6f %24.6f' % (lambda_value, potential / units.kilocalories_per_mole)
        outfile.write(line + '\n')
        logger.info(line)
    outfile.close()

    return

def test_intermediates():

    if False:
        # DEBUG
        logger.info("Creating T4 lysozyme system...")
        system_container = testsystems.LysozymeImplicit()
        (reference_system, positions) = system_container.system, system_container.positions        
        receptor_atoms = range(0,2603) # T4 lysozyme L99A
        ligand_atoms = range(2603,2621) # p-xylene
        lambda_trace(reference_system, positions, receptor_atoms, ligand_atoms)    
        testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)    
        benchmark(reference_system, positions, receptor_atoms, ligand_atoms)    
        logger.info("")
        stop

    # Run tests on individual systems.

    logger.info("Creating T4 lysozyme system...")
    system_container = testsystems.LysozymeImplicit()
    (reference_system, positions) = system_container.system, system_container.positions    
    receptor_atoms = range(0,2603) # T4 lysozyme L99A
    ligand_atoms = range(2603,2621) # p-xylene
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)    
    benchmark(reference_system, positions, receptor_atoms, ligand_atoms)    
    logger.info("")

    logger.info("Creating Lennard-Jones fluid system without dispersion correction...")
    system_container = testsystems.LennardJonesFluid(dispersion_correction=False)
    (reference_system, positions) = system_container.system, system_container.positions
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(2,3) # second atom
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)
    logger.info("")

    logger.info("Creating Lennard-Jones fluid system with dispersion correction...")
    
    system_container = testsystems.LennardJonesFluid(dispersion_correction=True)
    (reference_system, positions) = system_container.system, system_container.positions
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(2,3) # second atom
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)
    benchmark(reference_system, positions, receptor_atoms, ligand_atoms)    
    logger.info("")

    logger.info("Creating Lennard-Jones cluster...")
    system_container = testsystems.LennardJonesCluster()
    (reference_system, positions) = system_container.system, system_container.positions
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(1,2) # second atom
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)
    logger.info("")

    logger.info("Creating alanine dipeptide implicit system...")
    
    system_container = testsystems.AlanineDipeptideImplicit()
    (reference_system, positions) = system_container.system, system_container.positions
    ligand_atoms = range(0,4) # methyl group
    receptor_atoms = range(4,22) # rest of system
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)
    benchmark(reference_system, positions, receptor_atoms, ligand_atoms)    
    logger.info("")

    logger.info("Creating alanine dipeptide explicit system...")
    system_container = testsystems.AlanineDipeptideExplicit()
    (reference_system, positions) = system_container.system, system_container.positions
    ligand_atoms = range(0,22) # alanine residue
    receptor_atoms = range(22,25) # one water
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)
    benchmark(reference_system, positions, receptor_atoms, ligand_atoms)    
    logger.info("")

    logger.info("Creating alanine dipeptide explicit system without dispersion correction...")
    #forces = { reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index)) for index in range(reference_system.getNumForces()) } # requires Python 2.7 features
    forces = dict( (reference_system.getForce(index).__class__.__name__,  reference_system.getForce(index)) for index in range(reference_system.getNumForces()) ) # python 2.6 compatible
    forces['NonbondedForce'].setUseDispersionCorrection(False) # turn off dispersion correction
    ligand_atoms = range(0,22) # alanine residue
    receptor_atoms = range(22,25) # one water
    testAlchemicalFactory(reference_system, positions, receptor_atoms, ligand_atoms)
    logger.info("")


#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == "__main__":
    # Run overlap tests.
    #test_overlap()
    
    # Test energy accuracy of intermediates near lambda = 1.
    test_intermediates()
