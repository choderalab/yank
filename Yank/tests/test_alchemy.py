#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Tests for alchemical factory in `alchemy.py`.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import numpy as np
import time
from functools import partial

import simtk.openmm as openmm
import simtk.unit as units
from simtk.openmm import app

from nose.plugins.attrib import attr
import pymbar

import logging
logger = logging.getLogger(__name__)

from openmmtools import testsystems

from yank import alchemy
import yank.alchemy
from yank.alchemy import AlchemicalState, AbsoluteAlchemicalFactory

from nose.plugins.skip import Skip, SkipTest

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant
temperature = 300.0 * units.kelvin # reference temperature
MAX_DELTA = 0.01 * kB * temperature # maximum allowable deviation

#=============================================================================================
# SUBROUTINES FOR TESTING
#=============================================================================================

def compareSystemEnergies(positions, systems, descriptions, platform=None, precision=None):
    # Compare energies.
    timestep = 1.0 * units.femtosecond

    if platform:
        platform_name = platform.getName()
        if precision:
            if platform_name == 'CUDA':
                platform.setDefaultPropertyValue('CudaPrecision', precision)
            elif platform_name == 'OpenCL':
                platform.setDefaultPropertyValue('OpenCLPrecision', precision)

    potentials = list()
    states = list()
    for system in systems:
        integrator = openmm.VerletIntegrator(timestep)
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getPositions=True)
        potential = state.getPotentialEnergy()
        potentials.append(potential)
        states.append(state)
        del context, integrator

    logger.info("========")
    for i in range(len(systems)):
        logger.info("%32s : %24.8f kcal/mol" % (descriptions[i], potentials[i] / units.kilocalories_per_mole))
        if (i > 0):
            delta = potentials[i] - potentials[0]
            logger.info("%32s : %24.8f kcal/mol" % ('ERROR', delta / units.kilocalories_per_mole))
            if (abs(delta) > MAX_DELTA):
                raise Exception("Maximum allowable deviation (%24.8f kcal/mol) exceeded; test failed." % (MAX_DELTA / units.kilocalories_per_mole))

    return potentials

def alchemical_factory_check(reference_system, positions, receptor_atoms, ligand_atoms, platform_name=None, annihilate_electrostatics=True, annihilate_sterics=False, precision=None):
    """
    Compare energies of reference system and fully-interacting alchemically modified system.

    ARGUMENTS

    reference_system (simtk.openmm.System) - the reference System object to compare with
    positions - the positions to assess energetics for
    receptor_atoms (list of int) - the list of receptor atoms
    ligand_atoms (list of int) - the list of ligand atoms to alchemically modify
    precision : str, optional, default=None
       Precision model, or default if not specified. ('single', 'double', 'mixed')

    """

    # Create a factory to produce alchemical intermediates.
    logger.info("Creating alchemical factory...")
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms, annihilate_electrostatics=annihilate_electrostatics, annihilate_sterics=annihilate_sterics)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    logger.info("AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time)

    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    alchemical_system = factory.createPerturbedSystem(AlchemicalState())

    # Serialize for debugging.
    with open('system.xml', 'w') as outfile:
        outfile.write(alchemical_system.__getstate__())

    compareSystemEnergies(positions, [reference_system, alchemical_system], ['reference', 'alchemical'], platform=platform, precision=precision)

    return

def benchmark(reference_system, positions, receptor_atoms, ligand_atoms, platform_name=None, annihilate_electrostatics=True, annihilate_sterics=False, nsteps=500, timestep=1.0*units.femtoseconds):
    """
    Benchmark performance of alchemically modified system relative to original system.

    Parameters
    ----------
    reference_system : simtk.openmm.System
       The reference System object to compare with
    positions : simtk.unit.Quantity with units compatible with nanometers
       The positions to assess energetics for.
    receptor_atoms : list of int
       The list of receptor atoms.
    ligand_atoms : list of int
       The list of ligand atoms to alchemically modify.
    platform_name : str, optional, default=None
       The name of the platform to use for benchmarking.
    annihilate_electrostatics : bool, optional, default=True
       If True, electrostatics will be annihilated; if False, decoupled.
    annihilate_sterics : bool, optional, default=False
       If True, sterics will be annihilated; if False, decoupled.
    nsteps : int, optional, default=500
       Number of molecular dynamics steps to use for benchmarking.
    timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional, default=1*femtoseconds
       Timestep to use for benchmarking.

    """

    # Create a factory to produce alchemical intermediates.
    logger.info("Creating alchemical factory...")
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms, annihilate_electrostatics=annihilate_electrostatics, annihilate_sterics=annihilate_sterics)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    logger.info("AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time)

    # Create an alchemically-perturbed state corresponding to nearly fully-interacting.
    # NOTE: We use a lambda slightly smaller than 1.0 because the AlchemicalFactory does not use Custom*Force softcore versions if lambda = 1.0 identically.
    lambda_value = 1.0 - 1.0e-6
    alchemical_state = AlchemicalState(lambda_coulomb=lambda_value, lambda_sterics=lambda_value, lambda_torsions=lambda_value)

    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    # Create the perturbed system.
    logger.info("Creating alchemically-modified state...")
    initial_time = time.time()
    alchemical_system = factory.createPerturbedSystem(alchemical_state)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    # Compare energies.
    logger.info("Computing reference energies...")
    reference_integrator = openmm.VerletIntegrator(timestep)
    if platform:
        reference_context = openmm.Context(reference_system, reference_integrator, platform)
    else:
        reference_context = openmm.Context(reference_system, reference_integrator)
    reference_context.setPositions(positions)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()
    logger.info("Computing alchemical energies...")
    alchemical_integrator = openmm.VerletIntegrator(timestep)
    if platform:
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, platform)
    else:
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
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

def overlap_check(reference_system, positions, receptor_atoms, ligand_atoms, platform_name=None, annihilate_electrostatics=True, annihilate_sterics=False, precision=None, nsteps=50, nsamples=200):
    """
    Test overlap between reference system and alchemical system by running a short simulation.

    Parameters
    ----------
    reference_system : simtk.openmm.System
       The reference System object to compare with
    positions : simtk.unit.Quantity with units compatible with nanometers
       The positions to assess energetics for.
    receptor_atoms : list of int
       The list of receptor atoms.
    ligand_atoms : list of int
       The list of ligand atoms to alchemically modify.
    platform_name : str, optional, default=None
       The name of the platform to use for benchmarking.
    annihilate_electrostatics : bool, optional, default=True
       If True, electrostatics will be annihilated; if False, decoupled.
    annihilate_sterics : bool, optional, default=False
       If True, sterics will be annihilated; if False, decoupled.
    nsteps : int, optional, default=50
       Number of molecular dynamics steps between samples.
    nsamples : int, optional, default=100
       Number of samples to collect.

    """

    # Create a fully-interacting alchemical state.
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    alchemical_state = AlchemicalState()
    alchemical_system = factory.createPerturbedSystem(alchemical_state)

    temperature = 300.0 * units.kelvin
    collision_rate = 5.0 / units.picoseconds
    timestep = 2.0 * units.femtoseconds
    kT = (kB * temperature)

    # Select platform.
    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    # Create integrators.
    reference_integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    alchemical_integrator = openmm.VerletIntegrator(timestep)

    # Create contexts.
    if platform:
        reference_context = openmm.Context(reference_system, reference_integrator, platform)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, platform)
    else:
        reference_context = openmm.Context(reference_system, reference_integrator)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)

    # Collect simulation data.
    reference_context.setPositions(positions)
    du_n = np.zeros([nsamples], np.float64) # du_n[n] is the
    for sample in range(nsamples):
        # Run dynamics.
        reference_integrator.step(nsteps)

        # Get reference energies.
        reference_state = reference_context.getState(getEnergy=True, getPositions=True)
        reference_potential = reference_state.getPotentialEnergy()

        # Get alchemical energies.
        alchemical_context.setPositions(reference_state.getPositions())
        alchemical_state = alchemical_context.getState(getEnergy=True)
        alchemical_potential = alchemical_state.getPotentialEnergy()

        du_n[sample] = (alchemical_potential - reference_potential) / kT

    # Clean up.
    del reference_context, alchemical_context

    # Discard data to equilibration and subsample.
    from pymbar import timeseries
    [t0, g, Neff] = timeseries.detectEquilibration(du_n)
    indices = timeseries.subsampleCorrelatedData(du_n, g=g)
    du_n = du_n[indices]

    # Compute statistics.
    from pymbar import EXP
    [DeltaF, dDeltaF] = EXP(du_n)

    # Raise an exception if the error is larger than 3kT.
    MAX_DEVIATION = 3.0 # kT
    if (dDeltaF > MAX_DEVIATION):
        report = "DeltaF = %12.3f +- %12.3f kT (%5d samples, g = %6.1f)" % (DeltaF, dDeltaF, Neff, g)
        raise Exception(report)

    return

def rstyle(ax):
    '''Styles x,y axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been
    carried out (needs to know final tick spacing)

    From:
    http://nbviewer.ipython.org/github/wrobstory/climatic/blob/master/examples/ggplot_styling_for_matplotlib.ipynb
    '''
    import pylab
    import matplotlib
    import matplotlib.pyplot as plt

    #Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.90')
    ax.set_axisbelow(True)

    #Set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator((pylab.MultipleLocator((plt.xticks()[0][1]
                                -plt.xticks()[0][0]) / 2.0 )))
    ax.yaxis.set_minor_locator((pylab.MultipleLocator((plt.yticks()[0][1]
                                -plt.yticks()[0][0]) / 2.0 )))

    #Remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)

    #Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    #Remove the minor tick lines
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    #Only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def lambda_trace(reference_system, positions, receptor_atoms, ligand_atoms, platform_name=None, precision=None, annihilate_electrostatics=True, annihilate_sterics=False, nsteps=100):
    """
    Compute potential energy as a function of lambda.

    """
    # Create a factory to produce alchemical intermediates.
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms, annihilate_electrostatics=annihilate_electrostatics, annihilate_sterics=annihilate_sterics)

    platform = None
    if platform_name:
        # Get platform.
        platform = openmm.Platform.getPlatformByName(platform_name)

    if precision:
        if platform_name == 'CUDA':
            platform.setDefaultPropertyValue('CudaPrecision', precision)
        elif platform_name == 'OpenCL':
            platform.setDefaultPropertyValue('OpenCLPrecision', precision)

    # Take equally-sized steps.
    delta = 1.0 / nsteps

    def compute_potential(system, positions, platform=None):
        timestep = 1.0 * units.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()
        del integrator, context
        return potential

    # Compute unmodified energy.
    u_original = compute_potential(reference_system, positions, platform)

    # Scan through lambda values.
    lambda_i = np.zeros([nsteps+1], np.float64) # lambda values for u_i
    u_i = units.Quantity(np.zeros([nsteps+1], np.float64), units.kilocalories_per_mole) # u_i[i] is the potential energy for lambda_i[i]
    for i in range(nsteps+1):
        lambda_value = 1.0-i*delta # compute lambda value for this step
        alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_coulomb=lambda_value, lambda_sterics=lambda_value, lambda_torsions=lambda_value))
        lambda_i[i] = lambda_value
        u_i[i] = compute_potential(alchemical_system, positions, platform)
        logger.info("%12.9f %24.8f kcal/mol" % (lambda_i[i], u_i[i] / units.kilocalories_per_mole))

    # Write figure as PDF.
    import pylab
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages('lambda-trace.pdf') as pdf:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        plt.plot(1, u_original / units.kilocalories_per_mole, 'ro', label='unmodified')
        plt.plot(lambda_i, u_i / units.kilocalories_per_mole, 'k.', label='alchemical')
        plt.title('T4 lysozyme L99A + p-xylene : AMBER96 + OBC GBSA')
        plt.ylabel('potential (kcal/mol)')
        plt.xlabel('lambda')
        ax.legend()
        rstyle(ax)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    return

def generate_trace(test_system):
    lambda_trace(test_system['test'].system, test_system['test'].positions, test_system['receptor_atoms'], test_system['ligand_atoms'])
    return

#=============================================================================================
# TEST SYSTEM DEFINITIONS
#=============================================================================================

test_systems = dict()
test_systems['Lennard-Jones cluster'] = {
    'test' : testsystems.LennardJonesCluster(),
    'ligand_atoms' : range(0,1), 'receptor_atoms' : range(1,2) }
test_systems['Lennard-Jones fluid without dispersion correction'] = {
    'test' : testsystems.LennardJonesFluid(dispersion_correction=False),
    'ligand_atoms' : range(0,1), 'receptor_atoms' : range(1,2) }
test_systems['Lennard-Jones fluid with dispersion correction'] = {
    'test' : testsystems.LennardJonesFluid(dispersion_correction=True),
    'ligand_atoms' : range(0,1), 'receptor_atoms' : range(1,2) }
test_systems['TIP3P with reaction field, no charges, no switch, no dispersion correction'] = {
    'test' : testsystems.DischargedWaterBox(dispersion_correction=False, switch=False, nonbondedMethod=app.CutoffPeriodic),
    'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }
test_systems['TIP3P with reaction field, switch, no dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=False, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }
test_systems['TIP3P with reaction field, no switch, dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=True, switch=False, nonbondedMethod=app.CutoffPeriodic),
    'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }
test_systems['TIP3P with reaction field, switch, dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=True, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }
test_systems['alanine dipeptide in vacuum'] = {
    'test' : testsystems.AlanineDipeptideVacuum(),
    'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22) }
test_systems['alanine dipeptide in OBC GBSA'] = {
    'test' : testsystems.AlanineDipeptideImplicit(),
    'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22) }
test_systems['alanine dipeptide in TIP3P with reaction field'] = {
    'test' : testsystems.AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic),
    'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22) }
test_systems['T4 lysozyme L99A with p-xylene in OBC GBSA'] = {
    'test' : testsystems.LysozymeImplicit(),
    'ligand_atoms' : range(2603,2621), 'receptor_atoms' : range(0,2603) }

# Problematic tests: PME is not fully implemented yet
#test_systems['TIP3P with PME, no switch, no dispersion correction'] = {
#    'test' : testsystems.WaterBox(dispersion_correction=False, switch=False, nonbondedMethod=app.PME),
#    'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }

# Slow tests
#test_systems['Src in OBC GBSA'] = {
#    'test' : testsystems.SrcImplicit(),
#    'ligand_atoms' : range(0,21), 'receptor_atoms' : range(21,4091) }
#test_systems['Src in TIP3P with reaction field'] = {
#    'test' : testsystems.SrcExplicit(nonbondedMethod=app.CutoffPeriodic),
#    'ligand_atoms' : range(0,21), 'receptor_atoms' : range(21,4091) }

fast_testsystem_names = [
    'Lennard-Jones cluster',
    'Lennard-Jones fluid without dispersion correction',
    'Lennard-Jones fluid with dispersion correction',
    'TIP3P with reaction field, no charges, no switch, no dispersion correction',
    'TIP3P with reaction field, switch, no dispersion correction',
    'TIP3P with reaction field, switch, dispersion correction',
#    'TIP3P with PME, no switch, no dispersion correction' # PME still problematic
    ]


#=============================================================================================
# NOSETEST GENERATORS
#=============================================================================================

@attr('slow')
def test_overlap():
    """
    Generate nose tests for overlap for all alchemical test systems.
    """
    for name in fast_testsystem_names:
        test_system = test_systems[name]
        reference_system = test_system['test'].system
        positions = test_system['test'].positions
        ligand_atoms = test_system['ligand_atoms']
        receptor_atoms = test_system['receptor_atoms']
        f = partial(overlap_check, reference_system, positions, receptor_atoms, ligand_atoms)
        f.description = "Testing reference/alchemical overlap for %s..." % name
        yield f

    return

def test_alchemical_accuracy():
    """
    Generate nose tests for overlap for all alchemical test systems.
    """
    for name in test_systems.keys():
        test_system = test_systems[name]
        reference_system = test_system['test'].system
        positions = test_system['test'].positions
        ligand_atoms = test_system['ligand_atoms']
        receptor_atoms = test_system['receptor_atoms']
        f = partial(alchemical_factory_check, reference_system, positions, receptor_atoms, ligand_atoms)
        f.description = "Testing alchemical fidelity of %s..." % name
        yield f

    return

#=============================================================================================
# MAIN FOR MANUAL DEBUGGING
#=============================================================================================

if __name__ == "__main__":
    #generate_trace(test_systems['TIP3P with reaction field, switch, dispersion correction'])

    test_systems = dict()
    test_systems['Src in TIP3P with reaction field'] = {
        'test' : testsystems.SrcExplicit(nonbondedMethod=app.CutoffPeriodic),
        'ligand_atoms' : range(0,21), 'receptor_atoms' : range(21,4091) }
    name = 'Src in TIP3P with reaction field'
    test_system = test_systems[name]
    reference_system = test_system['test'].system
    positions = test_system['test'].positions
    ligand_atoms = test_system['ligand_atoms']
    receptor_atoms = test_system['receptor_atoms']
    alchemical_factory_check(reference_system, positions, receptor_atoms, ligand_atoms)

