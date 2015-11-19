#!/usr/bin/env python

"""
Compute the binding affinity of a peptide to an MHC:TCR complex.

@author John D. Chodera

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os, os.path
import yank
import mdtraj
import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
import logging
logger = logging.getLogger(__name__)

# ==============================================================================
# PARAMETERS
# ==============================================================================

solvent = 'explicit' # one of ['explicit', 'implicit']
solvent = 'implicit' # one of ['explicit', 'implicit']
verbose = True # if True, use logger

setup_dir = 'setup' # directory to put setup files in
store_dir = 'output' # directory to put output files in

pdb_filename = None
#pdbid = '1AO7' # HTLV1 LLFGYPVYV Tax Garboczi 1996 (1AO7), Ding 1999 - peptide agonist
pdbid = '1QSE' # HTLV1 Tax LLFGYPRYV V7R Ding 1999 (1QSE) - peptide agonist
chain_ids_to_keep = None # list of chains to retain, or None to keep all

# mdtraj DSL selection for components (after filtering to retain only desired chains)
component_dsl = {
    'ligand' : "chainid == 2",
    'receptor' : "chainid != 2",
    'complex' : "all",
    'solvent' : "water"
}

pH = 7.0 # pH to use in setting protein protonation states
keep_crystallographic_water = False # True if crystal waters are to be retained

# Forcefield and simulation parameters.
water_name = 'tip4pew' # water to use for explicit solvent
ffxmls = ['amber99sbildn.xml', water_name + '.xml'] # list of ffxml files to use
if solvent == 'implicit':
    ffxmls += ['amber99_obc.xml'] # Add GBSA if desired
padding = 10.0 * unit.angstroms
nonbonded_cutoff = 10.0 * unit.angstroms
if solvent == 'explicit':
    nonbonded_method = app.CutoffPeriodic # for explicit solvent
else:
    nonbonded_method = app.NoCutoff # for implicit solvent
constraints = app.HBonds

max_minimization_iterations = 250

temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
collision_rate = 5.0 / unit.picoseconds
barostat_frequency = 50

timestep = 2.0 * unit.femtoseconds
nsteps_per_iteration = 1250
niterations = 1000
nequiliterations = 0

minimize = True # Minimize structures

platform_name = 'CUDA'
precision_model = 'mixed'
platform = openmm.Platform.getPlatformByName(platform_name)
if platform_name == 'CUDA':
    platform.setPropertyDefaultValue('CudaPrecision', precision_model)
elif platform_name == 'OpenCL':
    platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)

# ==============================================================================
# SET UP STORAGE
# ==============================================================================

# Create directory to store files in.
workdir = os.path.join(os.getcwd(), setup_dir)
if not os.path.exists(workdir):
    os.makedirs(workdir)
    logger.info("Creating path %s" % workdir)

# Create directory to store files in.
outdir = os.path.join(os.getcwd(), store_dir)
if not os.path.exists(outdir):
    os.makedirs(outdir)
    logger.info("Creating path %s" % outdir)

# ==============================================================================
# CONFIGURE LOGGER
# ==============================================================================

from yank import utils
utils.config_root_logger(verbose, log_file_path=os.path.join(setup_dir, 'prepare.log'))

# ==============================================================================
# PREPARE STRUCTURE
# ==============================================================================

from pdbfixer import PDBFixer

is_periodic = (nonbonded_method not in [app.NoCutoff, app.CutoffNonPeriodic])

# ==============================================================================
# Retrieve the PDB file
# ==============================================================================

if pdb_filename:
    logger.info("Retrieving PDB '%s'..." % pdb_filename)
    fixer = PDBFixer(filename=pdb_filename)
else:
    logger.info("Retrieving PDB '%s'..." % pdbid)
    fixer = PDBFixer(pdbid=pdbid)

# ==============================================================================
# Prepare the structure
# ==============================================================================

# DEBUG
print "fixer.topology.chains(): %s" % str([ chain.id for chain in fixer.topology.chains() ])

# Write PDB file for solute only.
logger.info("Writing source PDB...")
pdb_filename = os.path.join(workdir, pdbid + '.pdb')
outfile = open(pdb_filename, 'w')
app.PDBFile.writeFile(fixer.topology, fixer.positions, outfile)
outfile.close()

if chain_ids_to_keep is not None:
    # Hack to get chain id to chain number mapping.
    chain_id_list = [c.chain_id for c in fixer.structure.models[0].chains]

    # Build list of chains to remove
    chain_numbers_to_remove = list()
    for (chain_number, chain_id) in enumerate(chain_id_list):
        if chain_id not in chain_ids_to_keep:
            chain_numbers_to_remove.append(chain_number)

    # Remove all but desired chains.
    logger.info("Removing chains...")
    fixer.removeChains(chain_numbers_to_remove)

# DEBUG
print "fixer.topology.chains(): %s" % str([ chain.id for chain in fixer.topology.chains() ])

# Add missing atoms and residues.
logger.info("Adding missing atoms and residues...")
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
#fixer.addMissingHydrogens(pH) # DEBUG
fixer.removeHeterogens(keepWater=keep_crystallographic_water)

# Write PDB file for completed output.
logger.info("Writing pdbfixer output...")
pdb_filename = os.path.join(workdir, 'pdbfixer.pdb')
outfile = open(pdb_filename, 'w')
app.PDBFile.writeFile(fixer.topology, fixer.positions, outfile)
outfile.close()

# ==============================================================================
# UTILITIES
# ==============================================================================

def write_file(filename, contents):
    with open(filename, 'w') as outfile:
        outfile.write(contents)

def solvate_and_minimize(topology, positions, phase=''):
    """
    Solvate the given system and minimize.

    Parameters
    ----------
    topology : simtk.openmm.Topology
        The topology
    positions : simtk.unit.Quantity of numpy.array of (natoms,3) with units compatible with nanometer.
        The positions
    phase : string, optional, default=''
        The phase prefix to prepend to written files.

    Returns
    -------
    topology : simtk.openmm.app.Topology
        The new Topology object
    system : simtk.openm.System
        The solvated system.
    positions : simtk.unit.Quantity of dimension natoms x 3 with units compatible with angstroms
        The minimized positions.

    """

    # Solvate (if desired) and create system.
    logger.info("Solvating...")
    forcefield = app.ForceField(*ffxmls)
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield=forcefield, pH=pH) # DEBUG
    if is_periodic:
        # Add solvent if in a periodic box.
        modeller.addSolvent(forcefield, padding=padding, model=water_name)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=constraints)
    if is_periodic:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency))
    logger.info("System has %d atoms." % system.getNumParticles())

    # DEBUG
    print "modeller.topology.chains(): %s" % str([ chain.id for chain in modeller.topology.chains() ])

    # Serialize to XML files.
    logger.info("Serializing to XML...")
    system_filename = os.path.join(workdir, 'system.xml')
    write_file(system_filename, openmm.XmlSerializer.serialize(system))

    if minimize:
        # Create simulation.
        logger.info("Creating simulation...")
        errorTol = 0.001
        integrator = openmm.VariableLangevinIntegrator(temperature, collision_rate, errorTol)
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Print potential energy.
        potential = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        logger.info("Potential energy is %12.3f kcal/mol" % (potential / unit.kilocalories_per_mole))

        # Write modeller positions.
        logger.info("Writing modeller output...")
        filename = os.path.join(workdir, phase + 'modeller.pdb')
        app.PDBFile.writeFile(simulation.topology, modeller.positions, open(filename, 'w'))

        # Minimize energy.
        logger.info("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=max_minimization_iterations)
        #integrator.step(max_minimization_iterations)
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()
        if np.isnan(potential_energy / unit.kilocalories_per_mole):
            raise Exception("Potential energy is NaN after minimization.")
        logger.info("Potential energy after minimiziation: %.3f kcal/mol" % (potential_energy / unit.kilocalories_per_mole))
        modeller.positions = simulation.context.getState(getPositions=True).getPositions()

        # Print potential energy.
        potential = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        logger.info("Potential energy is %12.3f kcal/mol" % (potential / unit.kilocalories_per_mole))

        # Write minimized positions.
        filename = os.path.join(workdir, phase + 'minimized.pdb')
        app.PDBFile.writeFile(simulation.topology, modeller.positions, open(filename, 'w'))

        # Clean up
        del simulation

    # Return the modeller instance.
    return [modeller.topology, system, modeller.positions]

# ==============================================================================
# PREPARE YANK CALCULATION
# ==============================================================================

from yank.yank import Yank

# Set options.
options = dict()
options['number_of_iterations'] = niterations
options['number_of_equilibration_iterations'] = nequiliterations
options['nsteps_per_iteration'] = nsteps_per_iteration
options['online_analysis'] = False
options['randomize_ligand'] = False
options['minimize'] = True
options['timestep'] = timestep
options['collision_rate'] = collision_rate
options['platform'] = platform

options['restraint_type'] = None
if not is_periodic:
    options['restraint_type'] = 'harmonic'

# Turn off MC ligand displacement.
options['mc_displacement_sigma'] = None

# Prepare phases of calculation.
phase_prefixes = ['solvent', 'complex'] # list of calculation phases (thermodynamic legs) to set up
components = ['ligand', 'receptor', 'solvent'] # components of the binding system
phase_prefixes = ['complex'] # DEBUG, since 'solvent' doesn't work yet
systems = dict() # systems[phase] is the System object associated with phase 'phase'
positions = dict() # positions[phase] is a list of coordinates associated with phase 'phase'
atom_indices = dict() # ligand_atoms[phase] is a list of ligand atom indices associated with phase 'phase'
for phase_prefix in phase_prefixes:
    # Retain the whole system
    if is_periodic:
        phase_suffix = 'explicit'
    else:
        phase_suffix = 'implicit'

    # Form phase name.
    phase = '%s-%s' % (phase_prefix, phase_suffix)
    logger.info("phase %s: " % phase)

    # Determine selection phrase for atom subset to be used in this phase.
    if phase_prefix == 'solvent':
        dsl_to_retain = component_dsl['ligand']
    elif phase_prefix == 'complex':
        dsl_to_retain = component_dsl['complex']
    else:
        raise Exception("unknown phase_prefix '%s'" % phase_prefix)

    # Use mdtraj to create a topology for the desired subset of the system.
    mdtraj_top = mdtraj.Topology.from_openmm(fixer.topology)
    #mdtraj_traj = mdtraj.Trajectory(xyz=[fixer.positions], topology=mdtraj_top)
    atom_indices_to_retain = mdtraj_top.select(dsl_to_retain)
    subset_topology_openmm = mdtraj_top.subset(atom_indices_to_retain).to_openmm()

    # DEBUG
    print "subset_topology_openmm.chains(): %s" % str([ chain.id for chain in subset_topology_openmm.chains() ])

    # Extract the positions of the corresponding atoms.
    x = np.array(fixer.positions / unit.angstrom)
    subset_positions = unit.Quantity(x[atom_indices_to_retain,:], unit.angstrom)

    # Solvate subset (if needed), create System, and minimize (if requested).
    [solvated_topology_openmm, solvated_system, solvated_positions] = solvate_and_minimize(subset_topology_openmm, subset_positions, phase=phase + '-')

    # DEBUG
    print "solvated_topology_openmm.chains(): %s" % str([ chain.id for chain in solvated_topology_openmm.chains() ])

    # Write minimized positions.
    filename = os.path.join(workdir, phase + '-initial.pdb')
    app.PDBFile.writeFile(solvated_topology_openmm, solvated_positions, open(filename, 'w'))

    # Record components.
    systems[phase] = solvated_system
    positions[phase] = solvated_positions

    # Identify various components.
    solvated_topology_mdtraj = mdtraj.Topology.from_openmm(solvated_topology_openmm)
    atom_indices[phase] = dict()
    for component in components:
        atom_indices[phase][component] = solvated_topology_mdtraj.select(component_dsl[component])

    # DEBUG
    print "Atom indices of ligand:"
    print atom_indices[phase]['ligand']

# Create list of phase names.
phases = systems.keys()

# Create reference thermodynamic state.
from yank.repex import ThermodynamicState # TODO: Fix this weird import path to something more sane, like 'from yank.repex import ThermodynamicState'
if is_periodic:
    thermodynamic_state = ThermodynamicState(temperature=temperature, pressure=pressure)
else:
    thermodynamic_state = ThermodynamicState(temperature=temperature)

# TODO: Select protocols.

# Create new simulation.
yank = Yank(store_dir, **options)
yank.create(phases, systems, positions, atom_indices, thermodynamic_state)

# TODO: Write PDB files
