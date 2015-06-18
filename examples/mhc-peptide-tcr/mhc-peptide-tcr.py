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
import pdbfixer
import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
import logging
logger = logging.getLogger(__name__)

# ==============================================================================
# PARAMETERS
# ==============================================================================

verbose = True # if True, use logger

setup_dir = 'setup' # directory to put setup files in
store_dir = 'output' # directory to put output files in

pdbid = '1AO7'
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
ffxmls += ['amber99_obc.xml'] # Add GBSA if desired
padding = 10.0 * unit.angstroms
nonbonded_cutoff = 10.0 * unit.angstroms
#nonbonded_method = app.CutoffPeriodic # for explicit solvent
nonbonded_method = app.NoCutoff # for implicit solvent
constraints = app.HBonds

max_minimization_iterations = 250

temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
collision_rate = 5.0 / unit.picoseconds
barostat_frequency = 50

timestep = 2.0 * unit.femtoseconds
nsteps_per_iteration = 500
niterations = 100
nequiliterations = 10

minimize = False

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

logger.info("Retrieving PDB '%s'..." % pdbid)
fixer = PDBFixer(pdbid=pdbid)

# ==============================================================================
# Prepare the structure
# ==============================================================================

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
    
# Add missing atoms and residues.
logger.info("Adding missing atoms and residues...")
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(pH)
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
    modeller : simtk.openmm.app.Modeller
        The Modeller instance.

    """

    # Solvate (if desired) and create system.
    logger.info("Solvating...")
    forcefield = app.ForceField(*ffxmls)
    modeller = app.Modeller(topology, positions)    
    if is_periodic:
        # Add solvent if in a periodic box.
        modeller.addSolvent(forcefield, padding=padding, model=water_name)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=constraints)
    if is_periodic:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

    if minimize:
        # Create simulation.
        logger.info("Creating simulation...")
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Write modeller positions.
        logger.info("Writing modeller output...")
        filename = os.path.join(workdir, phase + 'modeller.pdb')
        app.PDBFile.writeFile(simulation.topology, modeller.positions, open(filename, 'w'))

        # Minimize energy.
        logger.info("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=max_minimization_iterations)
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()
        if numpy.isnan(potential_energy / unit.kilocalories_per_mole):
            raise Exception("Potential energy is NaN after minimization.")
        logger.info("Potential energy after minimiziation: %.3f kcal/mol" % (potential_energy / unit.kilocalories_per_mole))
        modeller.positions = simulation.context.getState(getPositions=True).getPositions()

        # Write minimized positions.
        filename = os.path.join(workdir, phase + 'minimized.pdb')
        app.PDBFile.writeFile(simulation.topology, modeller.positions, open(filename, 'w'))

        # Serialize to XML files.
        logger.info("Serializing to XML...")
        system_filename = os.path.join(workdir, phase + 'system.xml')
        integrator_filename = os.path.join(workdir, phase + 'integrator.xml')
        write_file(system_filename, openmm.XmlSerializer.serialize(system))
        write_file(integrator_filename, openmm.XmlSerializer.serialize(integrator))
    
        # Clean up
        del simulation

    # Return the modeller instance.
    return [modeller.topology, system, modeller.positions]

# ==============================================================================
# PREPARE YANK CALCULATION
# ==============================================================================

from yank.yank import Yank

# Initialize YANK object.
yank = Yank(store_dir)

# Set options.
options = dict()
options['number_of_iterations'] = niterations
options['number_of_equilibration_iterations'] = nequiliterations
options['online_analysis'] = True
yank.restraint_type = None
options['randomize_ligand'] = False
options['minimize'] = True
options['timestep'] = timestep
options['collision_rate'] = collision_rate

if not is_periodic:
    yank.restraint_type = 'harmonic'

# Prepare phases of calculation.
phase_prefixes = ['solvent', 'complex'] # list of calculation phases (thermodynamic legs) to set up
components = ['ligand', 'receptor', 'solvent'] # components of the binding system
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

    # Use mdtraj to extract the desired subset of the system.
    # TODO: This part isn't working yet.
    mdtraj_top = mdtraj.Topology.from_openmm(fixer.topology)
    #mdtraj_traj = mdtraj.Trajectory(xyz=[fixer.positions], topology=mdtraj_top)
    if phase_prefix == 'solvent':
        dsl = component_dsl['ligand']
    elif phase_prefix == 'complex':
        dsl = component_dsl['complex']
    else:
        raise Exception("unknown phase_prefix '%s'" % phase_prefix)
    subset_atom_indices = mdtraj_top.select(dsl)
    subset_topology = mdtraj_top.subset(subset_atom_indices).to_openmm()
    #subset_positions = mdtraj_traj.atom_slice(atom_indices).openmm_positions(0)
    x = np.array(fixer.positions / unit.angstrom)
    subset_positions = unit.Quantity(x[subset_atom_indices,:], unit.angstrom)

    # Solvate (if needed), create System, and minimize (if requested).
    [solvated_topology, solvated_system, solvated_positions] = solvate_and_minimize(subset_topology, subset_positions, phase=phase + '-')

    # Record components.
    systems[phase] = solvated_system
    positions[phase] = solvated_positions

    # Identify various components.
    mdtraj_top = mdtraj.Topology.from_openmm(solvated_topology)
    atom_indices[phase] = dict()
    for component in components:
        atom_indices[phase][component] = mdtraj_top.select(component_dsl[component]).tolist()

# Create list of phase names.
phases = systems.keys()

# Create reference thermodynamic state.
from yank.repex import ThermodynamicState # TODO: Fix this weird import path to something more sane, like 'from yank.repex import ThermodynamicState'
if is_periodic:
    thermodynamic_state = ThermodynamicState(temperature=temperature, pressure=pressure)
else:
    thermodynamic_state = ThermodynamicState(temperature=temperature)

# Create new simulation.
yank.create(phases, systems, positions, atom_indices, thermodynamic_state, options=options)

