#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Set up YANK calculations.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

from simtk import openmm
from simtk import unit
from simtk.openmm import app

from yank.yank import Yank # TODO: Fix this weird import path to something more sane, like 'from yank import Yank'?
from yank.oldrepex import ThermodynamicState

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    if args['binding']:
        return dispatch_binding(args)
    else:
        return False

#=============================================================================================
# SET UP BINDING FREE ENERGY CALCULATION
#=============================================================================================

# Common solvent and ions.
_SOLVENT_TYPES = frozenset(['118', '119', '1AL', '1CU', '2FK', '2HP', '2OF', '3CO', '3MT',
        '3NI', '3OF', '4MO', '543', '6MO', 'ACT', 'AG', 'AL', 'ALF', 'ATH',
        'AU', 'AU3', 'AUC', 'AZI', 'Ag', 'BA', 'BAR', 'BCT', 'BEF', 'BF4',
        'BO4', 'BR', 'BS3', 'BSY', 'Be', 'CA', 'CA+2', 'Ca+2', 'CAC', 'CAD',
        'CAL', 'CD', 'CD1', 'CD3', 'CD5', 'CE', 'CES', 'CHT', 'CL', 'CL-',
        'CLA', 'Cl-', 'CO', 'CO3', 'CO5', 'CON', 'CR', 'CS', 'CSB', 'CU',
        'CU1', 'CU3', 'CUA', 'CUZ', 'CYN', 'Cl-', 'Cr', 'DME', 'DMI', 'DSC',
        'DTI', 'DY', 'E4N', 'EDR', 'EMC', 'ER3', 'EU', 'EU3', 'F', 'FE', 'FE2',
        'FPO', 'GA', 'GD3', 'GEP', 'HAI', 'HG', 'HGC', 'HOH', 'IN', 'IOD',
        'ION', 'IR', 'IR3', 'IRI', 'IUM', 'K', 'K+', 'KO4', 'LA', 'LCO', 'LCP',
        'LI', 'LIT', 'LU', 'MAC', 'MG', 'MH2', 'MH3', 'MLI', 'MMC', 'MN',
        'MN3', 'MN5', 'MN6', 'MO1', 'MO2', 'MO3', 'MO4', 'MO5', 'MO6', 'MOO',
        'MOS', 'MOW', 'MW1', 'MW2', 'MW3', 'NA', 'NA+2', 'NA2', 'NA5', 'NA6',
        'NAO', 'NAW', 'Na+2', 'NET', 'NH4', 'NI', 'NI1', 'NI2', 'NI3', 'NO2',
        'NO3', 'NRU', 'Na+', 'O4M', 'OAA', 'OC1', 'OC2', 'OC3', 'OC4', 'OC5',
        'OC6', 'OC7', 'OC8', 'OCL', 'OCM', 'OCN', 'OCO', 'OF1', 'OF2', 'OF3',
        'OH', 'OS', 'OS4', 'OXL', 'PB', 'PBM', 'PD', 'PER', 'PI', 'PO3', 'PO4',
        'POT', 'PR', 'PT', 'PT4', 'PTN', 'RB', 'RH3', 'RHD', 'RU', 'RUB', 'Ra',
        'SB', 'SCN', 'SE4', 'SEK', 'SM', 'SMO', 'SO3', 'SO4', 'SOD', 'SR',
        'Sm', 'Sn', 'T1A', 'TB', 'TBA', 'TCN', 'TEA', 'THE', 'TL', 'TMA',
        'TRA', 'UNX', 'V', 'V2+', 'VN3', 'VO4', 'W', 'WO5', 'Y1', 'YB', 'YB2',
        'YH', 'YT3', 'ZN', 'ZN2', 'ZN3', 'ZNA', 'ZNO', 'ZO3'])

def find_components(topology, ligand_resnames=['MOL'], solvent_resnames=_SOLVENT_TYPES):
    """
    Return list of receptor and ligand atoms.
    Ligand atoms are specified by a residue name, while receptor atoms are everything else excluding water.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
       The topology object specifying the system.
    ligand_resname : list of str, optional, default=['MOL']
       List of three-letter ligand residue names used to identify the ligand(s).
    solvent_resnames : list of str, optional, default=['WAT', 'TIP', 'HOH']
       List of solvent residue names to exclude from receptor definition.

    Returns
    -------
    atom_indices : dict of list of int
       atom_indices[component] is a list of atom indices belonging to that component, where
       component is one of 'receptor', 'ligand', 'solvent', 'complex'

    """
    components = ['receptor', 'ligand', 'solvent', 'complex']
    atom_indices = { component : list() for component in components }

    for atom in topology.atoms():
        if atom.residue.name in ligand_resnames:
            atom_indices['ligand'].append(atom.index)
            atom_indices['complex'].append(atom.index)
        elif atom.residue.name in solvent_resnames:
            atom_indices['solvent'].append(atom.index)
        else:
            atom_indices['receptor'].append(atom.index)
            atom_indices['complex'].append(atom.index)

    return atom_indices

def process_unit_bearing_argument(args, argname, compatible_units):
    """
    """

    # Import all units.
    from simtk.unit import *
    # WARNING: This is dangerous!
    # See: http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    # TODO: Can we use a safer form of (or alternative to) 'eval' here?
    quantity = eval(args[argname])
    # Check to make sure units are compatible with expected units.
    if not quantity.unit.is_compatible(compatible_units):
        raise Exception("Argument %s must be compatible with units %s" % (agname, str(compatible_units)))
    # Return unit-bearing quantity.
    return quantity

def dispatch_binding(args):
    """
    Set up a binding free energy calculation.

    """

    verbose = args['--verbose']

    # Specify simulation parameters.
    nonbondedMethod = getattr(app, args['--nbmethod'])
    implicitSolvent = getattr(app, args['--gbsa'])
    if args['--constraints']==None: args['--constraints'] = None # Necessary because there is no 'None' in simtk.openmm.app
    constraints = getattr(app, args['--constraints'])
    removeCMMotion = False

    # Specify thermodynamic parameters.
    temperature = process_unit_bearing_argument(args, '--temperature', unit.kelvin)
    pressure = process_unit_bearing_argument(args, '--pressure', unit.atmospheres)
    thermodynamic_state = ThermodynamicState(temperature=temperature, pressure=pressure)

    # Create systems according to specified method.
    phases = ['ligand', 'complex'] # list of calculation phases (thermodynamic legs) to set up
    components = ['ligand', 'receptor', 'solvent'] # components of the binding system
    systems = dict() # systems[phase] is the System object associated with phase 'phase'
    positions = dict() # positions[phase] is a list of coordinates associated with phase 'phase'
    atom_indices = { phase : dict() for phase in phases } # ligand_atoms[phase] is a list of ligand atom indices associated with phase 'phase'
    is_periodic = False # True if calculations are in a periodic box
    if args['amber']:
        for phase in phases:
            if verbose: print "%s: " % phase
            # Read Amber prmtop and create System object.
            prmtop_filename = args['--%s_prmtop' % phase]
            prmtop = app.AmberPrmtopFile(prmtop_filename)
            systems[phase] = prmtop.createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)
            prmtop_natoms = systems[phase].getNumParticles()
            # Read Amber inpcrd and load positions.
            inpcrd_filename = args['--%s_inpcrd' % phase]
            inpcrd = app.AmberInpcrdFile(inpcrd_filename)
            positions[phase] = inpcrd.getPositions(asNumpy=True)
            inpcrd_natoms = positions[phase].shape[0]
            # Update box vectors (if needed).
            if inpcrd.boxVectors is not None:
                is_periodic = True
                systems[phase].setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
            # Check to make sure number of atoms match between prmtop and inpcrd.
            if prmtop_natoms != inpcrd_natoms:
                raise Exception("Atom number mismatch: prmtop %s has %d atoms; inpcrd %s has %d atoms." % (prmtop_filename, prmtop_natoms, inpcrd_filename, inpcrd_natoms))
            # Find ligand atoms and receptor atoms.
            ligand_resname = args['--ligname']
            atom_indices[phase] = find_components(prmtop.topology, [ligand_resname])
            # Report some useful properties.
            if verbose:
                print "  TOTAL ATOMS      : %9d" % prmtop_natoms
                print "  receptor         : %9d" % len(atom_indices[phase]['receptor'])
                print "  ligand           : %9d" % len(atom_indices[phase]['ligand'])
                print "  solvent and ions : %9d" % len(atom_indices[phase]['solvent'])

    elif args['systembuilder']:
        # TODO: This part is under construction

        # Create SystemBuilder objects
        ligand = Mol2SystemBuilder(args['--ligand'], 'ligand')
        receptor = BiomoleculePDBSystemBuilder(args['--receptor'], 'receptor')
        complex = ComplexSystemBuilder(ligand, receptor, 'complex')

        # Create phases.
        systems['solvent'] = ligand.system
        positions['solvent'] = [ligand.coordinates_as_quantity]
        atom_indices['solvent']['ligand'] = ligand.ligand_atoms
        atom_indices['solvent']['receptor'] = list()

        systems['complex'] = complex.system
        positions['complex'] = [complex.coordinates_as_quantity]
        atom_indices['complex']['ligand'] = complex.ligand_atoms
        atom_indices['complex']['receptor'] = complex.receptor_atoms

        # TODO: Set is_periodic flag appropriately

    # Initialize YANK object.
    # TODO: Maybe break up each alchemical leg into a separate Yank call?
    yank = Yank(args['--store'])

    # Set options.
    options = dict()
    if args['--iterations']:
        options['niterations'] = int(args['--iterations'])
    if args['--online-analysis']:
        options['online_analysis'] = True
    if args['--restraints']:
        options['restraint_type'] = args['--restraints']
    if args['--randomize-ligand']:
        options['randomize_ligand'] = True
    if args['--platform'] != 'None':
        options['platform'] = openmm.Platform.getPlatformByName(args['--platform'])

    # Create new simulation.
    yank.create(phases, systems, positions, atom_indices, thermodynamic_state, verbose=verbose, options=options)

    # Report success.
    return True
