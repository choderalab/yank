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

def find_components(topology, ligand_resnames=['MOL'], solvent_resnames=['WAT', 'TIP', 'HOH']):
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
    atom_indices = { list() for component in components }

    for atom in prmtop.topology:
        if atom.residue.name in ligand_resname:
            atom_indices['ligand'].append(atom.index)
            atom_indices['complex'].append(atom.index)
        elif atom.residue_name in solvent_resnames:
            atom_indices['solvent'].append(atom.index)
        else:
            atom_indices['receptor'].append(atom.index)
            atom_indices['complex'].append(atom.index)

    return atom_indices

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

    # Create systems according to specified method.
    phases = ['ligand', 'complex'] # list of calculation phases (thermodynamic legs) to set up
    components = ['ligand', 'receptor', 'solvent'] # components of the binding system
    systems = dict() # systems[phase] is the System object associated with phase 'phase'
    positions = dict() # positions[phase] is a list of coordinates associated with phase 'phase'
    atom_indices = dict() # ligand_atoms[phase] is a list of ligand atom indices associated with phase 'phase'
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
            ligand_resname = args['--resname']
            atom_indices[phase] = find_components(prmtop.topology, [ligand_resname])
            # Report some useful properties.
            if verbose:
                print "  total atoms: %9d" % prmtop_natoms
                print "  receptor   : %9d" % len(atom_indices[phase]['receptor'])
                print "  ligand     : %9d" % len(atom_indices[phase]['ligand'])

    # Process request to randomize ligand positions.
    # TODO: Handle this separately for AMBER prmtop/inpcrd loader and SystemBuilder schemes.
    if args['--randomize-ligand']:
        if is_periodic:
            raise Exception("Cannot randomize the ligand after explicit solvent coordinates have been loaded.")
        if verbose:
            print "Randomizing ligand positions and excluding overlapping configurations..."
        randomized_positions = list()
        sigma = 2*complex_restraints.getReceptorRadiusOfGyration()
        close_cutoff = 1.5 * units.angstrom # TODO: Allow this to be specified by user.
        nstates = len(systems)
        for state_index in range(nstates):
            positions = self.complex_positions[numpy.random.randint(0, len(self.complex_positions))]
            from sampling import ModifiedHamiltonianExchange # TODO: Modify to 'from yank.sampling import ModifiedHamiltonianExchange'?
            new_positions = ModifiedHamiltonianExchange.randomize_ligand_position(positions, self.receptor_atoms, self.ligand_atoms, sigma, close_cutoff)
            randomized_positions.append(new_positions)
        self.complex_positions = randomized_positions

    # Initialize YANK object.
    # TODO: Maybe break up each alchemical leg into a separate Yank call?
    from yank.yank import Yank # TODO: Fix this weird import path to something more sane, like 'from yank import Yank'?
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
    yank.create(phases, systems, positions, alchemical_atoms, thermodynamic_state, verbose=verbose, options=options)

    # Report success.
    return True
