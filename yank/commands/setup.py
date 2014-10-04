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

def read_openeye_crd(filename, natoms_expected, verbose=False):
    """
    Read one or more coordinate sets from a file that OpenEye supports.

    ARGUMENTS

    filename (string) - the coordinate filename to be read
    natoms_expected (int) - number of atoms expected

    RETURNS

    positions_list (list of numpy array of simtk.unit.Quantity) - list of coordinate sets read
    """

    if verbose: print "Reading cooordinate sets from '%s'..." % filename

    import openeye.oechem as oe
    imolstream = oe.oemolistream()
    imolstream.open(filename)
    positions_list = list()
    for molecule in imolstream.GetOEGraphMols():
        oecoords = molecule.GetCoords() # oecoords[atom_index] is tuple of atom positions, in angstroms
        natoms = len(oecoords) # number of atoms
        if natoms != natoms_expected:
            raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))
        positions = unit.Quantity(numpy.zeros([natoms,3], numpy.float32), unit.angstroms) # positions[atom_index,dim_index] is positions of dim_index dimension of atom atom_index
        for atom_index in range(natoms):
            positions[atom_index,:] = unit.Quantity(numpy.array(oecoords[atom_index]), unit.angstroms)
        positions_list.append(positions)

    if verbose: print "%d coordinate sets read." % len(positions_list)

    return positions_list

def read_pdb_crd(filename, natoms_expected, verbose=False):
    """
    Read one or more coordinate sets from a PDB file.
    Multiple coordinate sets (in the form of multiple MODELs) can be read.

    ARGUMENTS

    filename (string) - name of the file to be read
    natoms_expected (int) - number of atoms expected

    RETURNS

    positions_list (list of numpy array of simtk.unit.Quantity) - list of coordinate sets read

    """
    import simtk.openmm.app as app
    pdb = app.PDBFile(filename)
    positions_list = pdb.getPositions(asNumpy=True)
    natoms = positions_list.shape[0]
    if natoms != natoms_expected:
        raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))

    # Append if we haven't dumped positions yet.
 #   if (atom_index == natoms_expected):
  #       positions_list.append(copy.deepcopy(positions))

    # Return positions.
    return positions_list

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

def dispatch_binding(args):
    verbose = args['--verbose']

    # Specify simulation parameters.
    nonbondedMethod = getattr(app, args['--nbmethod'])
    implicitSolvent = getattr(app, args['--gbsa'])
    if args['--constraints']==None: args['--constraints'] = None # Necessary because there is no 'None' in simtk.openmm.app
    constraints = getattr(app, args['--constraints'])
    removeCMMotion = False

    # Create systems according to specified method.
    systems = dict()
    positions = dict()
    phases = ['ligand', 'receptor', 'complex']
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
                systems[phase].setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
            # Check to make sure number of atoms match expectation.
            if prmtop_natoms != inpcrd_natoms:
                raise Exception("Atom number mismatch: prmtop %s has %d atoms; inpcrd %s has %d atoms." % (prmtop_filename, prmtop_natoms, inpcrd_filename, inpcrd_natoms))
            # Report number of atoms.
            if verbose: print "  %9d atoms" % prmtop_natoms

    # Initialize YANK object.
    # TODO: Add arguments for ligand positions and atom indices in each system.
    # TODO: Perhaps we can use dicts for 'systems', 'positions', 'alchemical_atom_indices', etc.?
    # TODO: Maybe even break up each alchemical leg into a separate Yank call?
    from yank.yank import Yank # TODO: Fix this weird import path to something more sane, like 'from yank import Yank'?
    yank = Yank(ligand=systems['ligand'],
                receptor=systems['receptor'],
                complex=systems['complex'], complex_positions=positions['complex'],
                output_directory=args['--store'],
                verbose=verbose)

    # Configure YANK object with command-line parameter overrides.
    if args['--iterations']:
        yank.niterations = int(args['--iterations'])
    if args['--verbose']:
        yank.verbose = True
    if args['--online-analysis']:
        yank.online_analysis = True
    if args['--restraints']:
        yank.restraint_type = args['--restraints']
    if args['--randomize-ligand']:
        yank.randomize_ligand = True
    if args['--platform'] != 'None':
        yank.platform = openmm.Platform.getPlatformByName(args['--platform'])

    # Initialize.
    yank._initialize()

    # Report success.
    return True
