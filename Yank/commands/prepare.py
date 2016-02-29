#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Set up YANK calculations.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os, os.path
import logging
logger = logging.getLogger(__name__)

from simtk import unit
from simtk.openmm import app

from yank import utils
from yank import pipeline
from yank.yank import Yank # TODO: Fix this weird import path to something more sane, like 'from yank import Yank'
from yank.repex import ThermodynamicState # TODO: Fix this weird import path to something more sane, like 'from yank.repex import ThermodynamicState'
from yank.yamlbuild import YamlBuilder
from yank.pipeline import find_components

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

def process_unit_bearing_arg(args, argname, compatible_units):
    """
    Process a unit-bearing command-line argument and handle eventual errors.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.
    argname : str
       Key to use to extract value from args.
    compatible_units : simtk.unit.Unit
       The result will be checked for compatibility with specified units, and an exception raised if not compatible.

    Returns
    -------
    quantity : simtk.unit.Quantity
       The specified parameter, returned as a Quantity.

    See also
    --------
    yank.utils.process_unit_bearing_str : function used for the actual conversion.

    """
    try:
        return utils.process_unit_bearing_str(args[argname], compatible_units)
    except (TypeError, ValueError) as e:
        logger.error('Error while processing argument %s: %s' % (argname, str(e)))
        raise e

def setup_binding_amber(args):
    """
    Set up ligand binding free energy calculation using AMBER prmtop/inpcrd files.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.

    Returns
    -------
    phases : list of str
       Phases (thermodynamic legs) of the calculation.
    systems : dict
       systems[phase] is the OpenMM System reference object for phase 'phase'.
    positions : dict
       positions[phase] is a set of positions (or list of positions) for initializing replicas.
    atom_indices : dict
       atom_indices[phase][component] is list of atom indices for component 'component' in phase 'phase'.

    """
    verbose = args['--verbose']
    setup_directory = args['--setupdir']  # Directory where prmtop/inpcrd files are to be found
    system_parameters = {}  # parameters to pass to prmtop.createSystem

    # Implicit solvent
    if args['--gbsa']:
        system_parameters['implicitSolvent'] = getattr(app, args['--gbsa'])

    # Select nonbonded treatment
    if args['--nbmethod']:
        system_parameters['nonbondedMethod'] = getattr(app, args['--nbmethod'])

    # Constraints
    if args['--constraints']:
        system_parameters['constraints'] = getattr(app, args['--constraints'])

    # Cutoff
    if args['--cutoff']:
        system_parameters['nonbondedCutoff'] = process_unit_bearing_arg(args, '--cutoff', unit.nanometers)

    # Prepare phases of calculation.
    return pipeline.prepare_amber(setup_directory, args['--ligand'], system_parameters, verbose=verbose)

def setup_binding_gromacs(args):
    """
    Set up ligand binding free energy calculation using gromacs prmtop/inpcrd files.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.

    Returns
    -------
    phases : list of str
       Phases (thermodynamic legs) of the calculation.
    systems : dict
       systems[phase] is the OpenMM System reference object for phase 'phase'.
    positions : dict
       positions[phase] is a set of positions (or list of positions) for initializing replicas.
    atom_indices : dict
       atom_indices[phase][component] is list of atom indices for component 'component' in phase 'phase'.

    """
    verbose = args['--verbose']

    # Implicit solvent
    if args['--gbsa']:
        implicitSolvent = getattr(app, args['--gbsa'])
    else:
        implicitSolvent = None

    # Select nonbonded treatment
    # TODO: Carefully check whether input file is periodic or not.
    if args['--nbmethod']:
        nonbondedMethod = getattr(app, args['--nbmethod'])
    else:
        nonbondedMethod = None

    # Constraints
    if args['--constraints']:
        constraints = getattr(app, args['--constraints'])
    else:
        constraints = None

    # Cutoff
    if args['--cutoff']:
        nonbondedCutoff = process_unit_bearing_arg(args, '--cutoff', unit.nanometers)
    else:
        nonbondedCutoff = None

    # COM removal
    removeCMMotion = False

    # Prepare phases of calculation.
    phase_prefixes = ['solvent', 'complex'] # list of calculation phases (thermodynamic legs) to set up
    components = ['ligand', 'receptor', 'solvent'] # components of the binding system
    systems = dict() # systems[phase] is the System object associated with phase 'phase'
    positions = dict() # positions[phase] is a list of coordinates associated with phase 'phase'
    atom_indices = dict() # ligand_atoms[phase] is a list of ligand atom indices associated with phase 'phase'
    setup_directory = args['--setupdir'] # Directory where prmtop/inpcrd files are to be found
    for phase_prefix in phase_prefixes:
        if verbose: logger.info("reading phase %s: " % phase_prefix)
        # Read gromacs input files.
        gro_filename = os.path.join(setup_directory, '%s.gro' % phase_prefix)
        top_filename = os.path.join(setup_directory, '%s.top' % phase_prefix)
        if verbose: logger.info('reading gromacs .gro file: %s' % gro_filename)
        gro = app.GromacsGroFile(gro_filename)
        if verbose: logger.info('reading gromacs .top file "%s" using gromacs include directory "%s"' % (top_filename, args['--gromacsinclude']))
        top = app.GromacsTopFile(top_filename, unitCellDimensions=gro.getUnitCellDimensions(), includeDir=args['--gromacsinclude'])
        # Assume explicit solvent.
        # TODO: Modify this if we can have implicit solvent.
        is_periodic = True
        phase_suffix = 'explicit'
        # Adjust nonbondedMethod.
        # TODO: Ensure that selected method is appropriate.
        if nonbondedMethod == None:
            if is_periodic:
                nonbondedMethod = app.CutoffPeriodic
            else:
                nonbondedMethod = app.NoCutoff
        # TODO: Check to make sure both prmtop and inpcrd agree on explicit/implicit.
        phase = '%s-%s' % (phase_prefix, phase_suffix)
        systems[phase] = top.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints, removeCMMotion=removeCMMotion)
        positions[phase] = gro.getPositions(asNumpy=True)
        # Check to make sure number of atoms match between prmtop and inpcrd.
        prmtop_natoms = systems[phase].getNumParticles()
        inpcrd_natoms = positions[phase].shape[0]
        if prmtop_natoms != inpcrd_natoms:
            raise Exception("Atom number mismatch: prmtop %s has %d atoms; inpcrd %s has %d atoms." % (prmtop_filename, prmtop_natoms, inpcrd_filename, inpcrd_natoms))

        # Find ligand atoms and receptor atoms.
        ligand_dsl = args['--ligand'] # MDTraj DSL that specifies ligand atoms
        atom_indices[phase] = find_components(top.topology, ligand_dsl)

    phases = systems.keys()

    return [phases, systems, positions, atom_indices]

def setup_systembuilder(args):
    """
    Set up ligand binding free energy calculation using OpenMM app and openmoltools.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.

    Returns
    -------
    phases : list of str
       Phases (thermodynamic legs) of the calculation.
    systems : dict
       systems[phase] is the OpenMM System reference object for phase 'phase'.
    positions : dict
       positions[phase] is a set of positions (or list of positions) for initializing replicas.
    atom_indices : dict
       atom_indices[phase][component] is list of atom indices for component 'component' in phase 'phase'.

    """
    # TODO: Merge in systembuilder setup from PBG and JDC.
    raise Exception("Not implemented.")

def dispatch_binding(args):
    """
    Set up a binding free energy calculation.

    Parameters
    ----------
    args : dict
       Command-line arguments from docopt.

    """

    verbose = args['--verbose']
    store_dir = args['--store']
    utils.config_root_logger(verbose, log_file_path=os.path.join(store_dir, 'prepare.log'))

    #
    # Determine simulation options.
    #

    # Specify thermodynamic parameters.
    temperature = process_unit_bearing_arg(args, '--temperature', unit.kelvin)
    pressure = process_unit_bearing_arg(args, '--pressure', unit.atmospheres)
    thermodynamic_state = ThermodynamicState(temperature=temperature, pressure=pressure)

    # Create systems according to specified setup/import method.
    if args['amber']:
        [phases, systems, positions, atom_indices] = setup_binding_amber(args)
    elif args['gromacs']:
        [phases, systems, positions, atom_indices] = setup_binding_gromacs(args)
    else:
        logger.error("No valid binding free energy calculation setup command specified: Must be one of ['amber', 'systembuilder'].")
        # Trigger help argument to be returned.
        return False

    # Report some useful properties.
    if verbose:
        if 'complex-explicit' in atom_indices:
            phase = 'complex-explicit'
        else:
            phase = 'complex-implicit'
        logger.info("TOTAL ATOMS      : %9d" % len(atom_indices[phase]['complex']))
        logger.info("receptor         : %9d" % len(atom_indices[phase]['receptor']))
        logger.info("ligand           : %9d" % len(atom_indices[phase]['ligand']))
        if phase == 'complex-explicit':
            logger.info("solvent and ions : %9d" % len(atom_indices[phase]['solvent']))

    # Set options.
    options = dict()
    if args['--nsteps']:
        options['nsteps_per_iteration'] = int(args['--nsteps'])
    if args['--iterations']:
        options['number_of_iterations'] = int(args['--iterations'])
    if args['--equilibrate']:
        options['number_of_equilibration_iterations'] = int(args['--equilibrate'])
    if args['--online-analysis']:
        options['online_analysis'] = True
    if args['--restraints']:
        options['restraint_type'] = args['--restraints']
    if args['--randomize-ligand']:
        options['randomize_ligand'] = True
    if args['--minimize']:
        options['minimize'] = True

    # Allow platform to be optionally specified in order for alchemical tests to be carried out.
    if args['--platform'] not in [None, 'None']:
        options['platform'] = openmm.Platform.getPlatformByName(args['--platform'])
    if args['--precision']:
        # We need to modify the Platform object.
        if args['--platform'] is None:
            raise Exception("The --platform argument must be specified in order to specify platform precision.")

        # Set platform precision.
        precision = args['--precision']
        platform_name = args['--platform']
        logger.info("Setting %s platform to use precision model '%s'." % platform_name, precision)
        if precision is not None:
            if platform_name == 'CUDA':
                options['platform'].setPropertyDefaultValue('CudaPrecision', precision)
            elif platform_name == 'OpenCL':
                options['platform'].setPropertyDefaultValue('OpenCLPrecision', precision)
            elif platform_name == 'CPU':
                if precision != 'mixed':
                    raise Exception("CPU platform does not support precision model '%s'; only 'mixed' is supported." % precision)
            elif platform_name == 'Reference':
                if precision != 'double':
                    raise Exception("Reference platform does not support precision model '%s'; only 'double' is supported." % precision)
            else:
                raise Exception("Platform selection logic is outdated and needs to be updated to add platform '%s'." % platform_name)

    # Parse YAML options, CLI options have priority
    if args['--yaml']:
        options.update(YamlBuilder(args['--yaml']).yank_options)

    # Create new simulation.
    yank = Yank(store_dir, **options)
    yank.create(phases, systems, positions, atom_indices, thermodynamic_state)

    # Report success.
    return True
