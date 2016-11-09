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

import yaml
from simtk import unit
from simtk.openmm import app

from alchemy import AbsoluteAlchemicalFactory

from .. import utils
from .. import pipeline
from ..yank import AlchemicalPhase, Yank 
from ..repex import ThermodynamicState 
from ..yamlbuild import YamlBuilder
from ..pipeline import find_components

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """
YANK prepare

Usage:
  yank prepare binding amber --setupdir=DIRECTORY --ligand=DSLSTRING (-s=STORE | --store=STORE) [-n=NSTEPS | --nsteps=NSTEPS] [-i=NITER | --iterations=NITER] [--equilibrate=NEQUIL] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--cutoff=CUTOFF] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [--minimize] [--platform=PLATFORM] [--precision=PRECISION] [-o | --online-analysis] [-y=FILEPATH | --yaml=FILEPATH] [-v | --verbose]
  yank prepare binding gromacs --setupdir=DIRECTORY --ligand=DSLSTRING (-s=STORE | --store=STORE) [--gromacsinclude=DIRECTORY] [-n=NSTEPS | --nsteps=NSTEPS] [-i=NITER | --iterations=NITER] [--equilibrate=NEQUIL] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--cutoff=CUTOFF] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [-o | --online-analysis] [--minimize] [--platform=PLATFORM] [--precision=PRECISION] [-y=FILEPATH | --yaml=FILEPATH] [-v | --verbose]

Description:
  prepare binding amber         Set up binding free energy calculation using AMBER input files
  prepare binding gromacs       Set up binding free energy calculation using gromacs input files

Amber Required Arguments:
  --setupdir=DIRECTORY          Setup directory to look for AMBER {receptor|ligand|complex}.{prmtop|inpcrd} files.
  --ligand=DSLSTRING            Specification of the ligand atoms according to MDTraj DSL syntax [default: resname MOL]

Gromacs Required Arguments:
  --gromacsinclude=DIRECTORY    Include directory for gromacs files [default: /usr/local/gromacs/share/gromacs/top]

General Options:
  --equilibrate=NEQUIL          Number of equilibration iterations
  -i=NITER, --iterations=NITER  Number of iterations to run
  --minimize                    Minimize configurations before running simulation.
  -n=NSTEPS, --nsteps=NSTEPS    Number of steps per iteration
  -o, --online-analysis         Enable on-the-fly analysis
  --randomize-ligand            Randomize initial ligand positions if specified
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.
  -v, --verbose                 Print verbose output
  -y, --yaml=FILEPATH           Path to the YAML script specifying options and/or how to set up and run the experiment.

Simulation options:
  --constraints=CONSTRAINTS     OpenMM constraints (None, HBonds, AllBonds, HAngles) [default: HBonds]
  --cutoff=CUTOFF               OpenMM nonbonded cutoff (in units of distance) [default: 1*nanometer]
  --gbsa=GBSA                   OpenMM GBSA model (HCT, OBC1, OBC2, GBn, GBn2)
  --nbmethod=METHOD             OpenMM nonbonded method (NoCutoff, CutoffPeriodic, PME, Ewald)
  --platform=PLATFORM           OpenMM Platform to use (Reference, CPU, OpenCL, CUDA)
  --precision=PRECISION         OpenMM Platform precision model to use (for CUDA or OpenCL only, one of {mixed, double, single})
  --pressure=PRESSURE           Pressure for simulation (in atm, or simtk.unit readable string) [default: 1*atmospheres]
  --restraints=TYPE             Restraint type to add between protein and ligand in implicit solvent (Harmonic, FlatBottom)
  --temperature=TEMPERATURE     Temperature for simulation (in K, or simtk.unit readable string) [default: 298*kelvin]



"""

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
    alchemical_phases : list of AlchemicalPhase
       Phases (thermodynamic legs) of the calculation.

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

    # Determine if this will be an explicit or implicit solvent simulation
    if ('nonbondedMethod' in system_parameters and
                system_parameters['nonbondedMethod'] != app.NoCutoff):
        phases_names = ['complex-explicit', 'solvent-explicit']
        protocols = [AbsoluteAlchemicalFactory.defaultComplexProtocolExplicit(),
                     AbsoluteAlchemicalFactory.defaultSolventProtocolExplicit()]
    else:
        phases_names = ['complex-implicit', 'solvent-implicit']
        protocols = [AbsoluteAlchemicalFactory.defaultComplexProtocolImplicit(),
                 AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()]

    # Prepare Yank arguments
    alchemical_phases = [None, None]
    setup_directory = os.path.join(setup_directory, '')  # add final slash character
    system_files_paths = [[setup_directory + 'complex.inpcrd', setup_directory + 'complex.prmtop'],
                          [setup_directory + 'solvent.inpcrd', setup_directory + 'solvent.prmtop']]
    for i, phase_name in enumerate(phases_names):
        positions_file_path = system_files_paths[i][0]
        topology_file_path = system_files_paths[i][1]

        logger.info("Reading phase {}".format(phase_name))
        alchemical_phases[i] = pipeline.prepare_phase(positions_file_path, topology_file_path, args['--ligand'],
                                                      system_parameters, verbose=verbose)
        alchemical_phases[i].name = phase_name
        alchemical_phases[i].protocol = protocols[i]

    return alchemical_phases


def setup_binding_gromacs(args):
    """
    Set up ligand binding free energy calculation using gromacs prmtop/inpcrd files.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.

    Returns
    -------
    alchemical_phases : list of AlchemicalPhase
       Phases (thermodynamic legs) of the calculation.

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
    topologies = dict() # topologies[phase] is the Topology object associated with phase 'phase'
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
        topologies[phase] = top.topology
        positions[phase] = gro.getPositions(asNumpy=True)
        # Check to make sure number of atoms match between prmtop and inpcrd.
        prmtop_natoms = systems[phase].getNumParticles()
        inpcrd_natoms = positions[phase].shape[0]
        if prmtop_natoms != inpcrd_natoms:
            raise Exception("Atom number mismatch: prmtop %s has %d atoms; inpcrd %s has %d atoms." % (prmtop_filename, prmtop_natoms, inpcrd_filename, inpcrd_natoms))

        # Find ligand atoms and receptor atoms.
        ligand_dsl = args['--ligand'] # MDTraj DSL that specifies ligand atoms
        atom_indices[phase] = find_components(systems[phase], top.topology, ligand_dsl)

    phases = systems.keys()

    alchemical_phases = [None, None]
    protocols = {'complex-explicit': AbsoluteAlchemicalFactory.defaultComplexProtocolExplicit(),
                 'solvent-explicit': AbsoluteAlchemicalFactory.defaultSolventProtocolImplicit()}
    for i, name in enumerate(phases):
        alchemical_phases[i] = AlchemicalPhase(name, systems[name], topologies[name],
                                               positions[name], atom_indices[name],
                                               protocols[name])

    return alchemical_phases

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
        alchemical_phases = setup_binding_amber(args)
    elif args['gromacs']:
        alchemical_phases = setup_binding_gromacs(args)
    else:
        logger.error("No valid binding free energy calculation setup command specified: Must be one of ['amber', 'systembuilder'].")
        # Trigger help argument to be returned.
        return False

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
    if args['--restraints']:
        restraint_type = args['--restraints']
    yank.create(thermodynamic_state, *alchemical_phases, restraint_type)

    # Dump analysis object
    analysis = [[alchemical_phases[0].name, 1], [alchemical_phases[1].name, -1]]
    analysis_script_path = os.path.join(store_dir, 'analysis.yaml')
    with open(analysis_script_path, 'w') as f:
        yaml.dump(analysis, f)

    # Report success.
    return True
