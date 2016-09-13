#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
YANK command-line interface (cli)

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

import sys
import docopt

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

fastusage = """
YANK

Usage:
  yank [-h | --help] [-c | --cite] COMMAND [<ARGS>]...

Commands:
  help                          Get specific help for the command given in ARGS
  selftest                      Run selftests
  platforms                     List available OpenMM platforms.
  prepare binding amber         Set up binding free energy calculation using AMBER input files
  prepare binding gromacs       Set up binding free energy calculation using gromacs input files
  run                           Run the calculation that has been set up
  script                        Set up and run free energy calculations from a YAML script.
  status                        Get the current status
  analyze                       Analyze data OR extract trajectory from a NetCDF file in a common format.
  cleanup                       Clean up (delete) run files.

Options:
  -h --help                     Display this message and quit
  -c, --cite                    Print relevant citations

See 'yank help COMMAND' for more information on a specific command.

"""

usage = """
YANK

Usage:
  yank [-h | --help] [-c | --cite]
  yank selftest [-v | --verbose] [-d | --doctests]
  yank platforms
  yank prepare binding amber --setupdir=DIRECTORY --ligand=DSLSTRING (-s=STORE | --store=STORE) [-n=NSTEPS | --nsteps=NSTEPS] [-i=NITER | --iterations=NITER] [--equilibrate=NEQUIL] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--cutoff=CUTOFF] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [--minimize] [-y=FILEPATH | --yaml=FILEPATH] [-v | --verbose]
  yank prepare binding gromacs --setupdir=DIRECTORY --ligand=DSLSTRING (-s=STORE | --store=STORE) [--gromacsinclude=DIRECTORY] [-n=NSTEPS | --nsteps=NSTEPS] [-i=NITER | --iterations=NITER] [--equilibrate=NEQUIL] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--cutoff=CUTOFF] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [--minimize] [-y=FILEPATH | --yaml=FILEPATH] [-v | --verbose]
  yank run (-s=STORE | --store=STORE) [-m | --mpi] [-i=NITER | --iterations=NITER] [--platform=PLATFORM] [--precision=PRECISION] [--phase=PHASE] [-o | --online-analysis] [-v | --verbose]
  yank script (-y=FILEPATH | --yaml=FILEPATH)
  yank status (-s=STORE | --store=STORE) [-v | --verbose]
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
  yank analyze extract-trajectory --netcdf=FILEPATH (--state=STATE | --replica=REPLICA) --trajectory=FILEPATH [--start=START_FRAME] [--skip=SKIP_FRAME] [--end=END_FRAME] [--nosolvent] [--discardequil] [-v | --verbose]
  yank cleanup (-s=STORE | --store=STORE) [-v | --verbose]

Commands:
  selftest                      Run selftests.
  platforms                     List available OpenMM platforms.
  prepare binding amber         Set up binding free energy calculation using AMBER input files
  prepare binding gromacs       Set up binding free energy calculation using gromacs input files
  run                           Run the calculation that has been set up
  script                        Set up and run free energy calculations from a YAML script.
  status                        Get the current status
  analyze                       Analyze data
  extract-trajectory            Extract trajectory from a NetCDF file in a common format.
  cleanup                       Clean up (delete) run files.

General options:
  -h, --help                    Print command line help
  -c, --cite                    Print relevant citations
  -n=NSTEPS, --nsteps=NSTEPS    Number of steps per iteration
  -i=NITER, --iterations=NITER  Number of iterations to run
  --randomize-ligand            Randomize initial ligand positions if specified
  -v, --verbose                 Print verbose output
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.
  -o, --online-analysis         Enable on-the-fly analysis
  --platform=PLATFORM           OpenMM Platform to use (Reference, CPU, OpenCL, CUDA)
  --precision=PRECISION         OpenMM Platform precision model to use (for CUDA or OpenCL only, one of {mixed, double, single})
  --equilibrate=NEQUIL          Number of equilibration iterations
  --minimize                    Minimize configurations before running simulation.
  -m, --mpi                     Use MPI to parallelize the calculation
  -y, --yaml=FILEPATH           Path to the YAML script specifying options and/or how to set up and run the experiment.

Selftest options:
  -d, --doctests                Run module doctests

Simulation options:
  --restraints=TYPE             Restraint type to add between protein and ligand in implicit solvent (harmonic, flat-bottom) [default: flat-bottom]
  --gbsa=GBSA                   OpenMM GBSA model (HCT, OBC1, OBC2, GBn, GBn2)
  --nbmethod=METHOD             OpenMM nonbonded method (NoCutoff, CutoffPeriodic, PME, Ewald)
  --cutoff=CUTOFF               OpenMM nonbonded cutoff (in units of distance) [default: 1*nanometer]
  --constraints=CONSTRAINTS     OpenMM constraints (None, HBonds, AllBonds, HAngles) [default: HBonds]
  --phase=PHASE                 Resume only specified phase of calculation (solvent, complex)
  --temperature=TEMPERATURE     Temperature for simulation (in K, or simtk.unit readable string) [default: 298*kelvin]
  --pressure=PRESSURE           Pressure for simulation (in atm, or simtk.unit readable string) [default: 1*atmospheres]

Amber options:
  --setupdir=DIRECTORY          Setup directory to look for AMBER {receptor|ligand|complex}.{prmtop|inpcrd} files.
  --ligand=DSLSTRING            Specification of the ligand atoms according to MDTraj DSL syntax [default: resname MOL]

Gromacs options:
  --gromacsinclude=DIRECTORY    Include directory for gromacs files [default: /usr/local/gromacs/share/gromacs/top]

Extract-trajectory options:
  --netcdf=FILEPATH             Path to the NetCDF file.
  --state=STATE_IDX             Index of the alchemical state for which to extract the trajectory
  --replica=REPLICA_IDX         Index of the replica for which to extract the trajectory
  --trajectory=FILEPATH         Path to the trajectory file to create (extension determines the format)
  --start=START_FRAME           Index of the first frame to keep
  --end=END_FRAME               Index of the last frame to keep
  --skip=SKIP_FRAME             Extract one frame every SKIP_FRAME
  --nosolvent                   Do not extract solvent
  --discardequil                Detect and discard equilibration frames

"""

# TODO: Add optional arguments that we can use to override sys.argv for testing purposes.
def main(argv=None):
    # Parse command-line arguments.
    from docopt import docopt
    import version
    args = docopt(fastusage, version=version.version, argv=argv, options_first=True)

    dispatched = False # Flag set to True if we have correctly dispatched a command.
    from . import commands # TODO: This would be clearer if we could do 'from yank import commands', but can't figure out how
    import inspect
    command_list = [module[0] for module in inspect.getmembers(commands, inspect.ismodule)] # Build the list of commands based on the <command>.py modules in the ./commands folder
        
    # Handle simple arguments.
    if args['--cite']:
        dispatched = commands.cite.dispatch(args)

    # Handle commands.
    if args['COMMAND'] in command_list:
        # Check that command is valid:
        command = args['COMMAND']
        command_usage = getattr(commands, command).usage
        command_args  = docopt(command_usage, version=version.version, argv=argv) # This will terminate if command is invalid
        # Execute Command
        dispatched = getattr(commands, command).dispatch(command_args)
        
    # If unsuccessful, print usage and exit with an error.
    if not dispatched:
        print usage
        return True

    # Indicate success.
    return False
