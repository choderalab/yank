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

usage = """
YANK

Usage:
  yank [-h | --help] [-c | --cite]
  yank selftest [-v | --verbose]
  yank platforms
  yank prepare binding amber --setupdir=DIRECTORY --ligand=DSLSTRING (-s=STORE | --store=STORE) [-n=NSTEPS | --nsteps=NSTEPS] [-i=NITER | --iterations=NITER] [--equilibrate=NEQUIL] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--cutoff=CUTOFF] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [--minimize] [-y=FILEPATH | --yaml=FILEPATH] [-v | --verbose]
  yank prepare binding gromacs --setupdir=DIRECTORY --ligand=DSLSTRING (-s=STORE | --store=STORE) [--gromacsinclude=DIRECTORY] [-n=NSTEPS | --nsteps=NSTEPS] [-i=NITER | --iterations=NITER] [--equilibrate=NEQUIL] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--cutoff=CUTOFF] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [--minimize] [-y=FILEPATH | --yaml=FILEPATH] [-v | --verbose]
  yank run (-s=STORE | --store=STORE) [-m | --mpi] [-i=NITER | --iterations=NITER] [--platform=PLATFORM] [--precision=PRECISION] [--phase=PHASE] [-o | --online-analysis] [-v | --verbose]
  yank script (-y=FILEPATH | --yaml=FILEPATH)
  yank status (-s=STORE | --store=STORE) [-v | --verbose]
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
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

Gromacs options
  --gromacsinclude=DIRECTORY    Include directory for gromacs files [default: /usr/local/gromacs/share/gromacs/top]

"""

# TODO: Add optional arguments that we can use to override sys.argv for testing purposes.
def main(argv=None):
    # Parse command-line arguments.
    from docopt import docopt
    import version
    args = docopt(usage, version=version.version, argv=argv)

    dispatched = False # Flag set to True if we have correctly dispatched a command.
    from . import commands # TODO: This would be clearer if we could do 'from yank import commands', but can't figure out how

    # Handle simple arguments.
    if args['--help']:
        print usage
        dispatched = True
    if args['--cite']:
        dispatched = commands.cite.dispatch(args)

    # Handle commands.
    command_list = ['selftest', 'platforms', 'prepare', 'run', 'script', 'status', 'analyze', 'cleanup'] # TODO: Build this list automagically by introspection of commands submodule.
    for command in command_list:
        if args[command]:
            dispatched = getattr(commands, command).dispatch(args)

    # If unsuccessful, print usage and exit with an error.
    if not dispatched:
        print usage
        return True

    # Indicate success.
    return False
