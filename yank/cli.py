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

usage = """\
YANK

Usage:
  yank [-h | --help] [-c | --cite]
  yank selftest [-v | --verbose]
  yank setup binding amber --ligand_prmtop=PRMTOP --ligand_inpcrd=INPCRD --receptor_prmtop=PRMTOP --receptor_inpcrd=INPCRD --complex_prmtop=PRMTOP --complex_inpcrd=INPCRD (-s=STORE | --store=STORE) [-v | --verbose] [-i=ITERATIONS | --iterations=ITERATIONS] [-m | --mpi] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--platform=PLATFORM] [-v | --verbose]
  yank run (-s=STORE | --store=STORE) [-i=ITERATIONS | --iterations ITERATIONS] [--platform=PLATFORM] [-o | --online-analysis] [-v | --verbose]
  yank status (-s=STORE | --store=STORE) [-v | --verbose]
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
  yank cleanup (-s=STORE | --store=STORE) [-v | --verbose]

Commands:
  selftest                      Run selftests.
  setup binding amber           Set up binding free energy calculation using AMBER.
  run                           Run the calculation that has been set up
  status                        Get the current status
  analyze                       Analyze data
  cleanup                       Clean up (delete) run files.

General options:
  -h, --help                    Print command line help
  -c, --cite                    Print relevant citations
  -i NITER, --iterations NITER  Number of iterations to run [default: 1000]
  --randomize-ligand            Randomize initial ligand positions if specified
  -v, --verbose                 Print verbose output
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.
  -o, --online-analysis         Enable on-the-fly analysis
  --platform=PLATFORM           OpenMM Platform to use (Reference, CPU, OpenCL, CUDA) [default: None]

Simulation options:
  --gbsa=GBSA                   OpenMM GBSA model (HCT, OBC1, OBC2, GBn, GBn2) [default: OBC2]
  --nbmethod=METHOD             OpenMM nonbonded method (NoCutoff, CutoffPeriodic, PME, Ewald) [default: NoCutoff]
  --constraints=CONSTRAINTS     OpenMM constraints (None, HBonds, AllBonds, HAngles) [default: HBonds]

Amber options:
  --ligand_prmtop=PRMTOP        AMBER prmtop file for ligand [default: ligand.prmtop]
  --ligand_inpcrd=INPCRD        AMBER inpcrd file for ligand [default: ligand.inpcrd]
  --complex_prmtop=PRMTOP       AMBER prmtop file for complex (ligand must appear first) [default: complex.prmtop]
  --complex_inpcrd=INPCRD       AMBER inpcrd file for complex (ligand must appear first) [default: complex.inpcrd]


"""

def main():
    # Parse command-line arguments.
    from docopt import docopt
    import version
    args = docopt(usage, version=version.version)

    dispatched = False # Flag set to True if we have correctly dispatched a command.
    from . import commands # TODO: This would be clearer if we could do 'from yank import commands', but can't figure out how

    # Handle simple arguments.
    if args['--help']:
        print usage
        dispatched = True
    if args['--cite']:
        dispatched = True
        success = commands.cite.dispatch(args)

    # Handle commands.
    command_list = ['selftest', 'setup', 'run', 'status', 'analyze', 'cleanup'] # TODO: Build this list automagically by introspection of commands submodule.
    for command in command_list:
        if args[command]:
            dispatched = True
            success = getattr(commands, command).dispatch(args)

    # If unsuccessful, print usage and exit with an error.
    if not dispatched:
        print usage
        sys.exit(1)
