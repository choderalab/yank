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
  yank setup binding amber --ligand_prmtop=PRMTOP --ligand_inpcrd=INPCRD --complex_prmtop=PRMTOP --complex_inpcrd=INPCRD [-v | --verbose] [-i=ITERATIONS | --iterations=ITERATIONS] [-m | --mpi] [--restraints <restraint_type>] [--randomize-ligand]
  yank run [-i=ITERATIONS | --iterations ITERATIONS] [-o | --online-analysis]
  yank status
  yank analyze
  yank cleanup

Commands:
  selftest                      Run selftests.
  setup binding amber           Set up binding free energy calculation using AMBER.
  run                           Run the calculation that has been set up
  status                        Get the current status
  analyze                       Analyze data
  cleanup                       Clean up (delete) run files.

Options:
  -h, --help                    Print command line help
  -c, --cite                    Print relevant citations
  --ligand_prmtop=PRMTOP        AMBER prmtop file for ligand [default: ligand.prmtop]
  --ligand_inpcrd=INPCRD        AMBER inpcrd file for ligand [default: ligand.inpcrd]
  --complex_prmtop=PRMTOP       AMBER prmtop file for complex (ligand must appear first) [default: complex.prmtop]
  --complex_inpcrd=INPCRD       AMBER inpcrd file for complex (ligand must appear first) [default: complex.inpcrd]
  -i NITER, --iterations NITER  Number of iterations to run [default: 1000]
  --randomize-ligand            Randomize initial ligand positions if specified
  -v, --verbose                 Print verbose output
  -o, --online-analysis         Enable on-the-fly analysis

"""

def main():
    # Parse command-line arguments.
    from docopt import docopt
    import version
    args = docopt(usage, version=version.version)

    success = False # Flag set to True if we have correctly formed command-line arguments.
    from . import commands # TODO: This would be clearer if we could do 'from yank import commands', but can't figure out how

    # Handle simple arguments.
    if args['--help']:
        print usage
        success = True
    if args['--cite']:
        success = commands.cite.dispatch(args)

    # Handle commands.
    command_list = ['selftest', 'setup', 'run', 'status', 'analyze', 'cleanup'] # TODO: Build this list automagically by introspection of commands submodule.
    for command in command_list:
        if args[command]:
            success = getattr(commands, command).dispatch(args)

    # If unsuccessful, print usage and exit with an error.
    if not success:
        print usage
        sys.exit(1)
