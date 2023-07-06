#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
YANK command-line interface (cli)

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

from . import version  # This file is created upon install
from docopt import docopt
from . import commands
import inspect


# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK

Usage:
  yank [-h | --help] [-c | --cite] COMMAND [<ARGS>]...

Commands:
  help                          Get specific help for the command given in ARGS
  selftest                      Run selftests
  platforms                     List available OpenMM platforms.
  script                        Set up and run free energy calculations from a YAML script.
  status                        Get the current status
  analyze                       Analyze data OR extract trajectory from a NetCDF file in a common format.
  cleanup                       Clean up (delete) run files.

Options:
  -h --help                     Display this message and quit
  -c, --cite                    Print relevant citations

See 'yank help COMMAND' for more information on a specific command.

"""


# TODO: Add optional arguments that we can use to override sys.argv for testing purposes.
def main(argv=None):
    # Parse command-line arguments.
    # Parse the initial options, the options_first flag must be True to for <ARGS> to act as a wildcard for options
    # (- and --) as well
    args = docopt(usage, version=version.version, argv=argv, options_first=True)

    dispatched = False  # Flag set to True if we have correctly dispatched a command.
    # Build the list of commands based on the <command>.py modules in the ./commands folder
    command_list = [module[0] for module in inspect.getmembers(commands, inspect.ismodule)]

    # Handle simple arguments.
    if args['--cite']:
        dispatched = commands.cite.dispatch(args)

    # Handle commands.
    if args['COMMAND'] in command_list:
        # Check that command is valid:
        command = args['COMMAND']
        command_usage = getattr(commands, command).usage
        # This will terminate if command is invalid
        command_args = docopt(command_usage, version=version.version, argv=argv)
        # Execute Command
        dispatched = getattr(commands, command).dispatch(command_args)

    # If unsuccessful, print usage and exit with an error.
    if not dispatched:
        print(usage)
        return True

    # Indicate success.
    return False
