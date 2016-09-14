#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Print YANK help.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """
YANK help

Usage: 
  yank help [COMMAND]

Description:
  Get COMMAND's usage usage and arguments

Required Arguments:
  COMMAND                       Name of the command you want more information about

"""

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    try:
        command = args['COMMAND']
    except:
        #Case when no args are passed (like in the nosetests)
        command = None
    # Handle the null case or itself
    if command is None or command == 'help':
        from yank import cli
        print(cli.usage)
        return True
    else:
        from yank import commands
        command_usage = getattr(commands, command).usage
        print(command_usage)
        return True
    # In case something went wrong
    return False
