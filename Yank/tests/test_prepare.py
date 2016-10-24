#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test 'yank prepare'.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import tempfile

from nose.plugins.attrib import attr

from docopt import docopt
from yank.commands.prepare import usage
from yank import version, utils

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def notest_prepare_amber_implicit(verbose=False):
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("../examples/benzene-toluene-implicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    command = 'yank prepare binding amber --setupdir=%(examples_path)s --store=%(store_directory)s --iterations=1 --restraints=Harmonic --gbsa=OBC2 --temperature=300*kelvin' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    argv.append('--ligand=resname TOL') # if included in the command string it is split in two
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import prepare
    prepare.dispatch(args)


@attr('slow')  # Skip on Travis-CI
def notest_prepare_amber_explicit(verbose=False):
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("../examples/benzene-toluene-explicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    command = 'yank prepare binding amber --setupdir=%(examples_path)s --store=%(store_directory)s --iterations=1 --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    argv.append('--ligand=resname TOL') # if included in the command string it is split in two
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import prepare
    prepare.dispatch(args)


@attr('slow')  # Skip on Travis-CI
def notest_prepare_gromacs_explicit(verbose=False):
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("../examples/p-xylene-gromacs-example/setup/")
    command = 'yank prepare binding gromacs --setupdir=%(examples_path)s --store=%(store_directory)s --iterations=1 --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres --cutoff=1*nanometer' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    argv.append("--ligand=resname 'p-xylene'") # if included in the command string it is split in two
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import prepare
    prepare.dispatch(args)

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == '__main__':
    #test_prepare_amber_implicit(verbose=True)
    #test_prepare_amber_explicit(verbose=True)
    notest_prepare_gromacs_explicit(verbose=True)
