#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Run YANK self tests after installation.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

import doctest
import pkgutil
import sys

from yank import utils, version
from yank.commands import platforms
import simtk.openmm as mm


#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================
usage = """
YANK selftest

Usage:
  yank selftest  [-d | --doctests] [-n | --nosetests] [-v... | --verbosity=#]]

Description:
  Run the YANK selftests to check that functions are behaving as expected and if external licenses are available

General Options:
  -d, --doctests                Run module doctests
  -n, --nosetests               Run the nosetests (slow) on YANK
  -v, --verbosity=n             Optional verbosity level of nosetests OR verbose doctests (default: 1)

"""
#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):

    # Determine verbosity in advance
    # nosetests: -v == --verbosity=2
    # Assuming that verbosity = 1 (or no -v) is no verbosity for doctests
    verbosity = max(args['-v'] + 1, int(args['--verbosity']))
    # Header
    print("YANK Selftest")
    print("-------------")
    # Yank Version
    print("Yank Version %s" % version.version)
    # OpenMM Platforms
    platforms.dispatch()
    # Errors
    platform_errors = mm.Platform.getPlatformLoadErrors()
    if len(platform_errors) > 0: # This check only required to make header
        print{  "************************************************")
        print("\nWarning! There were OpenMM Platform Load Errors!")
        print{  "************************************************")
        for e in platform_errors:
            print(e)
        print{  "************************************************")
        print{  "************************************************")
    # OpenEye checks
    try:
        import openeye
        import openeye.examples.openeye_tests as OETests
        print("OpenEye version $s Found! Checking install..." % openeye.__version__)
        OETests.run_test_suite()
    except:
        print("Valid OpenEye install not found")
        print("Not required, but please check install if you expected it")
    # Run nosetests
    # Note: These will not run during standard nosetests because they must be explicitly called
    # i.e. no infinite nosetest loop
    if args['-n'] or args['--nosetests']:
        # Clear some lines
        print("\n\n\n")
        # Alert User
        print{"******************************************")
        print("Nosetests invoked! This will take a while!")
        print{"******************************************")
        import nose
        try: #Check for timer install
            result = nose.run(argv=['yank', '--nocapture', '--verbosity=%d'%verbosity, '--with-timer', '-a', '!slow'] )
        except:
            result = nose.run(argv=['yank', '--nocapture', '--verbosity=%d'%verbosity, '-a', '!slow'] )

    if args['--doctests'] or args['-d']:
        print("Running doctests for all modules...")

        # Run tests on main module.
        import yank
        if verbosity > 1:
            verbose = True
        else:
            verbose = False
        (failure_count, test_count) = doctest.testmod(yank, verbose=verbose)
        # Run tests on all submodules.
        package = yank
        prefix = package.__name__ + "."
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
            module = __import__(modname, fromlist="dummy")
            (module_failure_count, module_test_count) = doctest.testmod(module, verbose=verbose)
            failure_count += module_failure_count
            test_count += module_test_count

        # Report results.
        if failure_count == 0:
            print("All doctests pass.")
        else:
            print("WARNING: There were %d doctest failures." % failure_count)

    return True
