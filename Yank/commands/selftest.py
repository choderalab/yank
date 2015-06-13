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

from yank import utils

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    # TODO: Figure out how to run nosetests instead.

    print "Running doctests for all modules..."
    verbose = args['--verbose']
    # Run tests on main module.
    import yank
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
        print "All doctests pass."
        return True
    else:
        print "WARNING: There were %d doctest failures." % failure_count
        sys.exit(1)



