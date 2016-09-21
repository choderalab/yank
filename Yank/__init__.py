#!/usr/local/bin/env python

"""
YANK

"""

# Define global version.
from . import version #Needed for yank 3.X. TODO: should be from yank import version for clarity.
__version__ = version.version

# Self module imports
# TODO: Figure out how to not require this syntax
from . import utils
from . import repex
from . import sampling
from . import restraints
from . import pipeline
from . import yamlbuild
from . import analyze
from .yank import Yank
