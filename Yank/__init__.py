#!/usr/local/bin/env python

"""
YANK

"""

# Define global version.
try:
    from . import version  # Needed for yank 3.X.
    __version__ = version.version
except ImportError:
    __version__ = "0.0.0dev"

# Self module imports
from . import utils
from . import restraints
from . import pipeline
from . import experiment
from .yank import Topography, AlchemicalPhase
