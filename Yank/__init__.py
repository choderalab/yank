#!/usr/local/bin/env python

"""
YANK

"""

# Define global version.
try:
    from . import version  # Needed for yank 3.X.
except:
    # Fill in information manually.
    class _Version:
        short_version = "dev"
        version = "dev"
        full_version = "dev"
        git_revision = "dev"
        release = False

    version = _Version()

__version__ = version.version

# Self module imports
from . import utils
from . import restraints
from . import pipeline
from . import experiment
from .yank import Topography, AlchemicalPhase
