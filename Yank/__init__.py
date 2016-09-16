#!/usr/local/bin/env python

"""
YANK

"""

# Define global version.
from . import version #Needed for yank 3.X. TODO: should be from yank import version for clarity.
__version__ = version.version

# Import modules.
#import alchemy
#import repex
#import sampling
#import analyze
#import restraints

from yank import Yank



