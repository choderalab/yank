#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Analyzers
=========

Analysis tools and module for MultiStateSampler simulations. Provides programmatic and automatic
"best practices" integration to determine free energy and other observables.

Fully extensible to support new samplers and observables.


"""

# The seemingly double import is to get docs to build correctly
# by building the __all__ for the __init__
# https://stackoverflow.com/questions/30856279
from . import multistateanalyzer as msa
from . import analyzerutils as au
from .multistateanalyzer import *
from .analyzerutils import *

__all__ = [x for x in au.__all__ + msa.__all__]
