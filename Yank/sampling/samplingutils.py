#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Sampling Utilities
==================

Sampling Utilities for the YANK Sampling Package. A collection of functions and small classes
which are common to help the samplers and other public hooks interface with the samplers.

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""


# =============================================================================================
# Sampling Exceptions
# =============================================================================================

class SimulationNaNError(Exception):
    """Error when a simulation goes to NaN"""
    pass
