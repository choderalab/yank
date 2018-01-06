#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Sampling
========

Multistate Sampling simulation algorithms and specific variants.

This module provides a general facility for running multiple thermodynamic state sampling simulations, both general
as well as derived classes for special cases such as parallel tempering (in which
the states differ only in temperature).

Provided classes include:

- :class:`MultiStateSampler`
    Base class for general, multi-thermodynamic state parallel sampling
- :class:`ReplicaExchange`
    Derived class from MultiStateSampler which allows sampled thermodynamic states
    to swap based on Hamiltonian Replica Exchange
- :class:`ParallelTempering`
    Convenience subclass of ReplicaExchange for parallel tempering simulations
    (one System object, many temperatures).
- :class:`Reporter`
    Replica Exchange reporter class to store all variables and data

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

from .multistatesampler import MultiStateSampler
from .multistatereporter import MultiStateReporter
from .replicaexchangesampler import ReplicaExchangeSampler
from .paralleltemperingsampler import ParallelTemperingSampler
from .samplingutils import *
