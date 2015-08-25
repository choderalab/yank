#!/usr/local/bin/env python

"""
Test YANK using simple models.

DESCRIPTION

This test suite generates a number of simple models to test the 'Yank' facility.

COPYRIGHT

Written by John D. Chodera <jchodera@gmail.com> while at the University of California Berkeley.

LICENSE

This code is licensed under the latest available version of the GNU General Public License.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import sys
import math
import copy
import time
import datetime
from functools import partial

import numpy

from openmmtools import testsystems

from yank import Yank

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

def test_LennardJonesPair():
    """
    Compute binding free energy of two Lennard-Jones particles and compare to numerical result.

    """

    # Create Lennard-Jones pair.

    test = testsystems.LennardJonesPair()
    system, positions = test.system, test.positions
    thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    # Create temporary directory for testing.
    import tempfile
    store_dir = tempfile.mkdtemp()

    # Initialize YANK object.
    options = dict()
    options['restraint_type'] = 'flat-bottom'
    options['temperature'] = temperature
    options['number_of_iterations'] = 100
    options['platform'] = openmm.Platform.getPlatformByName("Reference") # use Reference platform for speed

    # Create phases.
    phase = 'complex_explicit'
    systems = { phase : test.system }
    positions = { phase : test.positions }
    phases = systems.keys()
    atom_indices = [0]

    # Create new simulation.
    yank = Yank(store_dir)
    yank.create(phases, systems, positions, atom_indices, thermodynamic_state, options=options)

    # Run the simulation.
    yank.run()

    # Analyze the data.
    results = yank.analyze()

    # TODO: Check results against analytical results.

