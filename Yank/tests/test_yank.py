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

import numpy as np

from simtk import openmm
from openmmtools import testsystems

from nose import tools

from yank import Yank
from yank.repex import ThermodynamicState

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

from simtk import unit
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

def test_parameters():
    """Test Yank parameters initialization."""

    # Check that both Yank and Repex parameters are accepted
    Yank(store_directory='test', restraint_type='harmonic', nsteps_per_iteration=1)

@tools.raises(TypeError)
def test_unknown_parameters():
    """Test whether Yank raises exception on wrong initialization."""
    Yank(store_directory='test', wrong_parameter=False)


def notest_LennardJonesPair(box_width_nsigma=6.0):
    """
    Compute binding free energy of two Lennard-Jones particles and compare to numerical result.

    Parameters
    ----------
    box_width_nsigma : float, optional, default=6.0
        Box width is set to this multiple of Lennard-Jones sigma.

    """

    NSIGMA_MAX = 6.0 # number of standard errors tolerated for success

    # Create Lennard-Jones pair.
    thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    kT = kB * thermodynamic_state.temperature
    sigma = 3.5 * unit.angstroms
    epsilon = 6.0 * kT
    test = testsystems.LennardJonesPair(sigma=sigma, epsilon=epsilon)
    system, positions = test.system, test.positions
    binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    # Create temporary directory for testing.
    import tempfile
    store_dir = tempfile.mkdtemp()

    # Initialize YANK object.
    options = dict()
    options['restraint_type'] = None
    options['number_of_iterations'] = 10
    options['platform'] = openmm.Platform.getPlatformByName("Reference") # use Reference platform for speed
    options['mc_rotation'] = False
    options['mc_displacement'] = True
    options['mc_displacement_sigma'] = 1.0 * unit.nanometer
    options['timestep'] = 2 * unit.femtoseconds
    options['nsteps_per_iteration'] = 50

    # Override receptor mass to keep it stationary.
    #system.setParticleMass(0, 0)

    # Override box vectors.
    box_edge = 6*sigma
    a = unit.Quantity((box_edge, 0 * unit.angstrom, 0 * unit.angstrom))
    b = unit.Quantity((0 * unit.angstrom, box_edge, 0 * unit.angstrom))
    c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # Override positions
    positions[0,:] = box_edge/2
    positions[1,:] = box_edge/4

    phase = 'complex-explicit'

    # Alchemical protocol.
    from yank.alchemy import AlchemicalState
    alchemical_states = list()
    lambda_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    for lambda_value in lambda_values:
        alchemical_state = AlchemicalState()
        alchemical_state['lambda_electrostatics'] = lambda_value
        alchemical_state['lambda_sterics'] = lambda_value
        alchemical_states.append(alchemical_state)
    protocols = dict()
    protocols[phase] = alchemical_states

    # Create phases.
    systems = { phase : system }
    positions = { phase : positions }
    phases = [phase]
    atom_indices = { 'complex-explicit' : { 'ligand' : [1] } }

    # Create new simulation.
    yank = Yank(store_dir, **options)
    yank.create(phases, systems, positions, atom_indices, thermodynamic_state, protocols=protocols)

    # Run the simulation.
    yank.run()

    # Analyze the data.
    results = yank.analyze()
    standard_state_correction = results[phase]['standard_state_correction']
    Delta_f = results[phase]['Delta_f_ij'][0,1] - standard_state_correction
    dDelta_f = results[phase]['dDelta_f_ij'][0,1]
    nsigma = abs(binding_free_energy/kT - Delta_f) / dDelta_f

    # Check results against analytical results.
    # TODO: Incorporate standard state correction
    output = "\n"
    output += "Analytical binding free energy                                  : %10.5f +- %10.5f kT\n" % (binding_free_energy / kT, 0)
    output += "Computed binding free energy (with standard state correction)   : %10.5f +- %10.5f kT (nsigma = %3.1f)\n" % (Delta_f, dDelta_f, nsigma)
    output += "Computed binding free energy (without standard state correction): %10.5f +- %10.5f kT (nsigma = %3.1f)\n" % (Delta_f + standard_state_correction, dDelta_f, nsigma)
    output += "Standard state correction alone                                 : %10.5f           kT\n" % (standard_state_correction)
    print output

    #if (nsigma > NSIGMA_MAX):
    #    output += "\n"
    #    output += "Computed binding free energy differs from true binding free energy.\n"
    #    raise Exception(output)

    return [Delta_f, dDelta_f]

if __name__ == '__main__':
    from yank import utils
    utils.config_root_logger(True, log_file_path='test_LennardJones_pair.log')

    box_width_nsigma_values = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    Delta_f_n = list()
    dDelta_f_n = list()
    for (n, box_width_nsigma) in enumerate(box_width_nsigma_values):
        [Delta_f, dDelta_f] = notest_LennardJonesPair(box_width_nsigma=box_width_nsigma)
        Delta_f_n.append(Delta_f)
        dDelta_f_n.append(dDelta_f)
    Delta_f_n = np.array(Delta_f_n)
    dDelta_f_n = np.array(dDelta_f_n)

    for (box_width_nsigma, Delta_f, dDelta_f) in zip(box_width_nsigma_values, Delta_f_n, dDelta_f_n):
        print "%8.3f %12.6f %12.6f" % (box_width_nsigma, Delta_f, dDelta_f)
