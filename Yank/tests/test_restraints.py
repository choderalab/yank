#!/usr/bin/python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Test restraints module.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import math

import numpy as np
import netCDF4 as netcdf
from simtk import unit
import openmmtools as mmtools
from openmmtools import testsystems, states
from nose.plugins.attrib import attr

import yank.restraints
from yank import yamlbuild, analyze, Topography


# =============================================================================================
# UNIT TESTS
# =============================================================================================

class HostGuestNoninteracting(testsystems.HostGuestVacuum):
    """CB7:B2 host-guest system in vacuum with no nonbonded interactions.

    Parameters
    ----------
    Same as HostGuestVacuum

    Examples
    --------
    Create host:guest system with no nonbonded interactions.
    >>> testsystem = HostGuestVacuumNoninteracting()
    >>> system, positions = testsystem.system, testsystem.positions

    Properties
    ----------
    receptor_atoms : list of int
        Indices of receptor atoms
    ligand_atoms : list of int
        Indices of ligand atoms

    """
    def __init__(self, **kwargs):
        super(HostGuestNoninteracting, self).__init__(**kwargs)

        # Store receptor and ligand atom indices
        self.receptor_atoms = range(0, 126)
        self.ligand_atoms = range(126, 156)

        # Remove nonbonded interactions
        force_indices = {self.system.getForce(index).__class__.__name__: index
                         for index in range(self.system.getNumForces())}
        self.system.removeForce(force_indices['NonbondedForce'])

expected_restraints = {
    'Harmonic': yank.restraints.Harmonic,
    'FlatBottom': yank.restraints.FlatBottom,
    'Boresch': yank.restraints.Boresch,
}

restraint_test_yaml = """
---
options:
  minimize: no
  verbose: yes
  output_dir: %(output_directory)s
  number_of_iterations: %(number_of_iter)s
  nsteps_per_iteration: 100
  temperature: 300*kelvin
  pressure: null
  anisotropic_dispersion_correction: no
  platform: OpenCL

solvents:
  vacuum:
    nonbonded_method: PME
    nonbonded_cutoff: 0.59 * nanometer

systems:
  ship:
    phase1_path: [data/benzene-toluene-standard-state/standard_state_complex.inpcrd, data/benzene-toluene-standard-state/standard_state_complex.prmtop]
    phase2_path: [data/benzene-toluene-standard-state/standard_state_complex.inpcrd, data/benzene-toluene-standard-state/standard_state_complex.prmtop]
    ligand_dsl: resname ene
    solvent: vacuum

protocols:
  absolute-binding:
    complex:
      alchemical_path:
        lambda_restraints:     [0.0, 0.25, 0.5, 0.75, 1.0]
        lambda_electrostatics: [0.0, 0.00, 0.0, 0.00, 0.0]
        lambda_sterics:        [0.0, 0.00, 0.0, 0.00, 0.0]
    solvent:
      alchemical_path:
        lambda_electrostatics: [1.0, 1.0]
        lambda_sterics:        [1.0, 1.0]

experiments:
  system: ship
  protocol: absolute-binding
  restraint:
    type: %(restraint_type)s
"""


def general_restraint_run(options):
    """
    Generalized restraint simulation run to test free energy = standard state correction.

    options : Dict. A dictionary of substitutions for restraint_test_yaml
    """
    with mmtools.utils.temporary_directory() as output_directory:
        # TODO refactor this to use AlchemicalPhase API rather than a YAML script.
        options['output_directory'] = output_directory
        # run both setup and experiment
        yaml_builder = yamlbuild.YamlBuilder(restraint_test_yaml % options)
        yaml_builder.build_experiments()
        # Estimate Free Energies
        ncfile_path = os.path.join(output_directory, 'experiments', 'complex.nc')
        ncfile = netcdf.Dataset(ncfile_path, 'r')
        Deltaf_ij, dDeltaf_ij = analyze.estimate_free_energies(ncfile)
        # Correct the sign for the fact that we are adding vs removing the restraints
        DeltaF_simulated = Deltaf_ij[-1, 0]
        dDeltaF_simulated = dDeltaf_ij[-1, 0]
        DeltaF_restraints = ncfile.groups['metadata'].variables['standard_state_correction'][0]
        ncfile.close()

    # Check if they are close
    assert np.allclose(DeltaF_restraints, DeltaF_simulated, rtol=dDeltaF_simulated)


@attr('slow')  # Skip on Travis-CI
def test_harmonic_free_energy():
    """
    Test that the harmonic restraint simulated free energy equals the standard state correction
    """
    options = {'number_of_iter': '500',
               'restraint_type': 'Harmonic'}
    general_restraint_run(options)


@attr('slow')  # Skip on Travis-CI
def test_flat_bottom_free_energy():
    """
    Test that the harmonic restraint simulated free energy equals the standard state correction
    """
    options = {'number_of_iter': '500',
               'restraint_type': 'FlatBottom'}
    general_restraint_run(options)


@attr('slow')  # Skip on Travis-CI
def test_Boresch_free_energy():
    """
    Test that the harmonic restraint simulated free energy equals the standard state correction
    """
    # These need more samples to converge
    options = {'number_of_iter': '1000',
               'restraint_type': 'Boresch'}
    general_restraint_run(options)


def test_harmonic_standard_state():
    """
    Test that the expected harmonic standard state correction is close to our approximation

    Also ensures that PBC bonds are being computed and disabled correctly as expected
    """
    LJ_fluid = testsystems.LennardJonesFluid()

    # Create Harmonic restraint.
    restraint = yank.restraints.create_restraint('Harmonic', restrained_receptor_atom=1)

    # Determine other parameters.
    ligand_atoms = [3, 4, 5]
    topography = Topography(LJ_fluid.topology, ligand_atoms=ligand_atoms)
    sampler_state = states.SamplerState(positions=LJ_fluid.positions)
    thermodynamic_state = states.ThermodynamicState(system=LJ_fluid.system,
                                                    temperature=300.0 * unit.kelvin)
    restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    spring_constant = restraint.spring_constant

    # Compute standard-state volume for a single molecule in a box of size (1 L) / (avogadros number)
    liter = 1000.0 * unit.centimeters ** 3  # one liter
    box_volume = liter / (unit.AVOGADRO_CONSTANT_NA * unit.mole)  # standard state volume
    analytical_shell_volume = (2 * math.pi / (spring_constant * thermodynamic_state.beta))**(3.0/2)
    analytical_standard_state_G = - math.log(box_volume / analytical_shell_volume)
    restraint_standard_state_G = restraint.get_standard_state_correction(thermodynamic_state)
    np.testing.assert_allclose(analytical_standard_state_G, restraint_standard_state_G)


def test_available_restraint_classes():
    """Test to make sure expected restraint classes are available.
    """
    available_restraint_classes = yank.restraints.available_restraint_classes()
    available_restraint_types = yank.restraints.available_restraint_types()

    # We shouldn't have `None` (from the base class) as an available type
    assert None not in available_restraint_classes
    assert None not in available_restraint_types

    for restraint_type, restraint_class in expected_restraints.items():
        msg = "Failed comparing restraint type '%s' with %s" % (restraint_type, str(available_restraint_classes))
        assert restraint_type in available_restraint_classes, msg
        assert available_restraint_classes[restraint_type] is restraint_class, msg
        assert restraint_type in available_restraint_types, msg


def test_restraint_dispatch():
    """Test dispatch of various restraint types."""
    for restraint_type, restraint_class in expected_restraints.items():
        # Create a test system
        t = HostGuestNoninteracting()

        # Create states and topography encoding the info to determine the parameters.
        topography = Topography(t.topology, ligand_atoms='resname B2')
        sampler_state = states.SamplerState(positions=t.positions)
        thermodynamic_state = states.ThermodynamicState(system=t.system, temperature=300.0*unit.kelvin)

        # Add restraints and determine parameters.
        restraint = yank.restraints.create_restraint(restraint_type)
        restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)

        # Check that we got the right restraint class.
        assert restraint.__class__.__name__ == restraint_type
        assert restraint.__class__ is restraint_class

# =============================================================================================
# MAIN
# =============================================================================================

if __name__ == '__main__':
    test_restraint_dispatch()
