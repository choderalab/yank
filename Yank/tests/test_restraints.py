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

import tempfile
import shutil
import math
import numpy as np

from nose.plugins.attrib import attr

import yank.restraints
from openmmtools.testsystems import ThermodynamicState

from simtk import unit, openmm
from openmmtools import testsystems

# =============================================================================================
# UNIT TESTS
# =============================================================================================

from openmmtools.testsystems import HostGuestVacuum


class HostGuestNoninteracting(HostGuestVacuum):
    """CB7:B2 host-guest system in vacuum with no nonbonded interactions.

    Parameters
    ----------
    Same as HostGuestVacuum

    Examples
    --------
    Create host:guest system with no nonbonded interactions.
    >>> testsystem = HostGuestVacuumNoninteracting()
    >>> (system, positions) = testsystem.system, testsystem.positions

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
        self.receptor_atoms = range(0,126)
        self.ligand_atoms = range(126,156)

        # Remove nonbonded interactions
        force_indices = { self.system.getForce(index).__class__.__name__ : index for index in range(self.system.getNumForces()) }
        self.system.removeForce(force_indices['NonbondedForce'])

expected_restraints = {
    'Harmonic' : yank.restraints.Harmonic,
    'FlatBottom' : yank.restraints.FlatBottom,
    'Boresch' : yank.restraints.Boresch,
}


def test_harmonic_standard_state():
    """
    Test that the expected harmonic standard state correction is close to our approximation

    Also ensures that PBC bonds are being computed and disabled correctly as expected
    """
    LJ_fluid = testsystems.LennardJonesFluid()
    receptor_atoms = [0, 1, 2]
    ligand_atoms = [3, 4, 5]
    thermodynamic_state = ThermodynamicState(temperature=300.0 * unit.kelvin)
    restraint = yank.restraints.create_restraints('Harmonic', LJ_fluid.topology, thermodynamic_state, LJ_fluid.system,
                                                  LJ_fluid.positions, receptor_atoms, ligand_atoms)
    spring_constant = restraint._determine_bond_parameters()[0]
    # Compute standard-state volume for a single molecule in a box of size (1 L) / (avogadros number)
    liter = 1000.0 * unit.centimeters ** 3  # one liter
    box_volume = liter / (unit.AVOGADRO_CONSTANT_NA * unit.mole)  # standard state volume
    analytical_shell_volume = (2 * math.pi / (spring_constant * restraint.beta))**(3.0/2)
    analytical_standard_state_G = - math.log(box_volume / analytical_shell_volume)
    restraint_standard_state_G = restraint.get_standard_state_correction()
    np.testing.assert_allclose(analytical_standard_state_G, restraint_standard_state_G)


def test_available_restraint_classes():
    """Test to make sure expected restraint classes are available.
    """
    available_restraint_classes = yank.restraints.available_restraint_classes()
    available_restraint_types = yank.restraints.available_restraint_types()

    # We shouldn't have `None` (from the base class) as an available type
    assert(None not in available_restraint_classes)
    assert(None not in available_restraint_types)

    for (restraint_type, restraint_class) in expected_restraints.items():
        msg = "Failed comparing restraint type '%s' with %s" % (restraint_type, str(available_restraint_classes))
        assert(restraint_type in available_restraint_classes), msg
        assert(available_restraint_classes[restraint_type] is restraint_class), msg
        assert(restraint_type in available_restraint_types), msg


def test_restraint_dispatch():
    """Test dispatch of various restraint types.
    """
    for (restraint_type, restraint_class) in expected_restraints.items():
        # Create a test system
        t = HostGuestNoninteracting()
        # Create a thermodynamic state encoding temperature
        thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
        # Add restraints
        restraint = yank.restraints.create_restraints(restraint_type, t.topology, thermodynamic_state, t.system, t.positions, t.receptor_atoms, t.ligand_atoms)
        # Check that we got the right restraint class
        assert(restraint.__class__.__name__ == restraint_type)
        assert(restraint.__class__ == restraint_class)


def test_protein_ligand_restraints():
    """Test the restraints in a protein:ligand system.
    """
    from yank.yamlbuild import YamlBuilder
    from yank.utils import get_data_filename

    yaml_script = """
---
options:
  minimize: no
  verbose: no
  output_dir: %(output_directory)s
  number_of_iterations: 2
  nsteps_per_iteration: 10
  temperature: 300*kelvin

molecules:
  T4lysozyme:
    filepath: %(receptor_filepath)s
  p-xylene:
    filepath: %(ligand_filepath)s
    antechamber:
      charge_method: bcc

solvents:
  vacuum:
    nonbonded_method: NoCutoff

systems:
  lys-pxyl:
    receptor: T4lysozyme
    ligand: p-xylene
    solvent: vacuum
    leap:
      parameters: [oldff/leaprc.ff14SB, leaprc.gaff]

protocols:
  absolute-binding:
    complex:
      alchemical_path:
        lambda_restraints:     [0.0, 0.5, 1.0]
        lambda_electrostatics: [1.0, 1.0, 1.0]
        lambda_sterics:        [1.0, 1.0, 1.0]
    solvent:
      alchemical_path:
        lambda_electrostatics: [1.0, 1.0, 1.0]
        lambda_sterics:        [1.0, 1.0, 1.0]

experiments:
  system: lys-pxyl
  protocol: absolute-binding
  restraint:
    type: %(restraint_type)s
"""
    # Test all possible restraint types.
    available_restraint_types = yank.restraints.available_restraint_types()
    for restraint_type in available_restraint_types:
        print('***********************************')
        print('Testing %s restraints...' % restraint_type)
        print('***********************************')
        output_directory = tempfile.mkdtemp()
        data = {
            'output_directory' : output_directory,
            'restraint_type' : restraint_type,
            'receptor_filepath' : get_data_filename('tests/data/p-xylene-implicit/181L-pdbfixer.pdb'),
            'ligand_filepath'   : get_data_filename('tests/data/p-xylene-implicit/p-xylene.mol2'),
        }
        # run both setup and experiment
        yaml_builder = YamlBuilder(yaml_script % data)
        yaml_builder.build_experiments()
        # Clean up
        shutil.rmtree(output_directory)

# =============================================================================================
# MAIN
# =============================================================================================

if __name__ == '__main__':
    test_available_restraint_classes()
    test_restraint_dispatch()
