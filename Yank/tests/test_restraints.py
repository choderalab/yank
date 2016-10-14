#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test restraints module.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import tempfile

from nose.plugins.attrib import attr

import yank.restraints

from simtk import unit, openmm
from openmmtools import testsystems

#=============================================================================================
# UNIT TESTS
#=============================================================================================

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

        # Remove nonnbonded interactions
        force_indices = { self.system.getForce(index).__class__.__name__ : index for index in range(self.system.getNumForces()) }
        self.system.removeForce(force_indices['NonbondedForce'])

expected_restraints = {
    'Harmonic' : yank.restraints.Harmonic,
    'FlatBottom' : yank.restraints.FlatBottom,
}

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
        from openmmtools.testsystems import ThermodynamicState
        thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
        # Add restraints
        restraint = yank.restraints.createRestraints(restraint_type, thermodynamic_state, t.system, t.positions, t.receptor_atoms, t.ligand_atoms)
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
  verbose: yes
  output_dir: .
  number_of_iterations: 2
  restraint_type: %(restraint_type)s
  temperature: 300*kelvin
  softcore_beta: 0.0

molecules:
  T4lysozyme:
    filepath: %(receptor_filepath)s
  p-xylene:
    filepath: %(ligand_filepath)s
    antechamber:
      charge_method: bcc

solvents:
  OBC2:
    nonbonded_method: NoCutoff
    implicit_solvent: OBC2

systems:
  lys-pxyl:
    receptor: T4lysozyme
    ligand: p-xylene
    solvent: OBC2
    leap:
      parameters: [leaprc.ff14SB, leaprc.gaff]

protocols:
  absolute-binding:
    complex:
      alchemical_path:
        lambda_electrostatics: [1.0, 0.0, 0.0]
        lambda_sterics:        [1.0, 1.0, 0.0]
    solvent:
      alchemical_path:
        lambda_electrostatics: [1.0, 0.0, 0.0]
        lambda_sterics:        [1.0, 1.0, 0.0]

experiments:
  system: lys-pxyl
  protocol: absolute-binding
"""
    # Test all possible restraint types.
    for restraint_type in expected_restraints:
        data = {
            'restraint_type' : restraint_type,
            'receptor_filepath' : get_data_filename('tests/data/p-xylene-implicit/input/181L-pdbfixer.pdb'),
            'ligand_filepath'   : get_data_filename('tests/data/p-xylene-implicit/input/p-xylene.mol2'),
        }
        yaml_builder = YamlBuilder(yaml_script % data)
        yaml_builder.build_experiment()  # run both setup and experiments

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == '__main__':
    test_available_restraint_classes()
    test_restraint_dispatch()
