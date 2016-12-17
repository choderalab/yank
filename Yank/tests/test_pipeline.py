#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test pipeline functions in pipeline.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

from simtk import openmm

from yank import utils
from yank.pipeline import find_components


# =============================================================================
# TESTS
# =============================================================================

def test_method_find_components():
    """Test find_components() function."""
    data_dir = utils.get_data_filename(os.path.join('tests', 'data'))
    ben_tol_dir = os.path.join(data_dir, 'benzene-toluene-explicit')
    ben_tol_complex_path = os.path.join(ben_tol_dir, 'complex.prmtop')
    ben_tol_prmtop = openmm.app.AmberPrmtopFile(ben_tol_complex_path)

    topology = ben_tol_prmtop.topology
    system = ben_tol_prmtop.createSystem(nonbondedMethod=openmm.app.PME)
    n_atoms = system.getNumParticles()

    # Standard selection.
    atom_indices = find_components(system, topology, ligand_dsl='resname BEN')
    assert len(atom_indices['ligand']) == 12
    assert len(atom_indices['receptor']) == 15
    assert len(atom_indices['solvent']) == n_atoms - 12 - 15

    # Select toluene as solvent to test solvent_dsl.
    atom_indices = find_components(system, topology, ligand_dsl='resname BEN',
                                   solvent_dsl='resname TOL')
    assert len(atom_indices['ligand']) == 12
    assert len(atom_indices['solvent']) == 15
    assert len(atom_indices['receptor']) == n_atoms - 12 - 15
