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

from yank.pipeline import *


# =============================================================================
# TESTS
# =============================================================================

def test_compute_min_dist():
    """Test computation of minimum distance between two molecules"""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[3, 3, 3], [3, 4, 5]], np.float)
    mol3_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) == np.sqrt(3)


def test_compute_min_max_dist():
    """Test compute_min_max_dist() function."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]])
    mol2_pos = np.array([[2, 2, 2], [2, 4, 5]])  # determine min dist
    mol3_pos = np.array([[3, 3, 3], [3, 4, 5]])  # determine max dist
    min_dist, max_dist = compute_min_max_dist(mol1_pos, mol2_pos, mol3_pos)
    assert min_dist == np.linalg.norm(mol1_pos[1] - mol2_pos[0])
    assert max_dist == np.linalg.norm(mol1_pos[1] - mol3_pos[1])


# ==============================================================================
# SETUP PIPELINE UTILITY FUNCTIONS
# ==============================================================================

def test_remove_overlap():
    """Test function remove_overlap()."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[1, 1, 1], [3, 4, 5]], np.float)
    mol3_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) < 0.1
    mol1_pos = remove_overlap(mol1_pos, mol2_pos, mol3_pos, min_distance=0.1, sigma=2.0)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) >= 0.1


def test_pull_close():
    """Test function pull_close()."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol3_pos = np.array([[10, 10, 10], [13, 14, 15]], np.float)
    translation2 = pull_close(mol1_pos, mol2_pos, 1.5, 5)
    translation3 = pull_close(mol1_pos, mol3_pos, 1.5, 5)
    assert isinstance(translation2, np.ndarray)
    assert 1.5 <= compute_min_dist(mol1_pos, mol2_pos + translation2) <= 5
    assert 1.5 <= compute_min_dist(mol1_pos, mol3_pos + translation3) <= 5


def test_pack_transformation():
    """Test function pack_transformation()."""
    BOX_SIZE = 5
    CLASH_DIST = 1

    mol1 = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mols = [np.copy(mol1),  # distance = 0
            mol1 + 2 * BOX_SIZE]  # distance > box
    mols_affine = [np.append(mol, np.ones((2, 1)), axis=1) for mol in mols]

    transformations = [pack_transformation(mol1, mol2, CLASH_DIST, BOX_SIZE) for mol2 in mols]
    for mol, transf in zip(mols_affine, transformations):
        assert isinstance(transf, np.ndarray)
        mol2 = mol.dot(transf.T)[:, :3]  # transform and "de-affine"
        min_dist, max_dist = compute_min_max_dist(mol1, mol2)
        assert CLASH_DIST <= min_dist and max_dist <= BOX_SIZE

