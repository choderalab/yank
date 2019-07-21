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


# ==============================================================================
# TRAILBLAZE
# ==============================================================================

def test_trailblaze_checkpoint():
    """Test that trailblaze algorithm can resume if interrupted."""
    from openmmtools.states import GlobalParameterState, SamplerState, ThermodynamicState, CompoundThermodynamicState

    par_name = 'testsystems_HarmonicOscillator_x0'

    # Create composable state that control offset of harmonic oscillator.
    class X0State(GlobalParameterState):
        testsystems_HarmonicOscillator_x0 = GlobalParameterState.GlobalParameter(par_name, 1.0)

    # Create a harmonic oscillator thermo state.
    oscillator = mmtools.testsystems.HarmonicOscillator(K=1.0*unit.kilocalories_per_mole/unit.nanometer**2)
    sampler_state = SamplerState(positions=oscillator.positions)
    thermo_state = ThermodynamicState(oscillator.system, temperature=300*unit.kelvin)
    x0_state = X0State(testsystems_HarmonicOscillator_x0=0.0)
    compound_state = CompoundThermodynamicState(thermo_state, composable_states=[x0_state])

    # Run trailblaze to find path of x0 from 0 to 1 nm.
    platform = platform=openmm.Platform.getPlatformByName('CPU')
    mcmc_move = mmtools.mcmc.LangevinDynamicsMove(
        timestep=1.0*unit.femtosecond, n_steps=1,
        context_cache=mmtools.cache.ContextCache(platform=platform)
    )

    with mmtools.utils.temporary_directory() as checkpoint_dir_path:
        checkpoint_dir_path = os.path.join(checkpoint_dir_path, 'temp.yaml')

        # Running with a checkpoint path creates checkpoint files.
        first_protocol = trailblaze_alchemical_protocol(
            compound_state, sampler_state, mcmc_move,
            checkpoint_path=checkpoint_dir_path,
            state_parameters=[(par_name, [0.0, 1.0])]
        )

        # The path on the checkpoint files is correct.
        with open(checkpoint_dir_path, 'r') as f:
            checkpoint_protocol = yaml.load(f, Loader=yaml.FullLoader)
        assert checkpoint_protocol == first_protocol

        # Running a second time (with different final state) should
        # start from the previous alchemical protocol.
        second_protocol = trailblaze_alchemical_protocol(
            compound_state, sampler_state, mcmc_move,
            checkpoint_path=checkpoint_dir_path,
            state_parameters=[(par_name, [0.0, 2.0])]
        )
        len_first_protocol = len(first_protocol[par_name])
        assert second_protocol[par_name][:len_first_protocol] == first_protocol[par_name]

        # The path on the checkpoint is correct.
        with open(checkpoint_dir_path, 'r') as f:
            checkpoint_protocol = yaml.load(f, Loader=yaml.FullLoader)
        assert checkpoint_protocol == second_protocol
