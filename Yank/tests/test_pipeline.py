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
from yank.pipeline import _redistribute_trailblaze_states

from nose.tools import assert_raises_regexp


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

class TestThermodynamicTrailblazing:
    """Test suite for the thermodynamic trailblazing function."""

    PAR_NAME_X0 = 'testsystems_HarmonicOscillator_x0'
    PAR_NAME_K = 'testsystems_HarmonicOscillator_K'

    @classmethod
    def get_harmonic_oscillator(cls):
        """Create a harmonic oscillator thermodynamic state to test the trailblaze algorithm."""
        from openmmtools.states import (GlobalParameterState, ThermodynamicState,
                                        CompoundThermodynamicState, SamplerState)

        # Create composable state that control offset of harmonic oscillator.
        class X0State(GlobalParameterState):
            testsystems_HarmonicOscillator_x0 = GlobalParameterState.GlobalParameter(
                cls.PAR_NAME_X0, 1.0)
            testsystems_HarmonicOscillator_K = GlobalParameterState.GlobalParameter(
                cls.PAR_NAME_K, (1.0*unit.kilocalories_per_mole/unit.nanometer**2).value_in_unit_system(unit.md_unit_system))

        # Create a harmonic oscillator thermo state.
        k = 1.0*unit.kilocalories_per_mole/unit.nanometer**2
        oscillator = mmtools.testsystems.HarmonicOscillator(K=k)
        sampler_state = SamplerState(positions=oscillator.positions)
        thermo_state = ThermodynamicState(oscillator.system, temperature=300*unit.kelvin)
        x0_state = X0State(
            testsystems_HarmonicOscillator_x0=0.0,
            testsystems_HarmonicOscillator_K=k.value_in_unit_system(unit.md_unit_system)
        )
        compound_state = CompoundThermodynamicState(thermo_state, composable_states=[x0_state])

        return compound_state, sampler_state

    @staticmethod
    def get_langevin_dynamics_move():
        """Build a cheap Langevin dynamics move to test the trailblaze algorithm."""
        platform = openmm.Platform.getPlatformByName('CPU')
        mcmc_move = mmtools.mcmc.LangevinDynamicsMove(
            timestep=1.0*unit.femtosecond, n_steps=1,
            context_cache=mmtools.cache.ContextCache(platform=platform)
        )
        return mcmc_move

    def test_trailblaze_checkpoint(self):
        """Test that trailblaze algorithm can resume if interrupted."""

        def _check_checkpoint_files(checkpoint_dir_path, expected_protocol, n_atoms):

            checkpoint_protocol_path = os.path.join(checkpoint_dir_path, 'protocol.yaml')
            checkpoint_positions_path = os.path.join(checkpoint_dir_path, 'coordinates.dcd')

            # The protocol on the checkpoint file is correct.
            with open(checkpoint_protocol_path, 'r') as f:
                checkpoint_protocol = yaml.load(f, Loader=yaml.FullLoader)
            assert checkpoint_protocol == expected_protocol

            # The positions and box vectors have the correct dimension.
            expected_n_states = len(expected_protocol[self.PAR_NAME_X0])
            trajectory_file = mdtraj.formats.DCDTrajectoryFile(checkpoint_positions_path, 'r')
            xyz, cell_lengths, cell_angles = trajectory_file.read()
            assert (xyz.shape[0], xyz.shape[1]) == (expected_n_states, n_atoms)
            assert cell_lengths.shape[0] == expected_n_states

        # Create a harmonic oscillator system to test.
        compound_state, sampler_state = self.get_harmonic_oscillator()
        n_atoms = len(sampler_state.positions)
        mcmc_move = self.get_langevin_dynamics_move()

        # Run trailblaze to find path of x0 from 0 to 1 nm.
        with mmtools.utils.temporary_directory() as checkpoint_dir_path:

            # Running with a checkpoint path creates checkpoint files.
            first_protocol = run_thermodynamic_trailblazing(
                compound_state, sampler_state, mcmc_move,
                checkpoint_dir_path=checkpoint_dir_path,
                state_parameters=[(self.PAR_NAME_X0, [0.0, 1.0])]
            )

            # The info in the checkpoint files is correct.
            _check_checkpoint_files(checkpoint_dir_path, first_protocol, n_atoms)

            # Running a second time (with different final state) should
            # start from the previous alchemical protocol.
            second_protocol = run_thermodynamic_trailblazing(
                compound_state, sampler_state, mcmc_move,
                checkpoint_dir_path=checkpoint_dir_path,
                state_parameters=[(self.PAR_NAME_X0, [0.0, 2.0])],
                bidirectional_redistribution=False
            )
            len_first_protocol = len(first_protocol[self.PAR_NAME_X0])
            assert second_protocol[self.PAR_NAME_X0][:len_first_protocol] == first_protocol[self.PAR_NAME_X0]

            # The info in the checkpoint files is correct.
            _check_checkpoint_files(checkpoint_dir_path, second_protocol, n_atoms)

    def test_trailblaze_setters(self):
        """Test that the special setters and the global parameter function shortcut are working properly."""
        # Create a harmonic oscillator system to test.
        compound_state, sampler_state = self.get_harmonic_oscillator()
        mcmc_move = self.get_langevin_dynamics_move()

        # Assign to the parameter a function and run the
        # trailblaze algorithm over the function variable.
        global_parameter_functions = {self.PAR_NAME_X0: 'lambda**2'}
        function_variables = ['lambda']

        # Make sure it's not possible to have a parameter defined as a function and as a parameter state as well.
        err_msg = f"Cannot specify {self.PAR_NAME_X0} in 'state_parameters' and 'global_parameter_functions'"
        with assert_raises_regexp(ValueError, err_msg):
            run_thermodynamic_trailblazing(
                compound_state, sampler_state, mcmc_move,
                state_parameters=[(self.PAR_NAME_X0, [0.0, 1.0])],
                global_parameter_functions=global_parameter_functions,
                function_variables=function_variables,
            )

        # Trailblaze returns the protocol for the actual parameters, not for the function variables.
        protocol = run_thermodynamic_trailblazing(
            compound_state, sampler_state, mcmc_move,
            state_parameters=[('lambda', [0.0, 1.0])],
            global_parameter_functions=global_parameter_functions,
            function_variables=function_variables,
        )
        assert list(protocol.keys()) == [self.PAR_NAME_X0]
        parameter_protocol = protocol[self.PAR_NAME_X0]
        assert parameter_protocol[0] == 0
        assert parameter_protocol[-1] == 1

    def test_reversed_direction(self):
        """Make sure that a trailblaze run in the opposite direction still returns the parameters in the forward order."""
        # Create a harmonic oscillator system to test.
        compound_state, sampler_state = self.get_harmonic_oscillator()
        mcmc_move = self.get_langevin_dynamics_move()

        # For this, we run through two variables to make sure they are executed in the correct order.
        k_start = getattr(compound_state, self.PAR_NAME_K)
        k_end = k_start * 2
        protocol = run_thermodynamic_trailblazing(
            compound_state, sampler_state, mcmc_move,
            state_parameters=[
                (self.PAR_NAME_X0, [1.0, 0.0]),
                (self.PAR_NAME_K, [k_start, k_end]),
            ],
            reversed_direction=True,
            bidirectional_redistribution=False
        )
        assert protocol[self.PAR_NAME_X0] == [1.0, 0.0, 0.0], protocol[self.PAR_NAME_X0]
        assert protocol[self.PAR_NAME_K] == [k_start, k_start, k_end], protocol[self.PAR_NAME_K]

    def test_redistribute_trailblaze_states(self):
        """States are redistributed correctly as a function of the bidirectional thermo length."""
        # This simulates the protocol found after trailblazing.
        optimal_protocol = {
            'lambda_electrostatics': [1.0, 0.5, 0.0, 0.0, 0.0],
            'lambda_sterics':        [1.0, 1.0, 1.0, 0.5, 0.0]
        }

        # Each test case is a tuple (states_stds, expected_redistributed_protocol, expected_states_map).
        test_cases = [
            (
                [[0.5, 0.5, 0.5, 0.5],
                 [1.5, 1.5, 1.5]],
                {'lambda_electrostatics': [1.0, 0.75, 0.5, 0.25, 0.0,  0.0, 0.0, 0.0],
                 'lambda_sterics':        [1.0,  1.0, 1.0,  1.0, 1.0, 0.75, 0.5, 0.0]},
                [0, 0, 1, 1, 2, 2, 3, 4]
            ),
            (
                [[0.5, 0.5, 0.5, 0.5],
                 [0.0, 0.0, 0.0]],
                {'lambda_electrostatics': [1.0, 0.0,  0.0, 0.0],
                 'lambda_sterics':        [1.0, 1.0, 0.25, 0.0]},
                [0, 2, 3, 4]
            ),
        ]

        for states_stds, expected_redistributed_protocol, expected_states_map in test_cases:
            redistributed_protocol, states_map = _redistribute_trailblaze_states(
                optimal_protocol, states_stds, thermodynamic_distance=0.5)
            assert expected_redistributed_protocol == redistributed_protocol, \
                f'{expected_redistributed_protocol} != {redistributed_protocol}'
            assert expected_states_map == states_map, f'{expected_states_map} != {states_map}'

    def test_read_trailblaze_checkpoint_coordinates(self):
        """read_trailblaze_checkpoint_coordinates() returns the correct number of SamplerStates."""
        # Create a harmonic oscillator system to test.
        compound_state, sampler_state = self.get_harmonic_oscillator()
        mcmc_move = self.get_langevin_dynamics_move()

        # The end states differ only in the spring constant
        k_initial = getattr(compound_state, self.PAR_NAME_K)
        k_final = 250 * k_initial

        # Helper function.
        def _call_run_thermodynamic_trailblazing(
            thermodynamic_distance, bidirectional_redistribution,
            bidirectional_search_thermo_dist, checkpoint_dir_path
        ):
            return run_thermodynamic_trailblazing(
                compound_state, sampler_state, mcmc_move,
                state_parameters=[
                    (self.PAR_NAME_X0, [0.0, 1.0]),
                    (self.PAR_NAME_K, [k_initial, k_final])
                ],
                thermodynamic_distance=thermodynamic_distance,
                bidirectional_redistribution=bidirectional_redistribution,
                bidirectional_search_thermo_dist=bidirectional_search_thermo_dist,
                checkpoint_dir_path=checkpoint_dir_path
            )

        # Each test case is (bidirectional_redistribution, thermodynamic_distance, bidirectional_search_thermo_dist).
        test_cases = [
            (False, 1.0, 'auto'),
            (True, 2.0, 1.0),
            (True, 0.5, 1.0),
        ]

        for bidirectional_redistribution, thermodynamic_distance, bidirectional_search_thermo_dist in test_cases:
            with mmtools.utils.temporary_directory() as checkpoint_dir_path:
                # Compute the protocol.
                protocol = _call_run_thermodynamic_trailblazing(
                    thermodynamic_distance, bidirectional_redistribution,
                    bidirectional_search_thermo_dist, checkpoint_dir_path
                )

                # The number of frames returned should be equal to the number of
                # states in the protocol. If this was redistributed, this might
                # be different than the number of frames generated.
                sampler_states = read_trailblaze_checkpoint_coordinates(checkpoint_dir_path)
                len_protocol = len(protocol['testsystems_HarmonicOscillator_x0'])
                err_msg = (
                    f'bidirectional_redistribution={bidirectional_redistribution}, '
                    f'thermodynamic_distance={thermodynamic_distance}, '
                    f'bidirectional_search_thermo_dist={bidirectional_search_thermo_dist}: '
                    f'{len(sampler_states)} != {len_protocol}: {protocol}'
                )
                assert len(sampler_states) == len_protocol, err_msg

                # Now, resuming should give me the same exact protocol.
                resumed_protocol = _call_run_thermodynamic_trailblazing(
                    thermodynamic_distance, bidirectional_redistribution,
                    bidirectional_search_thermo_dist, checkpoint_dir_path
                )
                assert protocol == resumed_protocol
