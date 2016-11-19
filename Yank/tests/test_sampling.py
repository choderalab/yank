#!/usr/local/bin/env python

"""
Test sampling.py facility.

"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

from openmmtools import testsystems
from mdtraj.utils import enter_temp_directory

from yank.sampling import *


# ==============================================================================
# TESTS
# ==============================================================================

def test_resuming():
    """Test that sampling correctly resumes."""
    # Prepare ModifiedHamiltonianExchange arguments
    toluene_test = testsystems.TolueneImplicit()
    ligand_atoms = range(15)
    alchemical_factory = AbsoluteAlchemicalFactory(toluene_test.system,
                                                   ligand_atoms=ligand_atoms)

    base_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    base_state.system = alchemical_factory.alchemically_modified_system

    alchemical_state1 = AlchemicalState(lambda_electrostatics=1.0, lambda_sterics=1.0)
    alchemical_state0 = AlchemicalState(lambda_electrostatics=0.0, lambda_sterics=0.0)
    alchemical_states = [alchemical_state1, alchemical_state0]

    positions = toluene_test.positions

    # We pass as fully_interacting_expanded_state and noninteracting_expanded_state the normal
    # reference state as we just want to check that they are correctly
    # set on resume
    fully_interacting_expanded_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    fully_interacting_expanded_state.system = toluene_test.system
    noninteracting_expanded_state = copy.deepcopy(base_state)

    with enter_temp_directory():
        store_file_name = 'simulation.nc'
        simulation = ModifiedHamiltonianExchange(store_file_name)
        simulation.create(base_state, alchemical_states, positions, mc_atoms=ligand_atoms,
                          fully_interacting_expanded_state=fully_interacting_expanded_state,
                          noninteracting_expanded_state = noninteracting_expanded_state)

        # Clean up simulation and resume
        del simulation
        simulation = ModifiedHamiltonianExchange(store_file_name)
        simulation.resume()
        assert simulation.fully_interacting_expanded_state is not None
        assert simulation.noninteracting_expanded_state is not None
