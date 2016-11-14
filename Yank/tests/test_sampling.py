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

    # We pass as reference_LJ_state and reference_LJ_state the normal
    # reference state as we just want to check that they are correctly
    # set on resume
    reference_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    reference_state.system = toluene_test.system
    reference_LJ_state = copy.deepcopy(base_state)
    reference_LJ_expanded_state = copy.deepcopy(base_state)

    with enter_temp_directory():
        store_file_name = 'simulation.nc'
        simulation = ModifiedHamiltonianExchange(store_file_name)
        simulation.create(base_state, alchemical_states, positions, mc_atoms=ligand_atoms,
                          reference_state=reference_state,
                          reference_LJ_state=reference_LJ_state,
                          reference_LJ_expanded_state=reference_LJ_expanded_state)

        # Clean up simulation and resume
        del simulation
        simulation = ModifiedHamiltonianExchange(store_file_name)
        simulation.resume()
        assert simulation.reference_state is not None
        assert simulation.reference_LJ_state is not None
        assert simulation.reference_LJ_expanded_state is not None
