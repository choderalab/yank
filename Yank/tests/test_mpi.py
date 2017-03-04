#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test MPI utility functions in mpi.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from yank.mpi import *


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_run_single_node():
    """Test run_single_node function."""
    def add(a, b):
        return a + b
    mpicomm = get_mpicomm()

    node_executing = 1
    result = run_single_node(node_executing, task=add, a=3, b=4, broadcast_result=False)
    if mpicomm is not None and mpicomm.rank != node_executing:
        assert result is None
    else:
        assert result == 7

    result = run_single_node(node_executing, task=add, a=3, b=4, broadcast_result=True)
    assert result == 7


def test_on_single_node():
    """Test on_single_node decorator."""
    mpicomm = get_mpicomm()
    node_executing = 1

    @on_single_node(broadcast_result=False)
    def add(a, b):
        return a + b
    result = add(3, 4)
    if mpicomm is not None and mpicomm.rank != node_executing:
        assert result is None
    else:
        assert result == 7

    @on_single_node(broadcast_result=True)
    def add(a, b):
        return a + b
    result = add(3, 4)
    assert result == 7


def test_distribute():
    """Test distribute function."""
    def square(x):
        return x**2
    all_args = [1, 2, 3, 4]
    all_expected_results = [1, 4, 9, 16]

    mpicomm = get_mpicomm()
    if mpicomm is not None:
        job_indices = list(range(mpicomm.rank, 4, mpicomm.size))
        partial_expected_results = [all_expected_results[i] for i in job_indices]

    result = distribute(square, all_args, send_result_to='all')
    assert result == [1, 4, 9, 16]

    result = distribute(square, all_args, send_result_to=1)
    if mpicomm is not None and mpicomm.rank != 1:
        assert result == partial_expected_results
    else:
        assert result == all_expected_results

    result = distribute(square, all_args, send_result_to=None)
    assert result == partial_expected_results
