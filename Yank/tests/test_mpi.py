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

import nose
from simtk import unit

from yank.mpi import *


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

NODE_RANK = 0  # The node rank passed to rank or send_results_to.


# =============================================================================
# UTILITY FUNCTION
# =============================================================================

def assert_is_equal(a, b):
    err_msg = '{} != {}'.format(a, b)
    try:
        assert a == b, err_msg
    except ValueError:
        # This is a list or tuple of numpy arrays.
        try:
            for element_a, element_b in zip(a, b):
                assert_is_equal(element_a, element_b)
        except AssertionError:
            raise AssertionError(err_msg)


# =============================================================================
# TEST CASES
# =============================================================================

def square(x):
    return x**2


def multiply(a, b):
    return a * b


@on_single_node(rank=NODE_RANK, broadcast_result=True)
def multiply_decorated_broadcast(a, b):
    return multiply(a, b)


@on_single_node(rank=NODE_RANK, broadcast_result=False)
def multiply_decorated_nobroadcast(a, b):
    return multiply(a, b)


class MyClass(object):

    def __init__(self, par):
        self.par = par

    @staticmethod
    def square_static(x):
        return x**2

    @staticmethod
    def multiply_static(a, b):
        return a * b

    @staticmethod
    @on_single_node(rank=NODE_RANK, broadcast_result=True)
    def multiply_decorated_broadcast_static(a, b):
        return a * b

    @classmethod
    @on_single_node(rank=NODE_RANK, broadcast_result=False)
    def multiply_decorated_nobroadcast_static(cls, a, b):
        return cls.multiply_static(a, b)

    def multiply_by_par(self, a):
        return self.par * a

    @on_single_node(rank=NODE_RANK, broadcast_result=True)
    def multiply_by_par_decorated_broadcast(self, a):
        return self.multiply_by_par(a)

    @on_single_node(rank=NODE_RANK, broadcast_result=False)
    def multiply_by_par_decorated_nobroadcast(self, a):
        return self.multiply_by_par(a)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_run_single_node():
    """Test run_single_node function."""
    mpicomm = get_mpicomm()
    my_instance = MyClass(3.0)

    # Test_case: (function, args, kwargs)
    test_cases = [
        (multiply, (3, 4), {}),
        (multiply, (), {'a': 5, 'b': 4}),
        (multiply, (2, 'teststring',), {}),
        (multiply, (5, [3, 4],), {}),
        (multiply, (4, np.array([3, 4, 5.0]),), {}),
        (multiply, (3.0, unit.Quantity(np.array([3, 4, 5.0]), unit=unit.angstrom),), {}),
        (square, (5,), {}),
        (square, (), {'x': 2}),
        (MyClass.multiply_static, (3, 4), {}),
        (MyClass.multiply_static, (), {'a': 5, 'b': 4}),
        (MyClass.square_static, (5,), {}),
        (MyClass.square_static, (), {'x': 2}),
        (my_instance.multiply_by_par, (4,), {}),
        (my_instance.multiply_by_par, (), {'a': 2}),
    ]

    for task, args, kwargs in test_cases:
        expected_result = task(*args, **kwargs)
        for broadcast_result in [True, False]:
            result = run_single_node(NODE_RANK, task, *args,
                                     broadcast_result=broadcast_result, **kwargs)
            if not broadcast_result and mpicomm is not None and mpicomm.rank != NODE_RANK:
                assert result is None
            else:
                assert_is_equal(result, expected_result)


def test_on_single_node():
    """Test on_single_node decorator."""
    mpicomm = get_mpicomm()
    my_instance = MyClass(3.0)

    # Test case: (function, args, kwargs, broadcast_result, expected_result)
    test_cases = [
        (multiply_decorated_broadcast, (3, 4), {}, True, 12),
        (multiply_decorated_nobroadcast, (), {'a': 5, 'b': 5}, False, 25),
        (MyClass.multiply_decorated_broadcast_static, (3, 4), {}, True, 12),
        (MyClass.multiply_decorated_nobroadcast_static, (), {'a': 5, 'b': 5}, False, 25),
        (my_instance.multiply_by_par_decorated_broadcast, (4,), {}, True, 4 * my_instance.par),
        (my_instance.multiply_by_par_decorated_broadcast, (), {'a': 5}, True, 5 * my_instance.par),
    ]

    for task, args, kwargs, broadcast_result, expected_result in test_cases:
        result = task(*args, **kwargs)
        if not broadcast_result and mpicomm is not None and mpicomm.rank != NODE_RANK:
            assert result is None
        else:
            assert_is_equal(result, expected_result)


def test_distribute():
    """Test distribute function."""
    mpicomm = get_mpicomm()
    my_instance = MyClass(4)

    # Testcase: (function, distributed_args)
    test_cases = [
        (square, [1, 2, 3]),
        (MyClass.square_static, [1, 2, 3, 4]),
        (my_instance.multiply_by_par, [1, 2, 3, 4, 5]),
        (my_instance.multiply_by_par, ['a', 'b', 'c', 'd', 'e', 'f']),
        (my_instance.multiply_by_par, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        (my_instance.multiply_by_par, [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([7, 8])]),
        (my_instance.multiply_by_par, [unit.Quantity(np.array([1, 2]), unit=unit.angstrom),
                                       unit.Quantity(np.array([3, 4]), unit=unit.angstrom),
                                       unit.Quantity(np.array([5, 6]), unit=unit.angstrom)]),
    ]

    for task, distributed_args in test_cases:
        all_indices = list(range(len(distributed_args)))
        all_expected_results = [task(x) for x in distributed_args]

        # Determining full and partial results.
        if mpicomm is not None:
            partial_job_indices = list(range(mpicomm.rank, len(distributed_args), mpicomm.size))
        else:
            partial_job_indices = all_indices
        partial_expected_results = [all_expected_results[i] for i in partial_job_indices]

        result = distribute(task, distributed_args, send_results_to='all')
        assert_is_equal(result, all_expected_results)

        result = distribute(task, distributed_args, send_results_to=NODE_RANK)
        if mpicomm is not None and mpicomm.rank != NODE_RANK:
            assert_is_equal(result, (partial_expected_results, partial_job_indices))
        else:
            assert_is_equal(result, (all_expected_results, all_indices))

        result = distribute(task, distributed_args, send_results_to=None)
        assert_is_equal(result, (partial_expected_results, partial_job_indices))


def test_distribute_groups():
    """Test distribute jobs among groups of nodes."""
    # Configuration.
    group_size = 2
    list_of_supertask_args = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]

    def supertask(list_of_bases):
        """Compute square of all the bases."""
        squared_values = distribute(square, list_of_bases, send_results_to='all')
        mpicomm = get_mpicomm()
        if mpicomm is None:
            mpi_size = 0
        else:
            mpi_size = mpicomm.size
        return squared_values, mpi_size

    # Super tasks will store results in the same temporary directory.
    all_results = distribute(supertask, distributed_args=list_of_supertask_args, sync_nodes=True,
                             send_results_to='all', group_size=group_size)

    mpicomm = get_mpicomm()
    n_jobs = len(list_of_supertask_args)

    # Find the job_ids assigned to the last group and the size of its communicator.
    if mpicomm is not None:
        n_groups = int(np.ceil(mpicomm.size / group_size))
        last_group_size = group_size - mpicomm.size % group_size
        last_group_job_ids = set(range(n_groups-1, n_jobs, n_groups))

    # Verify all tasks.
    for job_id, (supertask_args, job_result) in enumerate(zip(list_of_supertask_args, all_results)):
        squared_values, mpi_size = job_result  # Unpack.

        # Check that result is correct.
        assert len(supertask_args) == len(squared_values)
        for idx, value in enumerate(squared_values):
            assert value == supertask_args[idx]**2

        # Check that the correct group executed this task.
        if mpicomm is None:
            expected_mpi_size = 0
        elif job_id in last_group_job_ids:
            expected_mpi_size = last_group_size
        else:
            expected_mpi_size = 2
        assert mpi_size == expected_mpi_size


def test_exception_handling():
    """distribute() propagates exceptions correctly."""
    group_size = 2
    mpicomm = get_mpicomm()

    def task(arg):
        if arg == 0 and (mpicomm is None or get_mpicomm().rank == 0):
            raise RuntimeError(str(arg))
        return

    def call_distribute(propagate_exceptions_to, send_results_to=None):
        distribute(task, distributed_args=list(range(2*group_size)), send_results_to=send_results_to,
                   propagate_exceptions_to=propagate_exceptions_to, group_size=group_size)

    # When propagate_exceptions_to is set to 'all', the exception is raised in all nodes.
    with nose.tools.assert_raises(RuntimeError):
        call_distribute(propagate_exceptions_to='all', send_results_to='all')

    # It's impossible to send results and propagate exceptions to a subset of nodes.
    if mpicomm is not None:
        with nose.tools.assert_raises(ValueError):
            call_distribute(propagate_exceptions_to='group', send_results_to='all')

    # When propagate_exceptions_to is set to 'group', the exception is raised in the first group only.
    if mpicomm is None or mpicomm.rank in list(range(group_size)):
        with nose.tools.assert_raises(RuntimeError):
            call_distribute(propagate_exceptions_to='group')
    else:
        call_distribute(propagate_exceptions_to='group')  # No exception raised.

    # When propagate_exceptions_to is set to None, the exception is raised in the first node.
    if mpicomm is None or mpicomm.rank == 0:
        with nose.tools.assert_raises(RuntimeError):
            call_distribute(propagate_exceptions_to=None)
    else:
        call_distribute(propagate_exceptions_to=None)  # No exception raised.
