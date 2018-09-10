#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
MPI
===

Utilities to run on MPI.

Provide functions and decorators that simplify running the same code on
multiple nodes. One benefit is that serial and parallel code is exactly
the same.

Global variables
----------------
disable_mpi : bool
    Set this to True to force running serially.

Routines
--------
:func:`get_mpicomm`
    Automatically detect and configure MPI execution and return an
    MPI communicator.
:func:`run_single_node`
    Run a task on a single node.
:func:`on_single_node`
    Decorator version of :func:`run_single_node`.
:func:`distribute`
    Map a task on a sequence of arguments on all the nodes.
:func:`delay_termination`
    A context manager to delay the response to termination signals.
:func:`delayed_termination`
    A decorator version of :func:`delay_termination`.

"""


# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import functools
import logging
import os
import sys
import signal
from contextlib import contextmanager
from traceback import format_exception

import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# Force serial execution even in MPI environment.
disable_mpi = False

# A dummy MPI communicator used to simulate an MPI environment in tests.
_simulated_mpicomm = None


# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================

def get_mpicomm():
    """Retrieve the MPI communicator for this execution.

    The function automatically detects if the program runs on MPI by checking
    specific environment variables set by various MPI implementations. On
    first execution, it modifies sys.excepthook and register a handler for
    SIGINT, SIGTERM, SIGABRT to call MPI's ``Abort()`` to correctly terminate all
    processes.

    Returns
    -------
    mpicomm : mpi4py communicator or None
        The communicator for this node, None if the program doesn't run
        with MPI.

    """
    # If MPI execution is forcefully disabled, return None.
    if disable_mpi:
        return None

    # If MPI is simulated, return the Dummy implementation.
    if _simulated_mpicomm is not None:
        return _simulated_mpicomm

    # If we have already initialized MPI, return the cached MPI communicator.
    if get_mpicomm._is_initialized:
        return get_mpicomm._mpicomm

    # Check for environment variables set by mpirun. Variables are from
    # http://docs.roguewave.com/threadspotter/2012.1/linux/manual_html/apas03.html
    variables = ['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_MCA_ns_nds_vpid',
                 'PMI_ID', 'SLURM_PROCID', 'LAMRANK', 'MPI_RANKID',
                 'MP_CHILD', 'MP_RANK', 'MPIRUN_RANK',
                 'ALPS_APP_PE', # Cray aprun
                ]

    use_mpi = False
    for var in variables:
        if var in os.environ:
            use_mpi = True
            break

    # Return None if we are not running on MPI.
    if not use_mpi:
        logger.debug('Cannot find MPI environment. MPI disabled.')
        get_mpicomm._mpicomm = None
        get_mpicomm._is_initialized = True
        return get_mpicomm._mpicomm

    # Initialize MPI
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    # Override sys.excepthook to abort MPI on exception.
    def mpi_excepthook(type, value, traceback):
        sys.__excepthook__(type, value, traceback)
        node_name = '{}/{}'.format(MPI.COMM_WORLD.rank+1, MPI.COMM_WORLD.size)
        # logging.exception() automatically print the sys.exc_info(), but here
        # we may want to save the exception traceback of another MPI node so
        # we pass the traceback manually.
        logger.critical('MPI node {} raised an exception and called Abort()! The '
                        'exception traceback follows'.format(node_name), exc_info=value)
        # Flush everything.
        sys.stdout.flush()
        sys.stderr.flush()
        for logger_handler in logger.handlers:
            logger_handler.flush()
        # Abort MPI execution.
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
    # Use our exception handler.
    sys.excepthook = mpi_excepthook

    # Catch sigterm signals
    def handle_signal(signal, frame):
        if mpicomm.size > 1:
            mpicomm.Abort(1)
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]:
        signal.signal(sig, handle_signal)

    # Cache and return the MPI communicator.
    get_mpicomm._is_initialized = True
    get_mpicomm._mpicomm = mpicomm

    # Report initialization
    logger.debug("MPI initialized on node {}/{}".format(mpicomm.rank+1, mpicomm.size))

    return mpicomm

get_mpicomm._is_initialized = False  # Static variable


def run_single_node(rank, task, *args, **kwargs):
    """Run task on a single node.

    If MPI is not activated, this simply runs locally.

    Parameters
    ----------
    task : callable
        The task to run on node rank.
    rank : int
        The rank of the MPI communicator that must execute the task.
    broadcast_result : bool, optional
        If True, the result is broadcasted to all nodes. If False,
        only the node executing the task will receive the return
        value of the task, and all other nodes will receive None
        (default is False).
    sync_nodes : bool, optional
        If True, the nodes will be synchronized at the end of the
        execution (i.e. the task will be blocking) even if the
        result is not broadcasted  (default is False).

    Other Parameters
    ----------------
    args
        The ordered arguments to pass to task.
    kwargs
        The keyword arguments to pass to task.

    Returns
    -------
    result
        The return value of the task. This will be None on all nodes
        that is not the rank unless ``broadcast_result`` is set to True.

    Examples
    --------
    >>> def add(a, b):
    ...     return a + b
    >>> # Run 3+4 on node 0.
    >>> run_single_node(0, task=add, a=3, b=4, broadcast_result=True)
    7

    """
    broadcast_result = kwargs.pop('broadcast_result', False)
    sync_nodes = kwargs.pop('sync_nodes', False)
    result = None
    mpicomm = get_mpicomm()

    if mpicomm is not None:
        node_name = 'Node {}/{}'.format(mpicomm.rank+1, mpicomm.size)
    else:
        node_name = 'Single node'

    # Execute the task only on the specified node.
    if mpicomm is None or mpicomm.rank == rank:
        logger.debug('{}: executing {}'.format(node_name, task))
        result = task(*args, **kwargs)

    # Broadcast the result if required.
    if mpicomm is not None:
        if broadcast_result is True:
            logger.debug('{}: waiting for broadcast of {}'.format(node_name, task))
            result = mpicomm.bcast(result, root=rank)
        elif sync_nodes is True:
            logger.debug('{}: waiting for barrier after {}'.format(node_name, task))
            mpicomm.barrier()

    # Return result.
    return result


def on_single_node(rank, broadcast_result=False, sync_nodes=False):
    """A decorator version of run_single_node.

    Decorates a function to be always executed with :func:`run_single_node`.

    Parameters
    ----------
    rank : int
        The rank of the MPI communicator that must execute the task.
    broadcast_result : bool, optional
        If True the result is broadcasted to all nodes. If False,
        only the node executing the function will receive its return
        value, and all other nodes will receive None (default is False).
    sync_nodes : bool, optional
        If True, the nodes will be synchronized at the end of the
        execution (i.e. the task will be blocking) even if the
        result is not broadcasted (default is False).

    See Also
    --------
    run_single_node

    Examples
    --------
    >>> @on_single_node(rank=0, broadcast_result=True)
    ... def add(a, b):
    ...     return a + b
    >>> add(3, 4)
    7

    """
    def _on_single_node(task):
        @functools.wraps(task)
        def _wrapper(*args, **kwargs):
            kwargs['broadcast_result'] = broadcast_result
            kwargs['sync_nodes'] = sync_nodes
            return run_single_node(rank, task, *args, **kwargs)
        return _wrapper
    return _on_single_node


class _MpiProcessingUnit(object):
    """Context manager abstracting a single MPI processes and a group of nodes.

    Parameters
    ----------
    group_size : None, int or list of int, optional, default is None
        If not None, the ``distributed_args`` are distributed among groups of
        nodes that are isolated from each other. If an integer, the nodes are
        split into equal groups of ``group_size`` nodes. If a list of integers,
        the nodes are split in possibly unequal groups.

    Attributes
    ----------
    rank : int
        Either the rank of the node, or the color of the group.
    size : int
        Either the size of the mpicomm, or the number of groups.
    is_group

    """
    def __init__(self, group_size):
        # Store original mpicomm that we'll have to restore later.
        self._parent_mpicomm = get_mpicomm()

        # No need to split the comm if group size is None.
        if group_size is None:
            self._exec_mpicomm = self._parent_mpicomm
            self.rank = self._parent_mpicomm.rank
            self.size = self._parent_mpicomm.size
        else:
            # Determine the color that will be assigned to this node.
            node_color, n_groups = self._determine_node_color(group_size)
            # Split the mpicomm among nodes. Maintain same order using mpicomm.rank as rank.
            self._exec_mpicomm = self._parent_mpicomm.Split(color=node_color,
                                                            key=self._parent_mpicomm.rank)
            self.rank = node_color
            self.size = n_groups

    @property
    def is_group(self):
        """True if this is a group of nodes (i.e. :func:`get_mpicomm` is split)."""
        return self._exec_mpicomm != self._parent_mpicomm

    def exec_tasks(self, task, distributed_args, propagate_exceptions_to,
                   *other_args, **kwargs):
        """Run task on the given arguments.

        Parameters
        ----------
        propagate_exceptions_to : 'all', 'group', or None
            When one of the processes raise an exception during the task
            execution, this controls which other processes raise it.

        Returns
        -------
        results : list
            The list of the return values of the task. One for each argument.

        """
        # Determine where to propagate exceptions.
        if propagate_exceptions_to == 'all':
            exception_mpicomm = self._parent_mpicomm
        elif propagate_exceptions_to == 'group':
            exception_mpicomm = self._exec_mpicomm
        elif propagate_exceptions_to is None:
            exception_mpicomm = None
        else:
            raise ValueError('Unknown value for propagate_exceptions_to: '
                             '{}'.format(propagate_exceptions_to))

        # Determine name for logging.
        node_name = 'Node {}/{}'.format(self._exec_mpicomm.rank+1, self._exec_mpicomm.size)
        if self.is_group:
            node_name = 'Group {}/{} '.format(self.rank+1, self.size) + node_name

        # Compute all the results assigned to this node.
        results = []
        error = None
        for distributed_arg in distributed_args:
            logger.debug('{}: execute {}({})'.format(node_name, task.__name__, distributed_arg))
            try:
                results.append(task(distributed_arg, *other_args, **kwargs))
            except Exception as e:
                # Create an exception with same type and traceback but with node info.
                error = type(e)('{}: {}'.format(node_name, str(e)))
                error.with_traceback(e.__traceback__)
                # When sending the error over the network, the traceback seems to be lost,
                # so we create a string version of it, and expose it for others to print.
                traceback_str = ''.join(format_exception(type(e), e, e.__traceback__))
                error.traceback_str = traceback_str
                break

        # Propagate eventual exceptions to other nodes before raising.
        all_errors = []
        if exception_mpicomm is not None:
            all_errors = exception_mpicomm.allgather(error)
            all_errors = [e for e in all_errors if e is not None]
        # Each node raises its own exception first and then the others
        # (if any). This way the logs will be more informative.
        if error is not None:
            raise error
        elif len(all_errors) > 0:
            # Raise the first error received from a different MPI process.
            external_error = all_errors[0]
            # Include original traceback in the error message (indented 4 spaces).
            traceback_str = '\n    '.join(external_error.traceback_str.split('\n'))
            err_msg = ('{} received an exception from another MPI process. Original'
                       ' stack trace follow:\n{}').format(node_name, traceback_str)
            error = type(external_error)(err_msg)
            error.with_traceback(external_error.__traceback__)
            raise error

        return results

    def __enter__(self):
        # Cache execution mpicomm so that tasks will access the split mpicomm.
        get_mpicomm._mpicomm = self._exec_mpicomm
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Restore the original mpicomm.
        if self.is_group:
            self._exec_mpicomm.Free()
            get_mpicomm._mpicomm = self._parent_mpicomm

    def _determine_node_color(self, group_size):
        """Determine the color of this node."""
        try:  # Check if this is an integer.
            node_color = int(self._parent_mpicomm.rank / group_size)
            n_groups = int(np.ceil(self._parent_mpicomm.size / group_size))
        except TypeError:  # List of integers.
            # Check that the group division requested make sense. The sum
            # of all group sizes must be equal to the size of the mpicomm.
            cumulative_sum_nodes = np.cumsum(group_size)
            if cumulative_sum_nodes[-1] != self._parent_mpicomm.size:
                raise ValueError('The group division requested cannot be performed.\n'
                                 'Total number of nodes: {}\n'
                                 'Group nodes: {}'.format(self._parent_mpicomm.size, group_size))

            # The first group_size[0] nodes have color 0, the next group_size[1]
            # nodes have color 1 etc.
            node_color = next(i for i, v in enumerate(cumulative_sum_nodes)
                              if v > self._parent_mpicomm.rank)
            n_groups = len(group_size)
        return node_color, n_groups


def distribute(task, distributed_args, *other_args, send_results_to='all',
               propagate_exceptions_to='all', sync_nodes=False, group_size=None, **kwargs):
    """Map the task on a sequence of arguments to be executed on different nodes.

    If MPI is not activated, this simply runs serially on this node. The
    algorithm guarantees that each node will be assigned to the same job_id
    (i.e. the index of the argument in ``distributed_args``) every time.

    Parameters
    ----------
    task : callable
        The task to be distributed among nodes. The task will be called as
        ``task(distributed_args[job_id], *other_args, **kwargs)``, so the parameter
        to be distributed must the the first one.
    distributed_args : iterable
        The sequence of the parameters to distribute among nodes.
    send_results_to : int, 'all', or None, optional
        If the string 'all', the result will be sent to all nodes. If an
        int, the result will be send only to the node with rank ``send_results_to``.
        The return value of distribute depends on the value of this parameter
        (default is None).
    propagate_exceptions_to : 'all', 'group', or None, optional
        When one of the processes raise an exception during the task execution,
        this controls which other processes raise it (default is 'all'). This
        can be 'group' or None only if ``send_results_to`` is None.
    sync_nodes : bool, optional
        If True, the nodes will be synchronized at the end of the
        execution (i.e. the task will be blocking) even if the
        result is not shared (default is False).
    group_size : None, int or list of int, optional, default is None
        If not None, the ``distributed_args`` are distributed among groups of
        nodes that are isolated from each other. This is particularly useful
        if ``task`` also calls :func:`distribute`, since normally that would result
        in unexpected behavior.

        If an integer, the nodes are split into equal groups of ``group_size``
        nodes. If ``n_nodes % group_size != 0``, the first jobs are allocated
        more nodes than the latest. If a list of integers, the nodes are
        split in possibly unequal groups (see example below).

    Other Parameters
    ----------------
    other_args
        Other parameters to pass to task beside the assigned distributed
        parameters.
    kwargs
        Keyword arguments to pass to task beside the assigned distributed
        parameters.

    Returns
    -------
    all_results : list
        All the return values for all the arguments if the results where sent
        to the node, or only the return values of the arguments processed by
        this node otherwise.
    arg_indices : list of int, optional
        This is returned as part of a tuple ``(all_results, job_indices)`` only
        if ``send_results_to`` is set to an int or None. In this case ``all_results[i]``
        is the return value of ``task(all_args[arg_indices[i]])``.

    Examples
    --------
    >>> def square(x):
    ...     return x**2
    >>> distribute(square, [1, 2, 3, 4], send_results_to='all')
    [1, 4, 9, 16]

    When send_results_to is not set to `all`, the return value include also
    the indices of the arguments associated to the result.

    >>> distribute(square, [1, 2, 3, 4], send_results_to=0)
    ([1, 4, 9, 16], [0, 1, 2, 3])

    Divide the nodes in two groups of 2. The task, in turn, can
    distribute another task among the nodes in its own group.

    >>> def supertask(list_of_bases):
    ...     return distribute(square, list_of_bases, send_results_to='all')
    >>> list_of_supertask_args = [[1, 2, 3], [4], [5, 6]]
    >>> distribute(supertask, distributed_args=list_of_supertask_args,
    ...            send_results_to='all', group_size=2)
    [[1, 4, 9], [16], [25, 36]]

    """
    n_jobs = len(distributed_args)

    # If MPI is not activated, just run serially.
    if get_mpicomm() is None:
        logger.debug('Running {} serially.'.format(task.__name__))
        all_results = [task(job_args, *other_args, **kwargs) for job_args in distributed_args]
        if send_results_to == 'all':
            return all_results
        else:
            return all_results, list(range(n_jobs))

    # We can't propagate exceptions to a subset of nodes if we need to send all the results.
    if send_results_to is not None and propagate_exceptions_to != 'all':
        raise ValueError('Cannot propagate exceptions to a subset of nodes '
                         'with send_results_to != None')

    # Split the default mpicomm into group if necessary.
    with _MpiProcessingUnit(group_size) as processing_unit:
        # Determine the jobs that this node has to run.
        node_job_ids = range(processing_unit.rank, n_jobs, processing_unit.size)
        node_distributed_args = [distributed_args[job_id] for job_id in node_job_ids]

        # Run all jobs.
        results = processing_unit.exec_tasks(task, node_distributed_args, propagate_exceptions_to,
                                             *other_args, **kwargs)

        # If we have split the mpicomm, nodes belonging to the same group
        # have duplicate results. We gather only results from one node.
        if processing_unit.is_group and get_mpicomm().rank != 0:
            results_to_send = []
        else:
            results_to_send = results

    # Share result as requested.
    mpicomm = get_mpicomm()
    node_name = 'Node {}/{}'.format(mpicomm.rank+1, mpicomm.size)
    if send_results_to == 'all':
        logger.debug('{}: allgather results of {}'.format(node_name, task.__name__))
        all_results = mpicomm.allgather(results_to_send)
    elif isinstance(send_results_to, int):
        logger.debug('{}: sending results of {} to {}'.format(node_name, task.__name__,
                                                              send_results_to))
        all_results = mpicomm.gather(results_to_send, root=send_results_to)

        # If this is not the receiving node, we can safely return.
        if mpicomm.rank != send_results_to:
            return results, list(node_job_ids)
    else:
        assert send_results_to is None  # Safety check.
        if sync_nodes is True:
            logger.debug('{}: waiting for barrier after {}'.format(node_name, task.__name__))
            mpicomm.barrier()
        return results, list(node_job_ids)

    # all_results is a list of list of results. The internal lists of
    # results are ordered by rank. We need to reorder the results as a
    # flat list of results ordered by job_id.

    # job_indices[job_id] is the tuple of indices (rank, i). The result
    # of job_id is stored in all_results[rank][i].
    job_indices = []
    max_jobs_per_node = max([len(r) for r in all_results])
    for i in range(max_jobs_per_node):
        for rank in range(mpicomm.size):
            # Not all nodes have executed max_jobs_per_node tasks.
            if len(all_results[rank]) > i:
                job_indices.append((rank, i))

    # Reorder the results.
    all_results = [all_results[rank][i] for rank, i in job_indices]

    # Return result.
    if send_results_to == 'all':
        return all_results
    else:
        return all_results, list(range(n_jobs))


@contextmanager
def delay_termination():
    """Context manager to delay handling of termination signals.

    This allows to avoid interrupting tasks such as writing to the file
    system, which could result in the corruption of the file.

    """
    signals_to_catch = [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]
    old_handlers = {signum: signal.getsignal(signum) for signum in signals_to_catch}
    signals_received = {signum: None for signum in signals_to_catch}

    def delay_handler(signum, frame):
        signals_received[signum] = (signum, frame)

    # Set handlers fot delay
    for signum in signals_to_catch:
        signal.signal(signum, delay_handler)

    yield  # Resume program

    # Restore old handlers
    for signum, handler in old_handlers.items():
        signal.signal(signum, handler)

    # Fire delayed signals
    for signum, s in signals_received.items():
        if s is not None:
            old_handlers[signum](*s)


def delayed_termination(func):
    """Decorator that runs the function with :func:`delay_termination`."""
    @functools.wraps(func)
    def _delayed_termination(*args, **kwargs):
        with delay_termination():
            return func(*args, **kwargs)
    return _delayed_termination


# ==============================================================================
# MPI TEST CLASSES
# ==============================================================================

class _DummyMPIComm():
    """A Dummy MPI Communicator."""

    def __init__(self, rank=0, size=4):
        self.rank = rank
        self.size = size


@contextmanager
def _simulated_mpi_environment(**kwargs):
    """Context manager to temporarily set a simulated MPI environment.

    Parameters
    ----------
    **kwargs : dict
        The parameters to pass to _DummyMPIComm constructor.

    """
    global _simulated_mpicomm
    old_simulated_mpicomm = _simulated_mpicomm
    _simulated_mpicomm = _DummyMPIComm(**kwargs)
    try:
        yield
    finally:
        _simulated_mpicomm = old_simulated_mpicomm


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
