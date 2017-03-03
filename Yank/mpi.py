#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""Utilities to run on MPI.

Provide functions and decorators that simplify running the same code on
multiple nodes. One benefit is that serial and parallel code is exactly
the same.

Global variables
----------------
disable_mpi : bool
    Set this to True to force running serially.

Routines
--------
get_mpicomm
    Automatically detect and configure MPI execution and return an
    MPI communicator.
delay_termination
    A context manager to delay the response to termination signals.
delayed_termination
    A decorator version of delay_termination.

"""


# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import os
import sys
import signal
from contextlib import contextmanager

# TODO drop this when we drop Python 2 support
from openmoltools.utils import wraps_py2


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

disable_mpi = False


# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================

def get_mpicomm():
    """Retrieve the MPI communicator for this execution.

    The function automatically detects if the program runs on MPI by checking
    specific environment variables set by various MPI implementations. On
    first execution, it modifies sys.excepthook and register a handler for
    SIGINT, SIGTERM, SIGABRT to call Abort() to correctly terminate all
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

    # If we have already initialized MPI, return the cached MPI communicator.
    if get_mpicomm._is_initialized:
        return get_mpicomm._mpicomm

    # Check for environment variables set by mpirun. Variables are from
    # http://docs.roguewave.com/threadspotter/2012.1/linux/manual_html/apas03.html
    variables = ['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_MCA_ns_nds_vpid',
                 'PMI_ID', 'SLURM_PROCID', 'LAMRANK', 'MPI_RANKID',
                 'MP_CHILD', 'MP_RANK', 'MPIRUN_RANK']
    use_mpi = False
    for var in variables:
        if var in os.environ:
            use_mpi = True
            break

    # Return None if we are not running on MPI.
    if not use_mpi:
        get_mpicomm._mpicomm = None
        return get_mpicomm._mpicomm

    # Initialize MPI
    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()
    mpicomm = MPI.COMM_WORLD

    # Override sys.excepthook to abort MPI on exception
    def mpi_excepthook(type, value, traceback):
        sys.__excepthook__(type, value, traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        if mpicomm.size > 1:
            mpicomm.Abort(1)
    # Use our eception handler
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
        If True the result is broadcasted to all nodes. If False,
        only the node executing the task will receive the return
        value of the task, and all other nodes will receive None.

    Other Parameters
    ----------------
    *args
        The ordered arguments to pass to task.
    *kwargs
        The keyword arguments to pass to task.

    Returns
    -------
    result
        The return value of the task. This will be None on all nodes
        that is not the rank unless broadcast_result is set to True.

    Examples
    --------
    >>> def add(a, b):
    ...     return a + b
    >>> # Run 3+4 on node 0.
    >>> run_single_node(0, task=add, a=3, b=4, broadcast_result=True)
    7

    """
    broadcast_result = kwargs.pop('broadcast_result', False)
    result = None
    mpicomm = get_mpicomm()

    # Execute the task only on the specified node.
    if mpicomm is None or mpicomm.rank == rank:
        result = task(*args, **kwargs)

    # Broadcast the result if required.
    if mpicomm is not None and broadcast_result:
        result = mpicomm.bcast(result, rank=rank)

    # Return result.
    return result


def on_single_node(rank, broadcast_result=False):
    """A decorator version of run_single_node.

    Decorates a function to be always executed with run_single_node.

    Parameters
    ----------
    rank : int
        The rank of the MPI communicator that must execute the task.
    broadcast_result : bool, optional
        If True the result is broadcasted to all nodes. If False,
        only the node executing the function will receive its return
        value, and all other nodes will receive None.

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
        @wraps_py2(task)
        def _wrapper(*args, **kwargs):
            kwargs['broadcast_result'] = broadcast_result
            return run_single_node(rank, task, *args, **kwargs)
        return _wrapper
    return _on_single_node


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
    """Decorator that runs the function with delay_termination()."""
    @wraps_py2(func)
    def _delayed_termination(*args, **kwargs):
        with delay_termination():
            return func(*args, **kwargs)
    return _delayed_termination


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
