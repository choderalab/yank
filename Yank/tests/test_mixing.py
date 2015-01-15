

"""
Test Cython and weave mixing code.
"""


import scipy.stats as stats
import Yank.mixing as mixing
import numpy as np

def mix_replicas(n_swaps=100, n_states=16, u_kl=None):
    """
    Utility function to generate "replicas" and call the mixing function a certain number of times

    Arguments
    ---------
    n_swaps : int (optional)
        The number of times to call the mixing code (default 100)
    n_states : int (optional)
        The number of replica states to include (default 16)
    u_kl : ndarray of floats (optional)
        Energies for each state. If None, will be initialized to 0s

    Returns
    -------
    replica_swap_results : n_states x n_swaps ndarray
       Contains the result of swapping replicas for each state in each swap attempt
    """


def test_even_mixing():
    pass

def test_general_mixing()