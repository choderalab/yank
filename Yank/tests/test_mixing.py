

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
    u_kl : n_states x n_states ndarray of float64 (optional)
        Energies for each state. If None, will be initialized to zeros

    Returns
    -------
    Nij_proposed : n_states x n_states ndarray of np.int64
        Contains the number of times a given swap was proposed
    Nij_accepted : n_states x n_states ndarray of np.int64
        Contains the number of times a given swap was accepted
    """

    if u_kl is None:
        u_kl = np.zeros([n_states, n_states], dtype=np.float64)
    replica_states = range(n_states)
    Nij_proposed =  np.zeros([n_states,n_states], dtype=np.int64)
    Nij_accepted = np.zeros([n_states,n_states], dtype=np.int64)
    for i in range(n_swaps):
        mixing._mix_replicas_cython(n_states, replica_states, u_kl, Nij_proposed, Nij_accepted)
    return Nij_proposed, Nij_accepted




def test_even_mixing():
    pass

def test_general_mixing():