

"""
Test Cython and weave mixing code.
"""


import scipy.stats as stats
import mixing._mix_replicas as _mix_replicas
import numpy as np
import copy

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
    permutation_list : n_states x n_swaps ndarray of np.int64
        Contains the result of each swap
    """

    if u_kl is None:
        u_kl = np.zeros([n_states, n_states], dtype=np.float64)
    replica_states = range(n_states)
    Nij_proposed =  np.zeros([n_states,n_states], dtype=np.int64)
    Nij_accepted = np.zeros([n_states,n_states], dtype=np.int64)
    permutation_list = copy.deepcopy(replica_states)
    for i in range(n_swaps):
        _mix_replicas._mix_replicas_cython(n_states, replica_states, u_kl, Nij_proposed, Nij_accepted)
        permutation_list.append(copy.deepcopy(replica_states))
    permutation_list_np = np.array(permutation_list, dtype=np.int64)
    return permutation_list_np


def calculate_state_counts(permutation_list, n_swaps, n_states):
    """
    This function accepts a list of permutation vectors, and for each replica,
    produces a list of the number of occurrences of each state.

    Arguments
    ---------
    permutation_list : n_states x n_swaps ndarray of np.int64
        For each swap attempt, a permutation vector n_states long
    n_swaps : int
        The number of swap attempts
    n_states : int
        The number of replica states

    Returns
    -------
    state_counts : n_states x n_states numpy array of ints
        For each replica, contains the number of occurrences of each state (replica_index x state)
    """
    state_counts = np.zeros([n_states, n_states])
    for swap in range(n_swaps):
        for replica in range(n_states):
            current_state = permutation_list[swap, replica]
            state_counts[replica, current_state] += 1
    return state_counts



def test_even_mixing(verbose=True):
    """
    Using 1000 swap attempts, this code sets all energies to 0 and tests to see if the observed
    mixing is uniform (as would be expected) using a chi-square test and a cutoff of 0.001
    for Bonferroni-corrected p-values
    """
    if verbose: print("Testing Cython mixing code with uniform zero energies")
    n_swaps = 1000
    n_states = 16
    permutation_list = mix_replicas(n_swaps=n_swaps, n_states=n_states)
    state_counts = calculate_state_counts(permutation_list, n_swaps, n_states)
    for replica in range(n_states):
        _, p_val = stats.chisquare(state_counts[replica,:]) / n_states
        if p_val < 0.001:
            print("Detected a significant difference between expected even mixing\n")
            print("and observed mixing, p=%f" % p_val)
            raise Exception("Replica %d failed the even mixing test" % replica)
    return 0


def test_general_mixing(verbose=True):
    """
    Using 1000 swap attempts, this code generates a random matrix of energies from a multivariate
    normal distribution, and tests to see if the distribution of states follows the correct
    """
    if verbose: print("Testing Cython mixing code with random energies")
    n_swaps = 1000
    n_states = 16
    u_kl = np.random.randn(size=(n_states, n_states))
    permutation_list = mix_replicas(n_swaps=n_swaps, n_states=n_states, u_kl=u_kl)
    state_counts = calculate_state_counts(permutation_list, n_swaps, n_states)

