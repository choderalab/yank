

"""
Test Cython and weave mixing code.
"""


import scipy.stats as stats
import yank.mixing._mix_replicas as mixing
import yank.mixing._mix_replicas_old as mix_old
import numpy as np
import copy

def mix_replicas(n_swaps=100, n_states=16, u_kl=None, nswap_attempts=None):
    """
    Utility function to generate replicas and call the mixing function a certain number of times

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
    replica_states = np.array(range(n_states), np.int64)
    if nswap_attempts is None:
        nswap_attempts = n_states**4
    Nij_proposed =  np.zeros([n_states,n_states], dtype=np.int64)
    Nij_accepted = np.zeros([n_states,n_states], dtype=np.int64)
    permutation_list = []
    for i in range(n_swaps):
        mixing._mix_replicas_cython(nswap_attempts, n_states, replica_states, u_kl, Nij_proposed, Nij_accepted)
        #mix_old._mix_all_replicas_weave(n_states, replica_states, u_kl, Nij_proposed, Nij_accepted)
        permutation_list.append(copy.deepcopy(replica_states))
        print("Completed swap set %d" % i)
    permutation_list_np = np.array(permutation_list, dtype=np.int64)
    return permutation_list_np



def permanent(data):
    """
    This code was taken from github.com/arulalant/numpy-utils/lib/numutils.py
    permanent: Square Matrix permanent
    It would be numpy data or list data.
    Matrix permanent is just same as determinant of the matrix but change -ve
    sign into +ve sign through out its calculation of determinant.
    eg 1:
        >>> a = numpy.ones(9).reshape((3,3))
        >>> z = permanent(a)
        >>> print z
        >>> 6.0
    eg 2:
        >>> a = numpy.ones(16).reshape((4,4))
        >>> z = permanent(a)
        >>> print z
        >>> 24.0
    Written By : Arulalan.T
    Date : 01.08.2012
    """
    # initialize the local variables everytime when the function call by
    # itself.

    # initialize the result variable & row index as zero.
    res = 0
    rowIdx = 0
    data = np.array(data)
    dshape = data.shape
    if dshape[0] != dshape[1]:
        print "The data shape, ", dshape
        raise ValueError("The passed data is not square matrix")

    for colIdx in range(dshape[1]):
        # loop through the column index of the first row of data

        if dshape == (2, 2):
            # data shape is (2,2). So calculate the 2x2 matrix permanent
            # and return it. (return is import for routine call)
            return (data[0][0] * data[1][1]) + (data[0][1] * data[1][0])
        else:
            # get the value of the data(rowIdx, colIdx)
            rowVal = data[rowIdx][colIdx]
            # matrix shape is higher than the (2,2). So remove the current
            # row and column elements from the data and do calculate the
            # permanent for the rest of the matrix data.

            # multiply with the rowVal and add to the result.
            res += rowVal * permanent(remove_nxm(data, rowIdx, colIdx))
    # end of for colIdx in range(dshape[1]):
    return res
# end of def permanent(data):


def remove_nxm(data, row_idx, col_idx):
    """
    This function removes a given row and column from a numpy array
    """
    row_deleted = np.delete(data, row_idx, axis=0)
    col_row_deleted = np.delete(row_deleted, col_idx, axis=1)
    return col_row_deleted

def calculate_expected_state_probabilities(u_kl):
    """
    This function calculates the expected proportion of counts for each state j of replica i with k states
    as p(s_j) \propto e^{-u_{ij}} perm{X_i}

    Where X_i is the exponentiated reduced potential matrix ommitting row and column i

    Arguments
    ---------
    u_kl : n_states x n_states np.array of floats
        The reduced potential for each replica state

    Returns
    -------
    expected_state_probabilities : n_states x n_states np.array of floats
        For each replica (row), a set of columns reflecting the expected
        probability of visiting that state
    """
    expected_state_probabilities = np.zeros_like(u_kl)
    exponentiated_potentials = np.exp(-u_kl)
    n_replicas, n_states = u_kl.shape
    matrix_permanents = np.zeros(n_replicas)
    for replica in range(n_replicas):
        x_i = remove_nxm(exponentiated_potentials, replica, replica)
        matrix_permanents[replica] = permanent(x_i)
    for replica in range(n_replicas):
        for state in range(n_states):
            expected_state_probabilities[replica, state] = exponentiated_potentials[replica, state] * matrix_permanents[state]
    probability_sums = expected_state_probabilities.sum(axis=1)
    return expected_state_probabilities / probability_sums[:, np.newaxis]

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
    Testing Cython mixing code with 1000 swap attempts and uniform 0 energies
    """
    if verbose: print("Testing Cython mixing code with uniform zero energies")
    n_swaps = 1000
    n_states = 16
    corrected_threshold = 0.001 / n_states
    permutation_list = mix_replicas(n_swaps=n_swaps, n_states=n_states)
    state_counts = calculate_state_counts(permutation_list, n_swaps, n_states)
    for replica in range(n_states):
        _, p_val = stats.chisquare(state_counts[replica,:])
        if p_val < corrected_threshold:
            print("Detected a significant difference between expected even mixing\n")
            print("and observed mixing, p=%f" % p_val)
            raise Exception("Replica %d failed the even mixing test" % replica)
    return 0


def test_general_mixing(verbose=True):
    """
    Testing Cython mixing code with 1000 swap attempts and random energies
    """
    if verbose: print("Testing Cython mixing code with random energies")
    n_swaps = 100
    n_states = 4
    corrected_threshold = 0.001 / n_states
    u_kl = np.array(np.random.randn(n_states, n_states), dtype=np.float64)
    print(u_kl)
    permutation_list = mix_replicas(n_swaps=n_swaps, n_states=n_states, u_kl=u_kl, nswap_attempts=4194304)
    state_counts = np.array(calculate_state_counts(permutation_list, n_swaps, n_states), dtype=np.int64)
    expected_state_probabilities = calculate_expected_state_probabilities(u_kl)
    expected_state_counts = np.array(n_swaps*expected_state_probabilities, dtype=np.int64)
    for replica in range(n_states):
        print replica
        print(state_counts[replica, :])
        print(expected_state_counts[replica, :])
        _, p_val = stats.chisquare(state_counts[replica,:], expected_state_counts[replica, :])
        print(str(p_val))
       # if p_val < corrected_threshold:
       #     print("Detected a significant difference between expected mixing\n")
       #     print("and observed mixing, p=%f" % p_val)
       #     raise Exception("Replica %d failed the mixing test" % replica)

if __name__ == "__main__":
  # test_even_mixing()
   test_general_mixing()