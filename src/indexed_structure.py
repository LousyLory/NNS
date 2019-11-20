"""
code to compute the indexed structure online
"""

import numpy as np
from kernel_approximation import approximate_kernels, online_row_sampling

def randomize_data(base_vectors):
    """
    randomly permute the data rows
    """
    np.random.shuffle(base_vectors)

    return base_vectors

def compute_online(base_vectors, method = "AK"):
    """
    computes the indexed structure online
    """
    K = []
    indexed_set = []
    randommized_data = randomize_data(base_vectors)

    if method == "AK":
        for i in range(len(randomized_data)):
            K, indexed_set = approximate_kernels(randomized_data, K, indexed_set)
    if method == "ORS":
        for i in range(len(randomized_data)):
            indexed_set = online_row_sampling(randomized_data, indexed_set)

    return indexed_set
