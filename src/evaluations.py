"""
functions to compute accuracy of our methods
"""

import numpy as np

def compute_regret(ideal_distance, experimental_distance):
    """
    computes regret given the parameters
    """
    regret = np.abs(ideal_distance - experimental_distance)
    total_regret = np.sum(regret)
    average_regret = total_regret / len(regret)

    return total_regret

def compute_nn_dist(experimental_distance):
    """
    computes NN-dist
    """
    dist = np.sum(experimental_distance) / len(experimental_distance)

    return dist
