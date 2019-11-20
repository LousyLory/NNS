"""
kernel approximation wrapper
"""

import numpy as np
from utils import compute_squared_distance_no_loops, compute_one_vs_all_squared_distance

def approximate_kernels(similarity_matrix, base_vector, selected_vector_set):
    """
    performs approximate online by using leverage scores
    """
    return updated_similarity_matrix, selected_vector_set

def online_row_sampling(base_vector, selected_vector_set):
    """
    performs online row sampling as in https://arxiv.org/pdf/1604.05448.pdf
    """
