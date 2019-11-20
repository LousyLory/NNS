"""
approximate nearest neighbor search wrapper
"""

import numpy as np
from utils import compute_squared_distance_no_loops, compute_one_vs_all_squared_distance

def single_query_nearest(similarity_matrix, selected_vector_set, query):
    """
    perform approximate online nearest neighbor
    """
    SSD_vecs, min_index = compute_one_vs_all_squared_distance(selected_vector_set, query)

    return SSD_vec, min_index

def all_query_nearest(similarity_matrix, selected_vector_set, queries):
    """
    perform approximate nearesr neighbor for queries
    """
    dist_mat = compute_squared_distance_no_loops(selected_vector_set, queries)
    chosen_indices = np.argmin(dist_mat, axis = 1)

    return dist_mat, chosen_indices
