"""
Code by Nick and Ari
"""

import numpy as np

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def ivecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def compute_squared_distance_no_loops(X, Y):
    """
    input:
    X - train_data
    Y - test_data

    output:
    dist_mat - computed distance values
    """
    num_test = Y.shape[0]
    num_train = X.shape[0]
    dist_mat = np.zeros((num_test, num_train))

    # compute distance
    sum1 = np.sum(np.power(Y,2), axis=1)
    sum2 = np.sum(np.power(X,2), axis=1)
    sum3 = 2*np.dot(Y, X.T)
    dists = sum1.reshape(-1,1) + sum2
    dist_mat = dists - sum3
    return dist_mat.T

def compute_one_vs_all_squared_distance(X, q):
    """
    input:
    X - indexed structure
    q - query point

    output:
    dist_vec = distance vectors
    closest_match = closest match index
    """
    difference_vectors = X - q
    square_of_difference_vectors = np.power(difference_vectors, 2)
    dist_vec = square_of_difference_vectors.sum(axis=1)

    return dist_vec, np.where(dist_vec == dist_vec.min())
