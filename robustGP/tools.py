#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools


def pairify(vectors):
    """meshgrid generation and combinations of vectors

    :returns: tuple of "pairs" and mg
    :rtype:

    """
    mg = np.meshgrid(*vectors, indexing='ij')
    return np.stack((mg), axis=-1).reshape(-1, len(mg)), mg


def scale_unit(X, bounds):
    """scales the columns of X [0, 1] to the bounds

    :param X: input vector of format (n, k)
    :param bounds: bounds (k, 2)
    :returns: scaled vector

    """
    return X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def get_other_indices(idx, N):
    """
    """
    return list(filter(lambda i: i in range(N) and i not in idx, range(N)))


def construct_input(X1, X2, idx2=None, product=False):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dim = X1.shape[1] + X2.shape[1]
    if idx2 is None:
        idx1 = range(X1.shape[1])
        idx2 = range(X1.shape[1], dim)
    else:
        idx1 = get_other_indices(idx2, dim)

    if not product:
        X = np.empty((len(X1), dim))
        for (x1, x2, x) in zip(X1, X2, X):
            x[idx1] = x1
            x[idx2] = x2
    if product:
        X = np.empty((len(X1) * len(X2), dim))
        for x, pr in zip(X, itertools.product(X1, X2)):
            x[idx1] = pr[0]
            x[idx2] = pr[1]
    return X
