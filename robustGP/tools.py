#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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

