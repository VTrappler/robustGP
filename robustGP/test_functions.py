# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np


"""
This is a python module with some test function, to declutter other scripts

https://www.sfu.ca/~ssurjano/index.html
"""

# 2 Dimensional functions ----------------------------------------------------------------------


def branin_2d(X, damp=1.0, switch=False):
    """Scaled branin function:
    global minimizers are
    [0.124, 0.818], [0.54277, 0.1513], [0.96133, 0.16466]
    """
    X = np.atleast_2d(X)
    y, x = X[:, 1], X[:, 0]
    if switch:
        x, y = y, x
    x2 = 15 * y
    x1 = 15 * x - 5
    quad = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + (5 / np.pi) * x1 - 6) ** 2
    cosi = (10 - (10 / np.pi * 8)) * np.cos(x1) - 44.81
    return (quad + cosi) / (51.95 * damp) + 2.0
    # Xmin1 = [0.124, 0.818]
    # Xmin2 = [0.51277, 0.1513]
    # Xmin3 = [0.96133, 0.16466]


def two_valleys(X, sigma=1, rotation_angle=0):
    X = np.atleast_2d(X) * 2 - 1

    X[:, 0] = np.cos(rotation_angle) * X[:, 0] - np.sin(rotation_angle) * X[:, 1]
    X[:, 1] = np.sin(rotation_angle) * X[:, 0] + np.cos(rotation_angle) * X[:, 1]
    X = (X + 1) / 2
    k = X[:, 0] * 6 - 3
    u = X[:, 1]
    return (
        -u * np.exp(-((k - 1) ** 2) / sigma ** 2)
        - (1 - u) * 1.01 * np.exp(-((k + 1) ** 2) / sigma ** 2)
        + np.exp(-(k ** 2) / sigma ** 2)
        + 1 / (sigma ** 2)
    )


def gaussian_peaks(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0] * 5, X[:, 1] * 5
    return (
        0.8 * np.exp(-(((x) ** 2 + (y) ** 2) / 3))
        + 1.2 * np.exp(-(((y - 2.5) ** 2) + (x - 2.0) ** 2) / 1)
        + np.exp(-((x - 0.5) ** 2) / 3 - (y - 4) ** 2 / 2)
        + 0.8 * np.exp(-((x - 5) ** 2) / 4 - (y) ** 2 / 4)
        + np.exp(-((x - 5) ** 2) / 4 - (y - 5) ** 2 / 4)
        + (1 / (1 + x + y)) / 25
    )  # + 50 * np.exp((-(y - 2.5)**2 + -(x - 5)**2) / 2)


function_2d = lambda X: two_valleys(X, 1)  # , np.pi / 4)


# 3 Dimensional functions ----------------------------------------------------------------------
def hartmann_3d(X, positive=True):
    """3D function with unique global minimum on [0, 1]^3

    :param X: input array of dimension (N, 3)
    :param positive: adjust function to ensure positivity of output
    :returns: array of dimension N

    """

    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    Amatrix = np.array(
        [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ]
    )
    Pmatrix = 1e-4 * np.array(
        [
            [3689.0, 1170.0, 2673.0],
            [4699.0, 4387.0, 7470.0],
            [1091.0, 8732.0, 5547.0],
            [381.0, 5743.0, 8828.0],
        ]
    )
    out = np.empty(len(X))
    for i, x in enumerate(X):
        out[i] = -alpha.dot(np.exp(-np.sum(Amatrix * (x - Pmatrix) ** 2, 1)))
    if positive:
        out = out + 4
    return out


# 4 Dimensional functions ----------------------------------------------------------------------
def hartmann_4d(X, positive=True):
    """4d multimodal function

    :param X: input array of dimension (N, 4)
    :param positive: adjust function to ensure positivity of output
    :returns: array of dimension N

    """
    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    Amatrix = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    Pmatrix = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    out = np.empty(len(X))
    for i, x in enumerate(X):
        out[i] = -alpha.dot(np.exp(-np.sum(Amatrix * (x - Pmatrix) ** 2, 1)))
    if positive:
        out = out + 3
    return out


# 4 Dimensional functions ----------------------------------------------------------------------
def hartmann_6d(X, positive=True):
    """6d function, 6 local minima

    :param X: input array of dimension (N, 6)
    :param positive: adjust function to ensure positivity of output
    :returns: array of dimension N

    """
    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    Amatrix = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    Pmatrix = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    out = np.empty(len(X))
    for i, x in enumerate(X):
        out[i] = -alpha.dot(np.exp(-np.sum(Amatrix * (x - Pmatrix) ** 2, 1)))
    if positive:
        out = out + 4
    return out


# N Dimensional functions ----------------------------------------------------------------------
def rosenbrock_general(X):
    X = np.atleast_2d(X)
    X = X * 15 - 5
    return np.sum(
        100.0 * (X[:, 1:] - X[:, :-1] ** 2.0) ** 2.0 + (1 - X[:, :-1]) ** 2.0, 1
    )
