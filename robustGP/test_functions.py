# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np


"""
This is a python module with some test function, to declutter other scripts
"""

# 2 Dimensional functions ----------------------------------------------------------------------


def branin_2d(X, damp = 1.0, switch = False):
    """ Scaled branin function:
    global minimizers are
    [0.124, 0.818], [0.54277, 0.1513], [0.96133, 0.16466]
    """
    X = np.atleast_2d(X)
    y, x = X[:, 1], X[:, 0]
    if switch:
        x, y = y, x
    x2 = 15 * y
    x1 = 15 * x - 5
    quad = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2
    cosi = (10 - (10 / np.pi * 8)) * np.cos(x1) - 44.81
    return (quad + cosi) / (51.95 * damp) + 2.0
    # Xmin1 = [0.124, 0.818]
    # Xmin2 = [0.51277, 0.1513]
    # Xmin3 = [0.96133, 0.16466]


def two_valleys(X, sigma = 1, rotation_angle = 0):
    X = np.atleast_2d(X) * 2 - 1

    X[:, 0] = np.cos(rotation_angle) * X[:, 0] - np.sin(rotation_angle) * X[:, 1]
    X[:, 1] = np.sin(rotation_angle) * X[:, 0] + np.cos(rotation_angle) * X[:, 1]
    X = (X + 1) / 2
    k = X[:, 0] * 6 - 3
    u = X[:, 1]
    return -u * np.exp(-(k - 1)**2 / sigma**2) \
        - (1 - u) * 1.01 * np.exp(-(k + 1)**2 / sigma**2) \
        + np.exp(-k**2 / sigma**2) + 1 / (sigma**2)


def gaussian_peaks(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0] * 5, X[:, 1] * 5
    return 0.8 * np.exp(-(((x)**2 + (y)**2) / 3)) \
        + 1.2 * np.exp(-(((y - 2.5)**2) + (x - 2.0)**2) / 1) \
        + np.exp(-(x - 0.5)**2 / 3 - (y - 4)**2 / 2) \
        + 0.8 * np.exp(-(x - 5)**2 / 4 - (y)**2 / 4) \
        + np.exp(-(x - 5)**2 / 4 - (y - 5)**2 / 4) \
        + (1 / (1 + x + y)) / 25  # + 50 * np.exp((-(y - 2.5)**2 + -(x - 5)**2) / 2)


function_2d = lambda X: two_valleys(X, 1)  # , np.pi / 4)


# N Dimensional functions ----------------------------------------------------------------------
def rosenbrock_general(X):
        X = np.atleast_2d(X)
        X = X * 15 - 5
        return np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2.0)**2.0 + (1 - X[:, :-1])**2.0, 1)
