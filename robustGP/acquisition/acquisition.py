# !/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import copy


# -----------------------------------------------------------------------------
def expected_improvement(arg, X):
    """Analytical form of the EI (m centered)

    :param arg: tuple (m, s) or GaussianProcessRegressor
    :returns: EI
    """
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    with np.errstate(divide="ignore"):
        EI = s * scipy.stats.norm.pdf(m / s) + m * scipy.stats.norm.cdf(m / s)
        EI[s < 1e-9] = 0.0
    return EI


# -----------------------------------------------------------------------------
def probability_of_improvement(arg, X):
    """probability of improvement (m centered)

    :param m: mean parameter
    :param s: standard deviation

    """
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    with np.errstate(divide="ignore"):
        PI = scipy.stats.norm.cdf(m / s)
        PI[s < 1e-9] = 0.0
    return PI
