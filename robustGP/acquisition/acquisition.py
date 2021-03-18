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
        m_, s = arg.predict(X, return_std=True)
        try:
            m = arg.y_train_.min() - m_
        except AttributeError:
            m = arg.gp.y_train_.min() - m_
        EI = s * scipy.stats.norm.pdf(m / s) + m * scipy.stats.norm.cdf(m / s)
    EI[s < 1e-14] = 0.0
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
    PI = scipy.stats.norm.cdf(m / s)
    PI[s < 1e-9] = 0.0
    return PI


# -----------------------------------------------------------------------------
def quantile_of_m(arg, X, k=0):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    return m - k * s


# -----------------------------------------------------------------------------
def prediction_variance(arg, X):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    return s ** 2


# -----------------------------------------------------------------------------
def reliability_index_rho(arg, X, T):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    return -np.abs(m - T) / s


# --------------------------------------------------------------------------
def probability_coverage(arg, X, T):
    rho = reliability_index_rho(arg, X, T)
    return scipy.stats.norm.cdf(rho)


# --------------------------------------------------------------------------
def margin_indicator(arg, X, T, eta=0.05):
    pi = probability_coverage(arg, X, T)
    return np.logical_and(pi > eta, pi < 1 - eta)


# -----------------------------------------------------------------------------
def augmented_IMSE(arg, X, scenarios, integration_points):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    if scenarios is None:
        scenarios = lambda mp, sp: scipy.stats.norm.ppf(
            np.linspace(0.05, 0.95, 5, endpoint=True), loc=mp, scale=sp
        )
    aIMSE = np.empty(len(m))
    for j, (m_, s_) in enumerate(zip(m, s)):
        IMSE_ = np.empty(5)
        for i, zi in enumerate(scenarios(m_, s_)):
            aug_gp = arg.augmented_GP(X[j], zi)
            IMSE_[i] = prediction_variance(aug_gp, integration_points).mean()
        aIMSE[j] = IMSE_.mean()
    return aIMSE


def augmented_design(arg, X, scenarios, function_):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(X, return_std=True)
    if scenarios is None:
        scenarios = lambda mp, sp: scipy.stats.norm.ppf(
            np.linspace(0.05, 0.95, 5, endpoint=True), loc=mp, scale=sp
        )
    augmented_meas = np.empty(len(m))
    for j, (m_, s_) in enumerate(zip(m, s)):
        augmented_sc = np.empty(5)
        for i, zi in enumerate(scenarios(m_, s_)):
            aug_gp = arg.augmented_GP(X[j], zi)
            augmented_sc[i] = function_(aug_gp).mean()
        augmented_meas[j] = augmented_sc.mean()
    return augmented_meas
