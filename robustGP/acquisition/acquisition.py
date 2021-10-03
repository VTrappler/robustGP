# !/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import copy
import robustGP.tools as tools
import robustGP.optimisers as opt


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
    pi = scipy.stats.norm.cdf(rho)
    return pi


# --------------------------------------------------------------------------
def variance_probability_coverage(arg, X, T):
    pi = probability_coverage(arg, X, T)
    return pi * (1 - pi)


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


def augmented_design(arg, X, scenarios, function_, args):
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
            augmented_sc[i] = function_(aug_gp, **args).mean()
        augmented_meas[j] = augmented_sc.mean()
    return augmented_meas


def augmented_IVPC(arg, X, scenarios, integration_points, T):
    return augmented_design(
        arg,
        X,
        scenarios,
        variance_probability_coverage,
        {"X": integration_points, "T": T},
    )


## Profile Expected Improvement


def PEI(arg, X):
    X1, X2 = arg.separate_input(X)
    PEIvec = np.empty(len(X))
    for i, (x1, x2) in enumerate(zip(X1, X2)):
        threshold = np.max(
            [arg.get_best_so_far(), arg.get_conditional_minimiser(x2).fun[0]]
        )
        cond_pred = arg.get_predictor_conditional(x2)
        m, s = cond_pred(x1, return_std=True)
        PEIvec[i] = expected_improvement((threshold - m, s), None)
    return PEIvec


## EI-VAR: 2 step
def predict_meanGP(arg, intU):
    set_input = arg.create_input(intU)
    return lambda X1, **kwargs: [
        np.mean(out.reshape(len(np.atleast_2d(X1).T), -1), 1)
        for out in arg.gp.predict(set_input(np.atleast_2d(X1).T), **kwargs)
    ]


def predvar_meanGP(arg, intU):
    set_input = arg.create_input(intU)
    return lambda X1: [
        np.var(out.reshape(len(np.atleast_2d(X1).T), -1), 1)
        for out in arg.gp.predict(set_input(np.atleast_2d(X1).T), return_std=True)
    ][0]


def projected_EI(arg, X, intU):
    """
    Compute the EI of the projected process, with a first optimisation
    """
    proj_mean = predict_meanGP(arg, intU)
    m_th = opt.optimize_with_restart(
        lambda x: proj_mean(x, return_std=True)[0], np.atleast_2d(arg.bounds[0])
    )[2].fun

    m, c = predict_meanGP(arg, intU)(np.atleast_2d(X), return_cov=True)
    return expected_improvement((m_th - m, np.sqrt(c)), None)


def augmented_VAR(arg, X, Xnext, intU):
    m, s = arg.predict(X, return_std=True)
    var = np.empty(len(m))
    arg_ = copy.copy(arg)
    for j, m_ in enumerate(m):
        arg_.gp = arg.augmented_GP(X[j], m_)
        var[j] = -predvar_meanGP(arg_, intU)(np.atleast_2d(Xnext))
    return var
