#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import copy
from robustGP.tools import construct_input


def add_points_to_design(gp, pts, evals, optimize_cov=False):
    """Concatenate points to the design and their evaluation by the underlying function.

    :param gp: GaussianProcessRegressor
    :param points_to_add: Points to add to the design
    :param evaluated_points: Evaluated points of the design to be added
    :param optimize_cov: if True, a new ML estimation for the kernels parameter will be achieved
    :returns: The fitted gaussian process
    :rtype: GaussianProcessRegressor

    """
    gpp = copy.deepcopy(gp)
    X = np.vstack([gpp.X_train_, pts])
    # X = X[:,np.newaxis]
    y = np.append(gpp.y_train_, evals)
    if not optimize_cov:
        gpp.optimizer = None
    gpp.fit(X, y)
    return gpp


def gp_to_delta(arg, x, xstar, alpha=1.3, beta=0, return_var=False):
    """Compute mean and variance parameters for affine Z(x1) - alph Z(xstar) - beta

    :param arg: SURstrategy or GP
    :param x: first input
    :param xstar: second input
    :param alpha: multiplicative constant
    :param beta: additive constant
    :returns: tuple of vectors (mean and variance) of the resulting process

    """

    mul_vec = np.asarray([1, -alpha])
    if return_var:
        mean, cov = arg.predict([x, xstar], return_cov=True)
        muD, varD = mul_vec.dot(mean) - beta, mul_vec.dot(cov.dot(mul_vec.T))
        return muD, varD, (cov[0, 0], (alpha ** 2) * cov[1, 1])
    else:
        mean, cov = arg.predict([x, xstar])
        muD, varD = mul_vec.dot(mean) - beta
        return muD


def gp_to_Xi(arg, x, xstar, return_var=False):
    if return_var:
        mean, cov = arg.predict([x, xstar], return_cov=True)
        mul_vec = np.asarray([1.0 / mean[0], -1.0 / mean[1]])
        varXi = mul_vec.dot(cov.dot(mul_vec.T))
        return np.log(mean[0]) - np.log(mean[1]), varXi
    else:
        mean, cov = arg.predict([x, xstar])
        return np.log(mean[0]) - np.log(mean[1])


def m_s_delta(arg, X, alpha, beta):
    """Compute the mean and variance parameters of Delta_{alpha,beta}

    :param arg: SURstrategy
    :param X: vector of points which need to be evaluated
    :param alpha: multiplicative constant
    :param beta: additive constant
    :returns: tuple of vectors (mean and variance)

    """
    X1, X2 = arg.separate_input(X)
    mu = np.empty(len(X))
    var = np.empty(len(X))
    for i, (x1, x2) in enumerate(zip(X1, X2)):
        set_input = arg.create_input(x2)
        x1_star = arg.get_conditional_minimiser(x2).x
        mu[i], var[i], _ = gp_to_delta(
            arg,
            set_input(x1).flatten(),
            set_input(x1_star).flatten(),
            alpha=alpha,
            beta=beta,
            return_var=True,
        )
        # m, s = cond_pred(x1, return_std=True)
    return mu, var


def m_s_delta_product(arg, X1, X2, idx2=None, alpha=1.3, beta=0):
    """Compute the mean and variance parameters of Delta_{alpha,beta}

    :param arg: SURstrategy
    :param X: vector of points which need to be evaluated
    :param alpha: multiplicative constant
    :param beta: additive constant
    :returns: tuple of vectors (mean and variance)

    """
    mu = np.empty(len(X1) * len(X2))
    var = np.empty(len(X1) * len(X2))
    for i, x2 in enumerate(X2):
        set_input = arg.create_input(x2)
        x1_star = arg.get_conditional_minimiser(x2).x
        for j, x1 in enumerate(X1):
            mu[i + j * len(X1)], var[i + j * len(X1)], _ = gp_to_delta(
                arg,
                set_input(x1).flatten(),
                set_input(x1_star).flatten(),
                alpha=alpha,
                beta=beta,
                return_var=True,
            )
        # m, s = cond_pred(x1, return_std=True)
    return mu, var


def m_s_Xi(arg, X):
    """Compute the mean and variance parameters of Delta_{alpha,beta}

    :param arg: SURstrategy
    :param X: vector of points which need to be evaluated
    :param alpha: multiplicative constant
    :param beta: additive constant
    :returns: tuple of vectors (mean and variance)

    """
    X1, X2 = arg.separate_input(X)
    mu = np.empty(len(X))
    var = np.empty(len(X))
    for i, (x1, x2) in enumerate(zip(X1, X2)):
        set_input = arg.create_input(x2)
        x1_star = arg.get_conditional_minimiser(x2).x
        mu[i], var[i] = gp_to_Xi(
            arg,
            set_input(x1).flatten(),
            set_input(x1_star).flatten(),
            return_var=True,
        )
        # m, s = cond_pred(x1, return_std=True)
    return mu, var


def rm_obs_gp(gp, ninitial, n_added):
    gp_rm = copy.copy(gp)
    N = ninitial + n_added
    X_train_ = gp.X_train_[:N]
    y_train_ = gp.y_train_[:N]
    gp_rm.fit(X_train_, y_train_)
    return gp_rm
