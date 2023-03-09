#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import robustGP.tools as tools
from typing import Callable, Optional, Union


def optimize_with_restart(criterion, bounds, nrestart=10):
    """Optimize the criterion, using random restarts within the bounds for global optimisation

    :param criterion: criterion function to be minimised
    :param bounds: bounds of the problem, of dim (n, 2)
    :param nrestart: number of random restart
    :returns: tuple of minimiser, minimum, and the whole optimisation result object

    """
    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nrestart, len(bounds)))
    bestcrit = np.inf
    for x0_ in x0:
        optim = scipy.optimize.minimize(criterion, x0=x0_, bounds=bounds)
        if optim.fun < bestcrit:
            best_optim = optim
            bestcrit = optim.fun
    return best_optim.x, best_optim.fun, best_optim


def efficient_global_optimization(
    criterion: Callable,
    bounds: np.ndarray,
    n_iter: int,
    doe: Optional[Union[np.ndarray, int]],
):
    """Optimize the criterion, using the EGO algorithm from package smt

    :param criterion: function to minimize
    :type criterion: Callable
    :param bounds: bounds for the optimization
    :type bounds: np.ndarray
    :param n_iter: number of iterations
    :type n_iter: _type_
    :param doe: initial design of experiment
    :type doe: Optional[np.ndarray]
    """
    from smt.applications import EGO
    from smt.surrogate_models import KRG  # , XSpecs
    import pyDOE

    if isinstance(doe, int):
        initial_design = tools.scale_unit(
            pyDOE.lhs(n=len(bounds), samples=doe, criterion="maximin", iterations=20),
            bounds,
        )
    else:
        initial_design = doe

    ego = EGO(
        n_iter=n_iter,
        criterion="EI",
        xdoe=initial_design,
        surrogate=KRG(print_global=False),
        xlimits=bounds,
    )
    best_x, fun, *rest = ego.optimize(fun=criterion)
    return best_x, fun, rest


#  TODO: CMA-ES ?


# EOF ----------------------------------------------------------------------
