#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import numpy as np



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



#  TODO: CMA-ES ?


# EOF ----------------------------------------------------------------------
