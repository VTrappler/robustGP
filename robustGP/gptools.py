#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import copy


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
