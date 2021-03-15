#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import robustGP.acquisition.acquisition as ac
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt



class SURmodel:
    def __init__(self, bounds, function):
        self.bounds = bounds
        self.function = function
        self.gp = None

    def fit_gp(self, design, response, kernel, n_restarts_optimizer):
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer
        )
        self.gp.fit(design, response)

    def predict(self, X, **kwargs):
        return self.gp.predict(X, **kwargs)


    def add_points(self, pts, optimize_cov=False):
        self.gp = add_points_to_design(self.gp, pts, self.function(pts), optimize_cov)

    def set_criterion(self, criterion, maxi=False):
        if maxi:
            self.criterion = lambda gp, X: -criterion(gp, X)
        else:
            self.criterion = lambda gp, X: criterion(gp, X)

    def set_optim(self, optimiser, **params):
        if optimiser is None:
            self.optimiser = lambda cr: ac.optimize_with_restart(cr, self.bounds, **params)
        else:
            self.optimiser = lambda cr: optimiser(cr, self.bounds, **params)

    def get_global_minimiser(self, nrestart=100):
        return self.optimiser(lambda X: self.gp.predict(np.atleast_2d(X)))[2]

    def step(self):
        try:
            # Acquisition step
            pts, fun, optim = self.optimiser(
                lambda X: self.criterion(self.gp, np.atleast_2d(X))
            )
            # Evaluation step
            self.add_points(pts, optimize_cov=True)
        except KeyboardInterrupt:
            pass

    def run(self, Niter, callback=None):
        for i in range(Niter):
            self.step()
            if callback is not None:
                plt.scatter(i, (self.get_global_minimiser().fun))


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
    X = np.vstack([gp.X_train_, pts])
    # X = X[:,np.newaxis]
    y = np.append(gp.y_train_, evals)
    if not optimize_cov:
        gpp.optimizer = None
    gpp.fit(X, y)
    return gpp


from robustGP.test_functions import rosenbrock_general
import pyDOE

NDIM = 3
initial_design = pyDOE.lhs(
    n=NDIM, samples=10 * NDIM, criterion="maximin", iterations=50
)
response = rosenbrock_general(initial_design)

# random_points = np.random.uniform(size=).reshape(3, 5)


rosen = SURmodel(np.asarray([(0, 1)] * NDIM), rosenbrock_general)
rosen.fit_gp(
    initial_design, response, Matern(np.ones(NDIM) / NDIM), n_restarts_optimizer=20
)
rosen.set_criterion(ac.expected_improvement, maxi=True)

import robustGP.optimisers as opt
rosen.set_optim(optimiser=opt.optimize_with_restart, **{"nrestart": 20})
rosen.run(Niter=100, callback=True)
plt.show()

opt.optimize_with_restart(lambda X: rosen.gp.predict(np.atleast_2d(X)), rosen.bounds, nrestart=50)
