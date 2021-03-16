#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from robustGP.tools import pairify
from robustGP.test_functions import branin_2d, rosenbrock_general
import pyDOE


class AdaptiveStrategy:
    def __init__(self, bounds, function):
        self.bounds = bounds
        self.function = function
        self.gp = None

    def fit_gp(self, design, response, kernel, n_restarts_optimizer):
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer
        )
        self.gp.fit(design, response)

    def evaluate_function(self, x):
        return self.function(x)

    def predict(self, X, **kwargs):
        return self.gp.predict(X, **kwargs)

    def add_points(self, pts, optimize_cov=True):
        self.gp = add_points_to_design(self.gp, pts, self.function(pts), optimize_cov)

    def set_enrichment(self, enrichment):
        self.enrichment = enrichment

    def set_criterion(self, criterion, maxi=False):
        if maxi:
            self.criterion = lambda gp, X: -criterion(gp, X)
        else:
            self.criterion = lambda gp, X: criterion(gp, X)

    def get_global_minimiser(self, nrestart=100):
        return opt.optimize_with_restart(
            lambda X: self.gp.predict(np.atleast_2d(X)), self.bounds
        )[2]

    def get_best_so_far(self):
        return np.min(self.gp.y_train_)

    def step(self):
        # try:
        # Acquisition step
        pts, _, _ = self.enrichment.run(self.gp)
        # Evaluation step
        self.add_points(pts, optimize_cov=True)
        # except KeyboardInterrupt:

    def run(self, Niter, callback=None):
        for i in range(Niter):
            print(self.gp.kernel_)
            self.step()
            if callable(callback):
                callback(self, i)
                # opt.append(self.get_global_minimiser().fun)
                # bestsofar.append(self.get_best_so_far())


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


NDIM = 2
bounds = np.asarray([(0, 1)] * NDIM)
initial_design = pyDOE.lhs(n=NDIM, samples=5 * NDIM, criterion="maximin", iterations=50)
branin = AdaptiveStrategy(bounds, branin_2d)
branin.fit_gp(
    initial_design,
    branin.evaluate_function(initial_design),
    Matern(np.ones(NDIM)),
    n_restarts_optimizer=20,
)
EGO = enrich.OneStepEnrichment(bounds)
EGO.set_criterion(ac.expected_improvement, maxi=True)  #
EGO.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
branin.set_enrichment(EGO)

x, y = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
(XY, (xmg, ymg)) = pairify((x, y))

global opt1
global opt2
opt1 = []
opt2 = []


def callback(admethod, i):
    global opt1
    global opt2
    ax1 = plt.subplot(2, 2, 1)
    ax1.contourf(xmg, ymg, admethod.predict(XY).reshape(20, 20))
    ax1.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
    ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
    ax1.set_aspect("equal")
    ax2 = plt.subplot(2, 2, 2)
    ax2.contourf(
        xmg, ymg, -(admethod.enrichment.criterion(admethod.gp, XY).reshape(20, 20))
    )
    ax2.set_aspect("equal")

    ax3 = plt.subplot(2, 2, (3, 4))
    opt1.append(np.log((admethod.get_global_minimiser().fun) ** 2))
    opt2.append(np.log((admethod.get_best_so_far()) ** 2))
    ax3.plot(opt1)
    ax3.plot(opt2)

    plt.savefig(f"/home/victor/robustGP/robustGP/im{i}.png")
    plt.close()


branin.run(Niter=50, callback=callback)


reliability = enrich.OneStepEnrichment(bounds)
reliability.set_criterion(ac.reliability_index_rho, maxi=True, T=1.5)  #
reliability.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
branin.set_enrichment(reliability)
branin.run(Niter=50, callback=callback)


# ----------------------------------------------------------------------
##   ROSENBROCK


NDIM = 2
bounds = np.asarray([(-5, 5)] * NDIM)
initial_design = (
    pyDOE.lhs(n=NDIM, samples=10 * NDIM, criterion="maximin", iterations=50) - 0.5
) * 10
response = rosenbrock_general(initial_design)

# random_points = np.random.uniform(size=).reshape(3, 5)
x, y = np.linspace(-5, 5, 20), np.linspace(-5, 5, 20)
(XY, (xmg, ymg)) = pairify((x, y))

rosen = AdaptiveStrategy(bounds, rosenbrock_general)

rosen.fit_gp(
    initial_design,
    rosen.evaluate_function(initial_design),
    Matern(np.ones(NDIM) / NDIM),
    n_restarts_optimizer=20,
)

EGO = enrich.OneStepEnrichment(bounds)
EGO.set_criterion(ac.expected_improvement, maxi=True)
EGO.set_optim(opt.optimize_with_restart, **{"nrestart": 50})
rosen.set_enrichment(EGO)
rosen.run(Niter=100, callback=callback)
