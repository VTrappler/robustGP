#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
import robustGP.tools as tools
from robustGP.gptools import add_points_to_design
from robustGP.test_functions import branin_2d, rosenbrock_general
import pyDOE
from functools import partial


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

    def augmented_GP(self, pt, z, optimize_cov=False):
        return add_points_to_design(self.gp, pt, z, optimize_cov)

    def set_enrichment(self, EnrichmentStrategy):
        self.enrichment = EnrichmentStrategy

    def get_global_minimiser(self, nrestart=100):
        return opt.optimize_with_restart(
            lambda X: self.gp.predict(np.atleast_2d(X)), self.bounds
        )[2]

    def get_best_so_far(self):
        return np.min(self.gp.y_train_)

    def step(self):
        # try:
        # Acquisition step
        pts, _, _ = self.enrichment.run(self)
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

    def set_idxU(self, idxU):
        self.idxU = idxU

    def create_input(self, X2):
        return partial(tools.construct_input, X2=X2, idx2=self.idxU, product=True)

    def separate_input(self, X):
        idxK = tools.get_other_indices(self.idxU, X.shape[1])
        return X[:, idxK], X[:, self.idxU]

    def get_conditional_minimiser(self, u):
        set_input = self.create_input(u)
        return opt.optimize_with_restart(
            lambda X: self.gp.predict(set_input(np.atleast_2d(X).T)),
            self.bounds[self.idxU],
        )[2]

    def get_predictor_conditional(self, u):
        set_input = self.create_input(u)
        return lambda X1, **kwargs: self.gp.predict(
            set_input(np.atleast_2d(X1).T), **kwargs
        )


if __name__ == "__main__":
    global opt1

    def callback_IMSE(admethod, i):
        global opt1
        global opt2
        ax1 = plt.subplot(2, 2, 1)
        ax1.contourf(xmg, ymg, admethod.predict(XY).reshape(20, 20))
        ax1.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
        ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
        ax1.set_aspect("equal")
        ax2 = plt.subplot(2, 2, 2)
        ax2.contourf(
            xmg, ymg, -(admethod.enrichment.criterion(admethod, XY).reshape(20, 20))
        )
        ax2.set_aspect("equal")
        opt1.append(ac.prediction_variance(admethod, int_points).mean())
        ax3 = plt.subplot(2, 2, (3, 4))
        ax3.plot(opt1)
        ax3.set_yscale("log")
        plt.savefig(f"/home/victor/robustGP/robustGP/dump/im{i:02d}.png")
        plt.close()

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
    (XY, (xmg, ymg)) = tools.pairify((x, y))

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
