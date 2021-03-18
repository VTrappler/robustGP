#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from robustGP.tools import pairify
from robustGP.gptools import add_points_to_design
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

    opt1 = []

    import robustGP.acquisition.acquisition as ac

    plt.subplot(2, 1, 1)
    plt.contourf(xmg, ymg, ac.probability_coverage(branin.gp, XY, 1.4).reshape(20, 20))
    plt.subplot(2, 1, 2)
    plt.contourf(
        xmg, ymg, ac.margin_indicator(branin.gp, XY, 1.4, 0.05).reshape(20, 20)
    )
    ss = sampling_from_indicator(ac.margin_indicator, branin.gp, bounds, 100, 10, T=1.4)
    plt.scatter(ss[:, 0], ss[:, 1])
    plt.show()

    from robustGP.sampling.samplers import sampling_from_indicator

    aa = clustering(10, ss)
    plt.scatter(aa[0][:, 0], aa[0][:, 1])
    plt.scatter(aa[1].cluster_centers_[:, 0], aa[1].cluster_centers_[:, 1])
    plt.show()
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
