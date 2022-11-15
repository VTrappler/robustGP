#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Tuple
import numpy as np
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
import robustGP.tools as tools
from robustGP.gptools import add_points_to_design, m_s_delta, m_s_delta_product, m_s_Xi
from robustGP.test_functions import branin_2d, rosenbrock_general
import pyDOE
from functools import partial
from tqdm import tqdm, trange
import copy


class AdaptiveStrategy:
    def __init__(self, bounds, function: Callable[[np.ndarray], np.ndarray]):
        """
        Initialise an instance of an adaptive strategy
        Parameters
        ----------
        bounds : Bounds of the joint space KxU
        function : True function to be evaluated

        Returns
        -------
        out : None

        """

        self.bounds = bounds
        self.function = function
        self.gp = None
        self.diagnostic = []

    def save_gp_diag(self, diag, filename):
        to_save_dict = {"AdaptiveStrat": self.gp, "diag": diag}
        with open(filename, "wb") as open_file:
            pickle.dump(to_save_dict, open_file)

    def fit_gp(
        self,
        design: np.ndarray,
        response: np.ndarray,
        kernel,
        n_restarts_optimizer: int,
    ) -> None:
        """Fit a GP using a specific kernel, a design of experiment and the response

        :param design: Design of experiment
        :type design: np.ndarray
        :param response: Response corresponding to the evaluation of the design
        :type response: np.ndarray
        :param kernel: Which kernel to use
        :type kernel:
        :param n_restarts_optimizer: number of random restarts of the optimizer (scipy.optimize.minimize)
        :type n_restarts_optimizer: int
        """
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer
        )
        self.gp.fit(design, response)

    def evaluate_function(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the true underlying function

        :param x: points to evaluate
        :type x: np.ndarray
        :return: evaluations
        :rtype: np.ndarray
        """
        return self.function(x)

    def predict(self, X: np.ndarray, **kwargs):
        """Predict the function using the GP

        :param X: points to evaluate
        :type X: np.ndarray
        :return: prediction of the GP (depends on the kwargs)
        :rtype: _type_
        """
        return self.gp.predict(X, **kwargs)

    def add_points(self, pts: np.ndarray, optimize_cov=True) -> None:
        """Add points to the design and evaluate them

        :param pts: points to add
        :type pts: np.ndarray
        :param optimize_cov: Should the kernel parameters be optimized again, defaults to True
        :type optimize_cov: bool, optional
        """
        self.gp = add_points_to_design(self.gp, pts, self.function(pts), optimize_cov)

    def augmented_GP(
        self, pt: np.ndarray, z: float, optimize_cov: bool = False
    ) -> GaussianProcessRegressor:
        """Augment the current GP with the point (pt, z)

        :param pt: point to add
        :type pt: np.ndarray
        :param z: eval to add
        :type z: float
        :param optimize_cov: should the kernel params be optimized again, defaults to False
        :type optimize_cov: bool, optional
        :return: augmented GP
        :rtype: GaussianProcessRegressor
        """
        self_ = copy.copy(self)
        self_.gp = add_points_to_design(self.gp, pt, z, optimize_cov)
        return self_

    def set_enrichment(self, enrichmentStrategy: enrich.Enrichment) -> None:
        """Set the enrichment strategy

        :param enrichmentStrategy: Enrichment strategy to use
        :type enrichmentStrategy: enrich.Enrichment
        """
        self.enrichment = enrichmentStrategy

    def get_global_minimiser(self, nrestart: int = 100) -> np.ndarray:
        """Compute the global minimizer of the GP prediction

        :param nrestart: number of random restarts, defaults to 100
        :type nrestart: int, optional
        :return: Global minimum
        :rtype: np.ndarray
        """
        return opt.optimize_with_restart(
            lambda X: self.gp.predict(np.atleast_2d(X)), self.bounds
        )[2]

    def get_best_so_far(self) -> float:
        """Returns the smallest value evaluated yet

        :return: minimum of the evaluated responses
        :rtype: float
        """
        return np.min(self.gp.y_train_)

    def step(self) -> None:
        """Performs the acquisition step defined as self.enrichment, and add point to the design"""
        # try:
        # Acquisition step
        with tqdm(total=3, leave=False) as pbar:
            pbar.set_description("Acquisition")
            acq = self.enrichment.run(self)
            pts = acq[0]
            pbar.update(1)
            # Evaluation step
            self.add_points(pts, optimize_cov=True)

        # except KeyboardInterrupt:

    def run(self, Niter: int, callback=None) -> List:
        """Run the enrichment strategy for Niter steps

        :param Niter: Number of points to add iteratively
        :type Niter: int
        :param callback: Function to call at each iteration, defaults to None
        :type callback: _type_, optional
        :return: _description_
        :rtype: list
        """
        run_diag = []
        for i in trange(Niter):
            with tqdm(leave=False) as pbar:
                pbar.set_description("Acquisition")
                acq = self.enrichment.run(self)
                pts = acq[0]
                pbar.update(1)
                # Evaluation step
                pbar.set_description("Evaluate and add point to design")
                self.add_points(pts, optimize_cov=True)
                pbar.update(2)
                # self.step()
                pbar.set_description("Callback")
                if callable(callback):
                    diag = callback(self, i)
                    run_diag.append(diag)
                    # opt.append(self.get_global_minimiser().fun)
                    # bestsofar.append(self.get_best_so_far())
                self.diagnostic.append(run_diag)
                pbar.update(3)
        return run_diag

    def set_idxU(self, idxU: List, ndim: int = None) -> None:
        """Set the indices corresponding to the environmental parameter

        :param idxU: List of indices
        :type idxU: list
        :param ndim: Total number of dimension, defaults to None
        :type ndim: int, optional
        """
        self.idxU = idxU
        if ndim is not None:
            self.idxK = tools.get_other_indices(self.idxU, ndim)

    def create_input(self, X2: np.ndarray) -> Callable:
        """Creates the input to pass to the function when the environmental parameter is set to X2

        :param X2: Value of the environmental parameter
        :type X2: np.ndarray
        :return: Function that constructs the input
        :rtype: Callable
        """
        return partial(tools.construct_input, X2=X2, idx2=self.idxU, product=True)

    def separate_input(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate the inputs according to their indices

        :param X: merged inputs
        :type X: np.ndarray
        :return: Tuple of control parameter, environmental parameter
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        try:
            return X[:, self.idxK], X[:, self.idxU]
        except AttributeError:
            self.idxK = tools.get_other_indices(self.idxU, X.shape[1])
            return X[:, self.idxK], X[:, self.idxU]

    def get_conditional_minimiser(self, u: np.ndarray):
        set_input = self.create_input(u)
        return opt.optimize_with_restart(
            lambda X: self.gp.predict(set_input(np.atleast_2d(X).T)),
            self.bounds[self.idxK],
            10,
        )[2]

    def get_conditional_minimiser_true(self, u: np.ndarray):
        set_input = self.create_input(u)
        return opt.optimize_with_restart(
            lambda X: self.evaluate_function(set_input(np.atleast_2d(X).T)),
            self.bounds[self.idxK],
            10,
        )[2]

    def get_predictor_conditional(self, u: np.ndarray):
        set_input = self.create_input(u)
        return lambda X1, **kwargs: self.gp.predict(
            set_input(np.atleast_2d(X1).T), **kwargs
        )

    def predict_GPdelta(
        self, X: np.ndarray, alpha: float, beta: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return m_s_delta(self, X=X, alpha=alpha, beta=beta)

    def predict_GPdelta_product(
        self, X1: np.ndarray, X2: np.ndarray, alpha: float, beta: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return m_s_delta_product(self, X1, X2, idx2=self.idxU, alpha=alpha, beta=beta)


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
