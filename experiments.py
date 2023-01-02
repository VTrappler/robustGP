#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from matplotlib import cm
from functools import partial
import pyDOE
from sklearn.gaussian_process.kernels import Matern
import cma

# Imports: robustGP
from robustGP.SURmodel import AdaptiveStrategy
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from scipy.stats import qmc


def initialize_function(function, NDIM, idxU=[2], initial_design=None):
    """
    Create new instance of AdaptiveStrategy of the Branin 2d function
    with LHS as initial design
    """
    bounds = np.asarray([(0, 1)] * NDIM)

    if initial_design is None:
        initial_design = 5 * NDIM
    if isinstance(initial_design, int):
        initial_design = pyDOE.lhs(
            n=NDIM, samples=initial_design, criterion="maximin", iterations=50
        )
    else:
        pass  # initial_design is already a set of samples
    ada_strat = AdaptiveStrategy(bounds, function)
    ada_strat.fit_gp(
        initial_design,
        ada_strat.evaluate_function(initial_design),
        Matern(np.ones(NDIM)),
        n_restarts_optimizer=50,
    )
    ada_strat.set_idxU(idxU, ndim=NDIM)
    return ada_strat


def callback(arg, i, freq_log=5):
    if i % freq_log == 0:
        mdel, sdel = arg.predict_GPdelta(XYZ, alpha=2)
        m, s = arg.predict(XYZ, return_std=True)
        return (np.sum(s ** 2), np.sum(sdel ** 2))
    else:
        return np.nan, np.nan


class EnrichmentExperiment:
    def __init__(self, surmodel, bounds, NDIM):
        "docstring"
        self.surmodel = surmodel
        self.NDIM = NDIM
        self.bounds = bounds

    def maximum_variance_experience(self, Niter, freq_log):
        opts = cma.CMAOptions()
        opts["bounds"] = list(zip(*self.bounds))
        opts["maxfevals"] = 50
        opts["verbose"] = -5
        maximum_variance = enrich.OneStepEnrichment(self.bounds)
        maximum_variance.set_optim(
            cma.fmin2,
            **{"x0": 0.5 * np.ones(self.NDIM), "sigma0": 0.5, "options": opts}
        )

        def variance(arg, X):
            return arg.predict(X, return_std=True)[1] ** 2

        maximum_variance.set_criterion(variance, maxi=True)
        maxvar_model = copy.deepcopy(self.surmodel)
        maxvar_model.set_enrichment(maximum_variance)
        run_diag = maxvar_model.run(
            Niter=Niter, callback=partial(callback, freq_log=freq_log)
        )
        imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
        maxvar_dict = {
            "model": maxvar_model,
            "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
        }
        return maxvar_dict


def maxvar_exp(Niter):
    hartmann_maxvar = initialize_function(hartmann_3d, 3, idxU=[2])
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -5
    maximum_variance = enrich.OneStepEnrichment(bounds)
    maximum_variance.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def variance(arg, X):
        return arg.predict(X, return_std=True)[1] ** 2

    maximum_variance.set_criterion(variance, maxi=True)
    hartmann_maxvar.set_enrichment(maximum_variance)
    run_diag = hartmann_maxvar.run(
        Niter=Niter, callback=partial(callback, filename=fname("variance"))
    )
    imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
    maxvar_dict = {
        "model": hartmann_maxvar,
        "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
    }
    return maxvar_dict
