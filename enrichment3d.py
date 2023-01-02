#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle

from matplotlib import cm
from functools import partial
import pyDOE
from sklearn.gaussian_process.kernels import Matern
import cma
import os
import csv

# Imports: robustGP
from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import hartmann_3d, branin_2d
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from scipy.stats import qmc

from adaptive_article import initialize_branin, fname

NDIM = 3


def initialize_function(function, NDIM, idxU=[2], name=None, initial_design=None):
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
    ada_strat = AdaptiveStrategy(bounds, function, name)
    ada_strat.fit_gp(
        initial_design,
        ada_strat.evaluate_function(initial_design),
        Matern(np.ones(NDIM)),
        n_restarts_optimizer=50,
    )
    ada_strat.set_idxU(idxU, ndim=NDIM)
    ada_strat.save_design(last=False)
    return ada_strat


bounds = np.asarray([(0, 1)] * NDIM)
# For plots
npts = 2 ** 3
x, y, z = np.linspace(0, 1, npts), np.linspace(0, 1, npts), np.linspace(0, 1, npts)
(XYZ, (xmg, ymg, zmg)) = tools.pairify((x, y, z))
XYZ = np.concatenate(
    [XYZ, pyDOE.lhs(3, samples=88, criterion="maximin", iterations=50)]
)
xl, yl = np.linspace(0, 1, 2 ** 6), np.linspace(0, 1, 2 ** 6)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
Niter = 50

hartmann = initialize_function(hartmann_3d, 3, idxU=[2], name="test")
branin = initialize_function(branin_2d, 2, idxU=[1])

log_folder = "/home/victor/robustGP/logs/"


def callback(arg, i, freq_log=2, filename=None):
    arg.save_design(last=True)
    if i % freq_log == 0:
        mdel, sdel = arg.predict_GPdelta(XYZ, alpha=2)
        m, s = arg.predict(XYZ, return_std=True)
        to_write = (np.sum(s ** 2), np.sum(sdel ** 2))
    else:
        to_write = np.nan, np.nan
    with open(
        os.path.join(f"{log_folder}", f"{arg.name}_log.txt"),
        "a+",
    ) as log_file:
        log_file.write(f"{i}, {to_write[0]}, {to_write[1]}\n")
    return to_write


def maxvar_exp(Niter, name):
    hartmann_maxvar = initialize_function(hartmann_3d, 3, idxU=[2], name=name)
    hartmann_maxvar.save_design(name, last=False)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    maximum_variance = enrich.OneStepEnrichment(bounds)
    maximum_variance.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def variance(arg, X):
        return arg.predict(X, return_std=True)[1] ** 2

    maximum_variance.set_criterion(variance, maxi=True)
    hartmann_maxvar.set_enrichment(maximum_variance)
    run_diag = hartmann_maxvar.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
    maxvar_dict = {
        "model": hartmann_maxvar,
        "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
    }
    return maxvar_dict


def monte_carlo_exp(Niter, name):
    hartmann_MC = initialize_function(hartmann_3d, 3, idxU=[2], name=name)

    montecarlo_enrich = enrich.MonteCarloEnrich(dim=3, bounds=bounds, sampler=None)
    hartmann_MC.set_enrichment(montecarlo_enrich)

    run_diag = hartmann_MC.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_MC, imse_del_MC = list(zip(*run_diag))
    MC_dict = {
        "model": hartmann_MC,
        "logs": {"imse": imse_MC, "imse_del": imse_del_MC},
    }
    return MC_dict


def augmented_IMSE_delta_exp(Niter, name):
    hartmann_aIMSE_delta = initialize_function(hartmann_3d, 3, idxU=[2], name=name)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    aIMSE_delta = enrich.OneStepEnrichment(bounds)
    aIMSE_delta.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def augmented_IMSE_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
        if callable(integration_points):
            int_points = integration_points()
        else:
            int_points = integration_points

        def function_(arg):
            m, va = arg.predict_GPdelta(int_points, alpha=alpha, beta=beta)
            return va

        return ac.augmented_design(arg, X, scenarios, function_, {})

    aIMSE_delta.set_criterion(
        augmented_IMSE_Delta,
        maxi=False,
        scenarios=None,
        integration_points=lambda: pyDOE.lhs(
            3, 100, criterion="maximin", iterations=50
        ),
        alpha=2.0,
        beta=0.0,
    )  #

    hartmann_aIMSE_delta.set_enrichment(aIMSE_delta)
    run_diag = hartmann_aIMSE_delta.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_aIMSE_delta, imse_del_aIMSE_delta = list(zip(*run_diag))
    aIMSE_delta_dict = {
        "model": hartmann_aIMSE_delta,
        "logs": {"imse": imse_aIMSE_delta, "imse_del": imse_del_aIMSE_delta},
    }
    return aIMSE_delta_dict


def add_logs_on_axes(result_dictionary, exp_name, col, axs, lab=None, kwargs={}):
    if lab is None:
        lab = exp_name
    imse = np.array(result_dictionary[exp_name]["logs"]["imse"])
    imse = imse[~np.isnan(imse)]
    imse_del = np.array(result_dictionary[exp_name]["logs"]["imse_del"])
    imse_del = imse_del[~np.isnan(imse_del)]
    axs[0].plot(imse, ".-", color=col, label=lab, **kwargs)
    axs[1].plot(imse_del, ".-", color=col, label=lab, **kwargs)
    return axs


## Make experiments
result_dictionary = {}

# Monte Carlo
for i in range(10):
    result_dictionary[f"MC_{i}"] = monte_carlo_exp(Niter, f"MC_{i}")
# Maximum of variance
result_dictionary["maxvar"] = maxvar_exp(Niter)
## Show results
fig, axs = plt.subplots(ncols=2)
axs = add_logs_on_axes(result_dictionary, "maxvar", "r", axs)
for i in range(10):
    exp = "MC_" + str(i)
    if i == 0:
        lab = ""
    else:
        lab = ""
    axs = add_logs_on_axes(result_dictionary, exp, "k", axs, lab, {"alpha": 0.2})
# axs = add_logs_on_axes(result_dictionary, "halton", "g", axs)  #
axs = add_logs_on_axes(
    result_dictionary, "aIMSE_delta", "m", axs, lab=r"$\text{aIMSE}_{\Delta}$"
)
# axs = add_logs_on_axes(result_dictionary, "aIMSE", "b", axs, lab=r"$\text{aIMSE}_Z$")

axs[0].set_title(r"$\text{IMSE}_Z$")
axs[1].set_title(r"$\text{IMSE}_\Delta$")
for ax in axs:
    ax.set_yscale("log")
    ax.legend()
plt.show()


# Maximum augmented aIMSE delta
result_dictionary["aIMSE_delta"] = augmented_IMSE_delta_exp(Niter)
