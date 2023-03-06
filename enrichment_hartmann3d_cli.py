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
import argparse

# Imports: robustGP
from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import hartmann_3d, branin_2d
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from scipy.stats import qmc

# from adaptive_article import initialize_branin, fname

NDIM = 3
log_folder = os.path.join(os.getcwd(), "logs", "hartmann")


def initialize_function(
    function, NDIM, idxU=[2], name=None, initial_design=None, save=True
):
    """
    Create new instance of AdaptiveStrategy of the Hartmann 3d function
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
    ada_strat = AdaptiveStrategy(bounds, function, name, log_folder=log_folder)
    ada_strat.fit_gp(
        initial_design,
        ada_strat.evaluate_function(initial_design),
        Matern(np.ones(NDIM)),
        n_restarts_optimizer=50,
    )
    ada_strat.set_idxU(idxU, ndim=NDIM)
    if save:
        ada_strat.save_design(last=False)
    return ada_strat


bounds = np.asarray([(0, 1)] * NDIM)
# For plots
npts = 2**3
x, y, z = np.linspace(0, 1, npts), np.linspace(0, 1, npts), np.linspace(0, 1, npts)
(XYZ, (xmg, ymg, zmg)) = tools.pairify((x, y, z))
XYZ = np.concatenate(
    [XYZ, pyDOE.lhs(3, samples=88, criterion="maximin", iterations=50)]
)
xl, yl = np.linspace(0, 1, 2**6), np.linspace(0, 1, 2**6)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
Niter = 50

# hartmann = initialize_function(hartmann_3d, 3, idxU=[2], name="test")
# branin = initialize_function(branin_2d, 2, idxU=[1])

opts = cma.CMAOptions()
opts["bounds"] = list(zip(*bounds))
opts["maxfevals"] = 50
opts["verbose"] = -9
cma_options = {"x0": np.array(0.5 * np.ones(NDIM)), "sigma0": 0.5, "options": opts}


def callback(arg, i, freq_log=2, filename=None):
    arg.save_design(last=True)
    if i == 0:
        write_str = "w"
    else:
        write_str = "a+"
    if i % freq_log == 0:
        mdel, sdel = arg.predict_GPdelta(XYZ, alpha=2)
        m, s = arg.predict(XYZ, return_std=True)
        to_write = (np.sum(s**2), np.sum(sdel**2))
    else:
        to_write = np.nan, np.nan
    with open(
        os.path.join(f"{log_folder}", f"{arg.name}_log.txt"),
        write_str,
    ) as log_file:
        log_file.write(f"{i}, {to_write[0]}, {to_write[1]}\n")
    return to_write


def maxvar_exp(Niter, name):
    hartmann_maxvar = initialize_function(hartmann_3d, 3, idxU=[2], name=name)
    maximum_variance = enrich.OneStepEnrichment(bounds)
    maximum_variance.set_optim(cma.fmin2, **cma_options)

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


def augmented_IMSE_exp(Niter, name):
    hartmann_aIMSE = initialize_function(hartmann_3d, 3, idxU=[2], name=name)
    aIMSE = enrich.OneStepEnrichment(bounds)
    aIMSE.set_optim(
        cma.fmin2,
        **cma_options,
    )

    def augmented_IMSE(arg, X, scenarios, integration_points, alpha, beta=0):
        if callable(integration_points):
            int_points = integration_points()
        else:
            int_points = integration_points

        def function_(arg):
            m, va = arg.predict_GPdelta(int_points, alpha=alpha, beta=beta)
            return va

        return ac.augmented_design(arg, X, scenarios, function_, {})

    aIMSE.set_criterion(
        augmented_IMSE,
        maxi=False,
        scenarios=None,
        integration_points=lambda: pyDOE.lhs(
            3, 100, criterion="maximin", iterations=50
        ),
        alpha=2.0,
        beta=0.0,
    )  #

    hartmann_aIMSE.set_enrichment(aIMSE)
    run_diag = hartmann_aIMSE.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_aIMSE, imse_del_aIMSE = list(zip(*run_diag))
    aIMSE_dict = {
        "model": hartmann_aIMSE,
        "logs": {"imse": imse_aIMSE, "imse_del": imse_del_aIMSE},
    }
    return aIMSE_dict


def augmented_IMSE_delta_exp(Niter, name):
    hartmann_aIMSE_delta = initialize_function(hartmann_3d, 3, idxU=[2], name=name)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    aIMSE_delta = enrich.OneStepEnrichment(bounds)
    aIMSE_delta.set_optim(
        cma.fmin2,
        **{"x0": np.array(0.5 * np.ones(NDIM)), "sigma0": 0.5, "options": opts},
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


## Make experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make SUR experiment")
    parser.add_argument("experience", type=str, help="Type of experiment to run")
    parser.add_argument("Niter", type=int, help="Number of iterations")
    parser.add_argument("--reps", default=1, type=int, help="number of replications")
    parser.add_argument(
        "--offset", default=0, type=int, help="offset for experiment number"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Name to use for experiment"
    )
    parsed_args = parser.parse_args()
    exp = parsed_args.experience
    if parsed_args.name is None:
        name = exp
    else:
        name = parsed_args.name
    print(f"Saving logs in {log_folder}")
    for i in range(parsed_args.reps):
        if parsed_args.reps > 1:
            filename = name + f"_{i+parsed_args.offset}"
        else:
            filename = name
        if exp == "MC":
            monte_carlo_exp(parsed_args.Niter, filename)
        elif exp == "maxvar":
            maxvar_exp(parsed_args.Niter, filename)
        elif exp == "aIMSE":
            augmented_IMSE_delta_exp(parsed_args.Niter, filename)
