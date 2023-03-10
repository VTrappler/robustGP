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
from robustGP.test_functions import branin_2d
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from scipy.stats import qmc

from adaptive_article import fname

NDIM = 2
log_folder = os.path.join(os.sep, "home", "logs", "branin")


def initialize_function(
    function, NDIM, idxU=[1], name=None, initial_design=None, save=True
):
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
x, y = np.linspace(0, 1, npts), np.linspace(0, 1, npts)
(XY, (xmg, ymg)) = tools.pairify((x, y))
XY = np.concatenate([XY, pyDOE.lhs(2, samples=88, criterion="maximin", iterations=50)])
xl, yl = np.linspace(0, 1, 2**6), np.linspace(0, 1, 2**6)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
Niter = 50

# branin = initialize_function(branin_3d, 3, idxU=[2], name="test")
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
        mdel, sdel = arg.predict_GPdelta(XY, alpha=2)
        m, s = arg.predict(XY, return_std=True)
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
    """Make a maxvar experiment

    :param Niter: Number of iterations
    :type Niter: int
    :param name: Name of experiment
    :type name: str
    :return: Dictionary of logs and results
    :rtype: Dict
    """
    branin_maxvar = initialize_function(branin_2d, 2, idxU=[1], name=name)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    maximum_variance = enrich.OneStepEnrichment(bounds)
    maximum_variance.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def variance(arg, X):
        return arg.predict(X, return_std=True)[1] ** 2

    maximum_variance.set_criterion(variance, maxi=True)
    branin_maxvar.set_enrichment(maximum_variance)
    run_diag = branin_maxvar.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
    maxvar_dict = {
        "model": branin_maxvar,
        "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
    }
    return maxvar_dict


def maxvar_delta_exp(Niter, name):
    branin_maxvar_delta = initialize_function(branin_2d, 2, idxU=[1], name=name)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    maximum_variance_delta = enrich.OneStepEnrichment(bounds)
    maximum_variance_delta.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def variance(arg, X):
        _, va = arg.predict_GPdelta(X, alpha=2.0)
        return va

    maximum_variance_delta.set_criterion(variance, maxi=True)
    branin_maxvar_delta.set_enrichment(maximum_variance_delta)
    run_diag = branin_maxvar_delta.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
    maxvar_dict = {
        "model": branin_maxvar_delta,
        "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
    }
    return maxvar_dict


def maxvar_delta_adj_exp(Niter, name):
    branin_maxvar_delta = initialize_function(branin_2d, 2, idxU=[1], name=name)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    maximum_variance_delta = enrich.OneStepEnrichment(bounds)
    maximum_variance_delta.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def variance(arg, X):
        _, va = arg.predict_GPdelta(X, alpha=2.0)
        return va

    maximum_variance_delta.set_criterion(variance, maxi=True)
    branin_maxvar_delta.set_enrichment(maximum_variance_delta)
    run_diag = branin_maxvar_delta.run(
        Niter=Niter,
        callback=partial(callback, filename=fname(name)),
        post_treatment=lambda x: branin_maxvar_delta.compare_and_adjust(x, alpha=2.0),
    )
    imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
    maxvar_dict = {
        "model": branin_maxvar_delta,
        "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
    }
    return maxvar_dict


def monte_carlo_exp(Niter, name):
    branin_MC = initialize_function(branin_2d, 2, idxU=[1], name=name)

    montecarlo_enrich = enrich.MonteCarloEnrich(dim=2, bounds=bounds, sampler=None)
    branin_MC.set_enrichment(montecarlo_enrich)

    run_diag = branin_MC.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_MC, imse_del_MC = list(zip(*run_diag))
    MC_dict = {
        "model": branin_MC,
        "logs": {"imse": imse_MC, "imse_del": imse_del_MC},
    }
    return MC_dict


def augmented_IMSE_delta_exp(Niter, name):
    branin_aIMSE_delta = initialize_function(branin_2d, 2, idxU=[1], name=name)
    aIMSE_delta = enrich.OneStepEnrichment(bounds)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
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
            2, 100, criterion="maximin", iterations=50
        ),
        alpha=2.0,
        beta=0.0,
    )  #

    branin_aIMSE_delta.set_enrichment(aIMSE_delta)
    run_diag = branin_aIMSE_delta.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_aIMSE_delta, imse_del_aIMSE_delta = list(zip(*run_diag))
    aIMSE_delta_dict = {
        "model": branin_aIMSE_delta,
        "logs": {"imse": imse_aIMSE_delta, "imse_del": imse_del_aIMSE_delta},
    }
    return aIMSE_delta_dict


def augmented_IVPC_delta_exp(Niter, name):
    branin_aIVPC_delta = initialize_function(branin_2d, 2, idxU=[1], name=name)
    aIVPC_delta = enrich.OneStepEnrichment(bounds)
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    aIVPC_delta.set_optim(
        cma.fmin2,
        **{"x0": np.array(0.5 * np.ones(NDIM)), "sigma0": 0.5, "options": opts},
    )

    def augmented_IVPC_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
        if callable(integration_points):
            int_points = integration_points()
        else:
            int_points = integration_points

        def function_(arg):
            m, va = arg.predict_GPdelta(int_points, alpha=alpha, beta=beta)
            s = np.sqrt(va)
            return ac.variance_probability_coverage((m, s), None, 0)

        return ac.augmented_design(arg, X, scenarios, function_, {})

    aIVPC_delta.set_criterion(
        augmented_IVPC_Delta,
        maxi=False,
        scenarios=None,
        integration_points=lambda: pyDOE.lhs(2, 10, criterion="maximin", iterations=50),
        alpha=2.0,
        beta=0.0,
    )  #

    branin_aIVPC_delta.set_enrichment(aIVPC_delta)
    run_diag = branin_aIVPC_delta.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_aIVPC_delta, imse_del_aIVPC_delta = list(zip(*run_diag))
    aIVPC_delta_dict = {
        "model": branin_aIVPC_delta,
        "logs": {"imse": imse_aIVPC_delta, "imse_del": imse_del_aIVPC_delta},
    }
    return aIVPC_delta_dict


def augmented_IMSE_exp(Niter, name):
    branin_aIMSE = initialize_function(branin_2d, 2, idxU=[1], name=name)
    aIMSE = enrich.OneStepEnrichment(bounds)
    # aIMSE.set_optim(
    #     cma.fmin2,
    #     **cma_options,
    # )
    aIMSE.set_optim(None)

    def augmented_IMSE(arg, X, scenarios, integration_points, alpha, beta=0):
        if callable(integration_points):
            int_points = integration_points()
        else:
            int_points = integration_points

        def function_(arg):
            m, s = arg.predict(int_points, return_std=True)
            return s**2

        return ac.augmented_design(arg, X, scenarios, function_, {})

    aIMSE.set_criterion(
        augmented_IMSE,
        maxi=False,
        scenarios=None,
        integration_points=lambda: pyDOE.lhs(
            2, 100, criterion="maximin", iterations=50
        ),
        alpha=2.0,
        beta=0.0,
    )  #

    branin_aIMSE.set_enrichment(aIMSE)
    run_diag = branin_aIMSE.run(
        Niter=Niter, callback=partial(callback, filename=fname(name))
    )
    imse_aIMSE, imse_del_aIMSE = list(zip(*run_diag))
    aIMSE_dict = {
        "model": branin_aIMSE,
        "logs": {"imse": imse_aIMSE, "imse_del": imse_del_aIMSE},
    }
    return aIMSE_dict


## Make experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make SUR experiment with branin 2D")
    parser.add_argument(
        "experience",
        type=str,
        choices=[
            "MC",
            "maxvar",
            "maxvar_Delta",
            "maxvar_Delta_adj",
            "aIMSE_Delta",
            "aIMSE",
        ],
        help="Type of experiment to run",
    )
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
        if (parsed_args.reps > 1) or (parsed_args.offset > 0):
            filename = name + f"_{i+parsed_args.offset}"
        else:
            filename = name

        exp_dictionary = {
            "MC": monte_carlo_exp,
            "maxvar": maxvar_exp,
            "maxvar_Delta": maxvar_delta_exp,
            "maxvar_Delta_adj": maxvar_delta_adj_exp,
            "aIMSE_Delta": augmented_IMSE_delta_exp,
            "aIMSE": augmented_IMSE_exp,
            "aIVPC_Delta": augmented_IVPC_delta_exp,
        }

        # Run the experiment
        exp_dictionary[exp](parsed_args.Niter, filename)

        # if exp == "MC":
        #     monte_carlo_exp(parsed_args.Niter, filename)
        # elif exp == "maxvar":
        #     maxvar_exp(parsed_args.Niter, filename)
        # elif exp == "maxvar_Delta":
        #     maxvar_delta_exp(parsed_args.Niter, filename)
        # elif exp == "maxvar_Delta_adj":
        #     maxvar_delta_adj_exp(parsed_args.Niter, filename)
        # elif exp == "aIMSE_Delta":
        #     augmented_IMSE_delta_exp(parsed_args.Niter, filename)
        # elif exp == "aIMSE":
        #     augmented_IMSE_exp(parsed_args.Niter, filename)
