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
import tqdm

# from adaptive_article import initialize_branin, fname

NDIM = 3
log_folder = os.path.join(os.getcwd(), "logs")


def initialize_function(
    function, NDIM, idxU=[2], name=None, initial_design=None, save=True
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
    ada_strat = AdaptiveStrategy(bounds, function, name)
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


def probability_regret(strat, nx=50, ny=50, nu=50, alpha=2.0, truth=False):
    if truth:
        get_cond_mini = lambda u: strat.get_conditional_minimiser_true(u)
        get_fun = lambda xy_, u: strat.function(np.concatenate([xy_, np.array([u])]))
    else:
        get_cond_mini = lambda u: strat.get_conditional_minimiser(u)
        get_fun = lambda xy_, u: strat.gp.predict(
            np.concatenate([xy_, np.array([u])]).reshape(-1, 3)
        )[0]
    optim_gp = {"theta_star": np.empty((2, nu)), "J_star": np.empty((nu,))}
    Delta = np.empty((nx * ny, nu))
    for i, u in tqdm.tqdm(list(enumerate(np.linspace(0, 1, nu)))):
        opt = get_cond_mini(u)
        Jstar = opt.fun
        optim_gp["theta_star"][:, i] = opt.x
        optim_gp["J_star"][i] = Jstar
        for j, xy_ in enumerate(
            tools.pairify([np.linspace(0, 1, nx), np.linspace(0, 1, ny)])[0]
        ):
            J = get_fun(xy_, u)
            Delta[j, i] = J - alpha * Jstar
    return Delta.reshape(nx, ny, nu), optim_gp


def error_probability_regret(design_file, freq=10):
    design = np.genfromtxt(log_folder + "/" + design_file + "_design.txt")
    sur_strat = initialize_function(
        hartmann_3d, NDIM, initial_design=design[:20, :3], save=False
    )
    Delta_truth, optim_true = probability_regret(
        sur_strat, nx=50, ny=50, nu=100, alpha=2.0, truth=True
    )
    prob_Delta_leq_0 = []
    norm_Jstar_iters = []
    norm_theta_star_iters = []

    for npoints in (pbar := tqdm.trange(15, 115, freq)):
        sur_strat = initialize_function(
            hartmann_3d, NDIM, initial_design=design[:npoints, :3], save=False
        )
        Delta_gp, optim_gp = probability_regret(
            sur_strat, nx=50, ny=50, nu=100, alpha=2.0, truth=False
        )
        error_prob_ = np.sum(
            ((Delta_gp < 0).mean(-1) - (Delta_truth < 0).mean(-1)) ** 2
        )
        prob_Delta_leq_0.append((npoints, error_prob_))
        norm_Jstar = np.mean((optim_gp["J_star"] - optim_true["J_star"]) ** 2)
        norm_Jstar_iters.append(norm_Jstar)
        norm_theta_star = np.mean(
            (optim_gp["theta_star"] - optim_true["theta_star"]) ** 2
        )
        norm_theta_star_iters.append(norm_theta_star)
        with open(
            os.path.join(f"{log_folder}", f"{design_file}_diagnostic.txt"),
            "a+",
        ) as diag_file:
            diag_file.write(
                f"{npoints}, {error_prob_}, {norm_Jstar}, {norm_theta_star}\n"
            )

    return None


def get_experiments_files(exp_name):
    files = os.listdir(log_folder)
    exp_files = sorted(
        list(
            set(
                [
                    st.replace("_log.txt", "").replace("_design.txt", "")
                    for st in files
                    if st.startswith(exp_name)
                ]
            )
        )
    )
    return exp_files


## Make experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make SUR experiment")
    parser.add_argument("experience", type=str, help="Type of experiment to run")
    parser.add_argument("logfolder", type=str, help="folder containing the logs")
    parser.add_argument("freq", type=int, help="frequency of the diagnostics")

    parsed_args = parser.parse_args()
    exp = parsed_args.experience

    list_files = get_experiments_files(exp)

    for fi in list_files:
        print(fi)
        _ = error_probability_regret(fi, parsed_args.freq)
