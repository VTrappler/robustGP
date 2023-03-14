#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pyDOE
from tqdm.rich import tqdm
from tqdm import trange

from sklearn.gaussian_process.kernels import Matern

import robustGP.tools as tools

# Imports: robustGP
from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d

# from adaptive_article import initialize_branin, fname

NDIM = 2
# log_folder = os.path.join(os.getcwd(), "logs")


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
x, y = np.linspace(0, 1, npts), np.linspace(0, 1, npts)
(XY, (xmg, ymg)) = tools.pairify((x, y))
xl, yl = np.linspace(0, 1, 2**6), np.linspace(0, 1, 2**6)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
Niter = 50

# hartmann = initialize_function(hartmann_3d, 3, idxU=[2], name="test")
# branin = initialize_function(branin_2d, 2, idxU=[1])


def probability_regret(strat, nx=50, nu=100, alpha=2.0, truth=False):
    if truth:
        get_cond_mini = lambda u: strat.get_conditional_minimiser_true(u)
        get_fun = lambda x_, u: strat.function(
            np.concatenate([np.array([x_]), np.array([u])])
        )
    else:
        get_cond_mini = lambda u: strat.get_conditional_minimiser(u)
        get_fun = lambda x_, u: strat.gp.predict(
            np.concatenate([np.array([x_]), np.array([u])]).reshape(-1, 2)
        )[0]
    optim_gp = {"theta_star": np.empty((nu,)), "J_star": np.empty((nu,))}
    Delta = np.empty((nx, nu))
    for i, u in tqdm(list(enumerate(np.linspace(0, 1, nu)))):
        opt = get_cond_mini(u)
        Jstar = opt.fun
        optim_gp["theta_star"][i] = opt.x
        optim_gp["J_star"][i] = Jstar
        for j, xy_ in enumerate(np.linspace(0, 1, nx)):
            J = get_fun(xy_, u)
            Delta[j, i] = J - alpha * Jstar
    return Delta.reshape(nx, nu), optim_gp


def error_probability_regret(design_file, log_folder, nU, freq=10):
    design = np.genfromtxt(log_folder + "/" + design_file + "_design.txt")
    sur_strat = initialize_function(
        branin_2d, NDIM, initial_design=design[:15, :NDIM], save=False
    )
    Delta_truth, optim_true = probability_regret(
        sur_strat, nx=50, nu=nU, alpha=2.0, truth=True
    )
    prob_Delta_leq_0 = []
    norm_Jstar_iters = []
    norm_theta_star_iters = []

    for npoints in range(15, 115, freq):
        sur_strat = initialize_function(
            branin_2d, NDIM, initial_design=design[:npoints, :NDIM], save=False
        )
        Delta_gp, optim_gp = probability_regret(
            sur_strat, nx=50, nu=nU, alpha=2.0, truth=False
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
            "w+",
        ) as diag_file:
            diag_file.write(
                f"{npoints}, {error_prob_}, {norm_Jstar}, {norm_theta_star}\n"
            )

    return None


def get_experiments_files(exp_name, log_folder):
    files = os.listdir(log_folder)
    reg = rf"{exp_name}_[\d]+"
    exp_files = sorted(list(set(re.findall(reg, " ".join(files)))))
    return exp_files


## Make experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make diagnostics of SUR experiment")
    parser.add_argument("--experience", type=str, help="Type of experiment to run")
    parser.add_argument("--log_folder", type=str, help="folder containing the logs")
    parser.add_argument("--freq", type=int, help="frequency of the diagnostics")
    parser.add_argument("--nU", type=int, help="Number of u points")

    parsed_args = parser.parse_args()
    exp_list = parsed_args.experience

    for exp in exp_list:
        print(exp)
        list_files = get_experiments_files(exp, parsed_args.log_folder)
        for fi in list_files:
            print(fi)
            _ = error_probability_regret(
                fi, parsed_args.log_folder, parsed_args.nU, parsed_args.freq
            )
