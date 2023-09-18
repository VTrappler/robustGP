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

# Imports: robustGP
from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from scipy.stats import qmc

from adaptive_article import initialize_branin, fname, callback

# # Plotting options
# plt.style.use("seaborn")
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "sans-serif",
#         "font.serif": ["Computer Modern Roman"],
#         "image.cmap": "viridis",
#         "figure.figsize": [5.748031686730317, 3.552478950810724],
#         "savefig.dpi": 400,
#     }
# )
# plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
# graphics_folder = "/home/victor/collab_article/adaptive/figures/"
# graphics_folder = "./figures/"

bounds = np.asarray([[0, 1], [0, 1]])
# For plots
x, y = np.linspace(0, 1, 2 ** 5), np.linspace(0, 1, 2 ** 5)
(XY, (xmg, ymg)) = tools.pairify((x, y))
xl, yl = np.linspace(0, 1, 2 ** 6), np.linspace(0, 1, 2 ** 6)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
Niter = 100

if "result_dictionary" not in locals():
    result_dictionary = {}


def callback(arg, i, filename):
    if i % 5 == 0:
        mdel, sdel = arg.predict_GPdelta(XY, alpha=2)
        n_added_points = len(arg.gp.X_train_) - 10
        m, s = arg.predict(XY, return_std=True)
        npts = int(np.sqrt(len(XY)))

        # plt.figure(figsize=(8, 6))
        # plt.subplot(2, 2, 1)
        # plt.contourf(xmg, ymg, m.reshape(npts, npts))
        # plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
        # plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
        # plt.title(r"$m_Z$")

        # plt.subplot(2, 2, 3)
        # plt.contourf(xmg, ymg, s.reshape(npts, npts))
        # plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
        # plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
        # plt.title(r"$\sigma_{Z}$")

        # plt.subplot(2, 2, 2)
        # plt.contourf(xmg, ymg, mdel.reshape(npts, npts))
        # plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
        # plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
        # plt.title(r"$m_{\Delta}$")

        # plt.subplot(2, 2, 4)
        # plt.contourf(xmg, ymg, sdel.reshape(npts, npts))
        # plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
        # plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
        # plt.title(r"$\sigma_{\Delta}$")

        # if isinstance(filename, str):
        #     fname = filename
        # elif filename is None:
        #     fname = f"{graphics_folder}maxvar_{n_added_points}.png"
        # else:
        #     fname = filename(n_added_points)
        # plt.tight_layout()
        # plt.savefig(fname)
        # plt.close()
        return (np.sum(s ** 2), np.sum(sdel ** 2))
    else:
        return np.nan, np.nan


## Halton sequence


def halton_exp(Niter):
    halton_sampler = qmc.Halton(d=2)
    initial_design = halton_sampler.random(n=8)
    branin_halton = initialize_branin(initial_design=initial_design)

    class HaltonEnrich(enrich.InfillEnrichment):
        def __init__(self, bounds, sampler):
            super(HaltonEnrich, self).__init__(bounds)
            self.sampler = sampler

        def run(self, gp):
            return np.atleast_2d(self.sampler.random(n=1)), "Halton"

    halton_enrich = HaltonEnrich(bounds, halton_sampler)
    branin_halton.set_enrichment(halton_enrich)

    run_diag = branin_halton.run(
        Niter=Niter, callback=partial(callback, filename=fname("Halton"))
    )
    imse_halton, imse_del_halton = list(zip(*run_diag))
    halton_dict = {
        "model": branin_halton,
        "logs": {"imse": imse_halton, "imse_del": imse_del_halton},
    }
    return halton_dict


# In[ ]:


def monte_carlo_exp(Niter):
    branin_MC = initialize_branin()
    montecarlo_enrich = enrich.MonteCarloEnrich(dim=2, bounds=bounds, sampler=None)
    branin_MC.set_enrichment(montecarlo_enrich)

    run_diag = branin_MC.run(
        Niter=Niter, callback=partial(callback, filename=fname("MC"))
    )
    imse_MC, imse_del_MC = list(zip(*run_diag))
    MC_dict = {
        "model": branin_MC,
        "logs": {"imse": imse_MC, "imse_del": imse_del_MC},
    }
    return MC_dict


## Maximum of Variance


def maxvar_exp(Niter):
    branin_maxvar = initialize_branin()
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -5
    maximum_variance = enrich.OneStepEnrichment(bounds)
    maximum_variance.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def variance(arg, X):
        return arg.predict(X, return_std=True)[1] ** 2

    maximum_variance.set_criterion(variance, maxi=True)
    branin_maxvar.set_enrichment(maximum_variance)
    run_diag = branin_maxvar.run(
        Niter=Niter, callback=partial(callback, filename=fname("variance"))
    )
    imse_maxvar, imse_del_maxvar = list(zip(*run_diag))
    maxvar_dict = {
        "model": branin_maxvar,
        "logs": {"imse": imse_maxvar, "imse_del": imse_del_maxvar},
    }
    return maxvar_dict


## augmented IMSE


def augmented_IMSE_exp(Niter):
    branin_aIMSE = initialize_branin()
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    aIMSE = enrich.OneStepEnrichment(bounds)
    aIMSE.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.5, "options": opts}
    )

    def augmented_IMSE(arg, X, scenarios, integration_points):
        if callable(integration_points):
            int_points = integration_points()
        else:
            int_points = integration_points

        def function_(arg):
            m, sd = arg.predict(int_points, return_std=True)
            return sd ** 2

        return ac.augmented_design(arg, X, scenarios, function_, {})

    # integration_points = pyDOE.lhs(2, 50, criterion="maximin", iterations=50)  #
    aIMSE.set_criterion(
        augmented_IMSE,
        maxi=False,
        scenarios=None,
        integration_points=lambda: pyDOE.lhs(2, 50, criterion="maximin", iterations=50),
    )  #
    branin_aIMSE.set_enrichment(aIMSE)
    run_diag = branin_aIMSE.run(
        Niter=50, callback=partial(callback, filename=fname("aIMSE_50"))
    )
    imse_aIMSE, imse_del_aIMSE = list(zip(*run_diag))
    aIMSE_dict = {
        "model": branin_aIMSE,
        "logs": {"imse": imse_aIMSE, "imse_del": imse_del_aIMSE},
    }
    return aIMSE_dict


# aIMSE_experiment(100, 'test')

## augmented IMSE delta
def augmented_IMSE_delta_exp(Niter):
    branin_aIMSE_delta = initialize_branin()
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -9
    aIMSE_delta = enrich.OneStepEnrichment(bounds)
    aIMSE_delta.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.5, "options": opts}
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
        integration_points=lambda: pyDOE.lhs(2, 50, criterion="maximin", iterations=50),
        alpha=2.0,
        beta=0.0,
    )  #

    branin_aIMSE_delta.set_enrichment(aIMSE_delta)
    run_diag = branin_aIMSE_delta.run(
        Niter=Niter, callback=partial(callback, filename=fname("aIMSE_delta"))
    )
    imse_aIMSE_delta, imse_del_aIMSE_delta = list(zip(*run_diag))
    aIMSE_delta_dict = {
        "model": branin_aIMSE_delta,
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


result_dictionary["halton"] = halton_exp(Niter)
for i in range(20):
    result_dictionary[f"MC_{i}"] = monte_carlo_exp(Niter)
result_dictionary["maxvar"] = maxvar_exp(Niter)


# fig, axs = plt.subplots(ncols=2)
# axs = add_logs_on_axes(result_dictionary, "maxvar", "r", axs)
# for i in range(10):
#     exp = "MC_" + str(i)
#     if i == 0:
#         lab = ""
#     else:
#         lab = ""
#     axs = add_logs_on_axes(result_dictionary, exp, "k", axs, lab, {"alpha": 0.2})
# axs = add_logs_on_axes(result_dictionary, "halton", "g", axs)
# axs[0].set_title(r"$\text{IMSE}_Z$")
# axs[1].set_title(r"$\text{IMSE}_\Delta$")
# for ax in axs:
#     ax.set_yscale("log")
#     ax.legend()
# plt.show()

result_dictionary["aIMSE_delta"] = augmented_IMSE_delta_exp(Niter)
result_dictionary["aIMSE"] = augmented_IMSE_exp(Niter)


fig, axs = plt.subplots(ncols=2)
axs = add_logs_on_axes(result_dictionary, "maxvar", "r", axs)
for i in range(20):
    exp = "MC_" + str(i)
    if i == 0:
        lab = ""
    else:
        lab = ""
    axs = add_logs_on_axes(result_dictionary, exp, "k", axs, lab, {"alpha": 0.2})
axs = add_logs_on_axes(result_dictionary, "halton", "g", axs)
axs = add_logs_on_axes(
    result_dictionary, "aIMSE_delta", "m", axs, lab=r"$\text{aIMSE}_{\Delta}$"
)
axs = add_logs_on_axes(result_dictionary, "aIMSE", "b", axs, lab=r"$\text{aIMSE}_Z$")

axs[0].set_title(r"$\text{IMSE}_Z$")
axs[1].set_title(r"$\text{IMSE}_\Delta$")
for ax in axs:
    ax.set_yscale("log")
    ax.legend()
plt.show()


# plt.subplot(1, 2, 1)
# plt.plot(imse_MC, label="MC")
# plt.plot(imse_halton, label="Halton")
# plt.plot(imse_maxvar, label="maxvar")
# plt.plot(imse_aIMSE, label="aIMSE")
# plt.title(r"$\text{IMSE}_Z$")
# plt.yscale("log")

# plt.subplot(1, 2, 2)
# plt.plot(imse_del_MC, label="MC")
# plt.plot(imse_del_halton, label="Halton")
# plt.plot(imse_del_maxvar, label="maxvar")
# plt.plot(imse_del_aIMSE, label="aIMSE")
# plt.yscale("log")
# plt.title(r"$\text{IMSE}_{\Delta}$")
# plt.legend()
# plt.show()

# import pickle


# def save_gp_diag(sur_obj, diag, filename):
#     to_save_dict = {"AdaptiveStrat": sur_obj.gp, "diag": diag}
#     with open(filename, "wb") as open_file:
#         pickle.dump(to_save_dict, open_file)


aIMSE_diags = {"imse": imse_aIMSE, "imse_delta": imse_del_aIMSE}

save_gp_diag(branin_aIMSE, aIMSE_diags, "aIMSE.pkl")

with open("aIMSE.pkl", "rb") as open_file:
    gp_ = pickle.load(open_file)
