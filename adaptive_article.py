#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports: General tools
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dill

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

# Plotting options
plt.style.use("seaborn")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.serif": ["Computer Modern Roman"],
        "image.cmap": "viridis",
        "figure.figsize": [5.748031686730317, 3.552478950810724],
        "savefig.dpi": 400,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
graphics_folder = "/home/victor/collab_article/adaptive/figures/"


bounds = np.asarray([[0, 1], [0, 1]])
# For plots
x, y = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
(XY, (xmg, ymg)) = tools.pairify((x, y))
xl, yl = np.linspace(0, 1, 500), np.linspace(0, 1, 500)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))


def initialize_branin(initial_design=None):
    """
    Create new instance of AdaptiveStrategy of the Branin 2d function
    with LHS as initial design
    """
    NDIM = 2
    if initial_design is None:
        initial_design = 5 * NDIM
    bounds = np.asarray([(0, 1)] * NDIM)
    initial_design = pyDOE.lhs(
        n=NDIM, samples=initial_design, criterion="maximin", iterations=50
    )
    branin = AdaptiveStrategy(bounds, branin_2d)
    branin.fit_gp(
        initial_design,
        branin.evaluate_function(initial_design),
        Matern(np.ones(NDIM)),
        n_restarts_optimizer=50,
    )
    branin.set_idxU([1], ndim=2)
    return branin


fname = lambda prefix: (lambda n: f"{graphics_folder}{prefix}_{n}.png")


def callback(arg, i, filename):
    mdel, sdel = arg.predict_GPdelta(XY, alpha=2)
    n_added_points = len(arg.gp.X_train_) - 10
    m, s = arg.predict(XY, return_std=True)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.contourf(xmg, ymg, m.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title(r"$m_Z$")

    plt.subplot(2, 2, 3)
    plt.contourf(xmg, ymg, s.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title(r"$\sigma_{Z}$")

    plt.subplot(2, 2, 2)
    plt.contourf(xmg, ymg, mdel.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title(r"$m_{\Delta}$")

    plt.subplot(2, 2, 4)
    plt.contourf(xmg, ymg, sdel.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title(r"$\sigma_{\Delta}$")

    if isinstance(filename, str):
        fname = filename
    elif filename is None:
        fname = f"{graphics_folder}maxvar_{n_added_points}.png"
    else:
        fname = filename(n_added_points)
    plt.savefig(fname)
    plt.close()


## augmented IMSE
branin_aug = initialize_branin()
opts = cma.CMAOptions()
opts["bounds"] = list(zip(*bounds))
opts["maxfevals"] = 100
opts["verbose"] = -5
aIMSE_Delta = enrich.OneStepEnrichment(bounds)
aIMSE_Delta.set_optim(
    cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.3, "options": opts}
)


def augmented_IMSE_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
    def function_(arg):
        m, va = arg.predict_GPdelta(integration_points, alpha=alpha, beta=beta)
        return va

    return ac.augmented_design(arg, X, scenarios, function_, {})


integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=50)
aIMSE_Delta.set_criterion(
    augmented_IMSE_Delta,
    maxi=False,
    scenarios=None,
    integration_points=integration_points,
    alpha=2,
    beta=0,
)  #
branin_aug.set_enrichment(aIMSE_Delta)
branin_aug.run(Niter=40, callback=partial(callback, filename=fname("augIMSE")))
with open("branin_aug.dill", "wb") as dill_file:
    dill.dump(branin_aug, dill_file)


## Max variance Z

list_branin_maxZvar = []
for _ in range(5):
    _branin = initialize_branin()
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 50
    opts["verbose"] = -5
    maximum_variance = enrich.OneStepEnrichment(bounds)
    maximum_variance.set_optim(
        cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.3, "options": opts}
    )

    def variance(arg, X):
        return arg.predict(X, return_std=True)[1] ** 2

    maximum_variance.set_criterion(variance, maxi=True)
    _branin.set_enrichment(maximum_variance)
    _branin.run(Niter=100, callback=partial(callback, filename=fname("variance")))
    list_branin_maxZvar.append(_branin)

with open("branin.dill", "wb") as dill_file:
    dill.dump(branin, dill_file)


## Max variance Delta
branin_vdelta = initialize_branin()
opts = cma.CMAOptions()
opts["bounds"] = list(zip(*bounds))
opts["maxfevals"] = 150
opts["verbose"] = -5
maximum_variance_delta = enrich.OneStepEnrichment(bounds)
maximum_variance_delta.set_optim(
    cma.fmin2, **{"x0": np.array([0.5, 0.5]), "sigma0": 0.3, "options": opts}
)


def variance_delta(arg, X):
    return arg.predict_GPdelta(X, alpha=2.0)[1]


maximum_variance_delta.set_criterion(variance_delta, maxi=True)
branin_vdelta.set_enrichment(maximum_variance_delta)
branin_vdelta.run(
    Niter=50, callback=partial(callback, filename=fname("variance_delta"))
)
with open("branin_vdelta.dill", "wb") as dill_file:
    dill.dump(branin_vdelta, dill_file)


## 2step


def pro_var_coverage(mdel, sdel):
    pi = scipy.stats.norm.cdf(-mdel / sdel)
    return pi, pi * (1 - pi)


def optimiser_1D(cr):
    optim = opt.optimize_with_restart(cr, np.array([[0, 1]]), 5)
    print(f"first phase: {optim[0]}, {optim[1]}")
    return optim


def optimiser_2D_cma(cr, bounds):
    opts = cma.CMAOptions()
    opts["bounds"] = list(zip(*bounds))
    opts["maxfevals"] = 150
    opts["verbose"] = -5
    return partial(cma.fmin2, x0=np.array([0.5, 0.5]), sigma0=0.3, options=opts)(cr)


def nu_ev(i):
    return np.exp(-i / 50.0)


def lower_bound_prob(arg, X, nu=None):
    if nu is None:
        i = len(arg.gp.X_train_) - 10
        nu = nu_ev(i)
    inte = np.random.uniform(size=20)
    inp = tools.construct_input(
        np.atleast_2d(X).T, np.atleast_2d(inte).T, idx2=[1], product=True
    )
    mdel, vdel = arg.predict_GPdelta(inp, alpha=2)
    low, mid, hi = [
        (mdel + mul * nu * np.sqrt(vdel)).reshape(len(X), 20) for mul in [-1, 0, 1]
    ]
    return (low < 0).mean(1)


def augmented_slice_IMSE(gp, X, Xnext):
    integration_points = tools.construct_input(
        np.atleast_2d(Xnext).T,
        np.atleast_2d(np.linspace(0, 1, 30)).T,
        idx2=[1],
        product=True,
    )

    def function_(arg):
        m, va = arg.predict_GPdelta(integration_points, alpha=2.0)
        return va ** 2

    return ac.augmented_design(gp, X, None, function_, {})


def callback(arg, i, filename):
    mdel, sdel = arg.predict_GPdelta(XY, alpha=2)
    n_added_points = len(arg.gp.X_train_) - 10

    nu = nu_ev(i)
    low, mid, hi = [
        (mdel + mul * nu * np.sqrt(sdel)).reshape(50, 50) for mul in [-1, 0, 1]
    ]

    pro, var = pro_var_coverage(mdel, sdel)
    pro = pro.reshape(50, 50)
    var = var.reshape(50, 50)

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 2, 1)
    plt.contourf(xmg, ymg, mdel.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title(r"$m_{\Delta}$")

    plt.subplot(3, 2, 3)
    plt.contourf(xmg, ymg, sdel.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title(r"$\sigma^2_{\Delta}$")

    plt.subplot(3, 2, 2)
    plt.contourf(xmg, ymg, var)
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title("Coverage variance")

    plt.subplot(3, 2, 4)
    plt.contourf(xmg, ymg, pro)
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], ".w")
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], ".r")
    plt.title("Prob of coverage")

    plt.subplot(3, 2, 5)

    plt.plot((mid < 0).mean(1))
    plt.plot((hi < 0).mean(1))
    plt.plot((low < 0).mean(1))
    plt.axvline(((low < 0).mean(1)).argmax())
    plt.tight_layout()

    if isinstance(filename, str):
        fname = filename
    elif filename is None:
        fname = f"{graphics_folder}maxvar_{n_added_points}.png"
    else:
        fname = filename(n_added_points)
    plt.savefig(fname)
    plt.close()


branin_2step = initialize_branin()
two_steps = enrich.TwoStepEnrichment(bounds)
two_steps.set_criterion_step1(partial(lower_bound_prob, nu=1.9))
two_steps.set_criterion_step2(augmented_slice_IMSE)
two_steps.set_optim1(optimiser_1D)
two_steps.set_optim2(optimiser_2D_cma)
branin_2step.set_enrichment(two_steps)
branin_2step.run(Niter=40, callback=partial(callback, filename=fname("2step_aimse")))
with open("branin_2step.dill", "wb") as dill_file:
    dill.dump(branin_2step, dill_file)


bra_list = [
    # branin,  # max Z variance
    # branin_aug,  # min aIMSE
    # branin_vdelta,  # max Delta variance
    branin_2step,  # 2step aIMSE]
]
bra_str = [
    # "maxZvar",
    # "aIMSE",  # min aIMSE
    # "maxDvar",  # max Delta variance
    "2stepaIMSE",  # 2step aIMSE]
]


## Exploitations of results
# with open("bra_list.pkl", "wb") as f:
#     dill.dump(bra_list, f)

# with open("bra_list.pkl", "rb") as f:
#     bra_list = dill.load(f)


with open("branin_aug.dill", "rb") as dill_file:
    branin_aug = dill.load(dill_file)  # Augmented IMSE

with open("branin.dill", "rb") as dill_file:
    branin = dill.load(dill_file)  # maximum variance

with open("branin_vdelta.dill", "rb") as dill_file:
    branin_vdelta = dill.load(dill_file)  # maximum variance of Delta

with open("branin_2step.dill", "rb") as dill_file:
    branin_2step = dill.load(dill_file)  # maximum variance of Delta

strategies = [
    branin,
    branin_aug,
    branin_vdelta,
    branin_2step,
]

bra_str = [
    "maxZvar",
    "aIMSE",
    "maxDvar",
    "2stepaIMSE",
]

J_truth = branin_aug.function(XY).reshape(50, 50)
theta_star_truth = x[J_truth.argmin(0)]
Jstar_truth = J_truth.min(0)
Delta_truth = J_truth - 2.0 * Jstar_truth[np.newaxis, :]
Gamma_truth = (Delta_truth <= 0).mean(1)


def compute_error_stats(arg):
    mdel, vdel = arg.predict_GPdelta(XY, alpha=2)
    mdel, vdel = mdel.reshape(50, 50), vdel.reshape(50, 50)
    m, s = arg.predict(XY, return_std=True)
    m, s = m.reshape(50, 50), s.reshape(50, 50)
    IMSE_Z = (s ** 2).sum()
    IMSE_Delta = (vdel).sum()
    pro, var = pro_var_coverage(mdel, np.sqrt(vdel))
    pro = pro.reshape(50, 50)

    Gamma_error_pro = ((pro.mean(1) - Gamma_truth) ** 2).sum()
    Gamma_error_plug = (((mdel <= 0).mean(1) - Gamma_truth) ** 2).sum()

    max_gamma_pro = pro.mean(1).max() - Gamma_truth.max()
    max_gamma_plug = (mdel <= 0).mean(1).max() - Gamma_truth.max()
    Delta_error = ((Delta_truth - mdel) ** 2).sum()
    J_error = ((J_truth - m) ** 2).sum()
    theta_star_hat = x[m.argmin(0)]
    mstar, sstar = arg.predict(np.vstack([theta_star_hat, y]).T, return_std=True)
    Jstar_error = ((Jstar_truth - mstar) ** 2).sum()

    return (
        IMSE_Z,
        IMSE_Delta,
        Delta_error,
        J_error,
        Jstar_error,
        Gamma_error_pro,
        Gamma_error_plug,
        max_gamma_pro,
        max_gamma_plug,
    )


import tqdm
import copy

diagnostic = dict()
diagnostic_branin_simple = dict()

pbar = tqdm.tqdm(zip(strategies, bra_str))
pbar = tqdm.tqdm(zip(list_branin_maxZvar, ["1", "2", "3", "4", "5"]))

for bran, string in pbar:
    total = len(bran.gp.X_train_)
    IMSE_Z = np.empty((total))
    IMSE_Delta = np.empty((total))
    Delta_error = np.empty((total))
    J_error = np.empty((total))
    Jstar_error = np.empty((total))
    Gamma_error_pro = np.empty((total))
    Gamma_error_plug = np.empty((total))
    max_gamma_pro = np.empty((total))
    max_gamma_plug = np.empty((total))

    for i in tqdm.trange(total - 10):
        bran_tmp = copy.copy(bran)
        gp_tmp = robustGP.gptools.rm_obs_gp(bran.gp, 10, i)
        bran_tmp.gp = gp_tmp
        (
            IMSE_Z[i],
            IMSE_Delta[i],
            Delta_error[i],
            J_error[i],
            Jstar_error[i],
            Gamma_error_pro[i],
            Gamma_error_plug[i],
            max_gamma_pro[i],
            max_gamma_plug[i],
        ) = compute_error_stats(bran_tmp)
        dic_tmp = {
            "IMSE_Z": IMSE_Z,
            "IMSE_Delta": IMSE_Delta,
            "Delta_error": Delta_error,
            "J_error": J_error,
            "Jstar_error": Jstar_error,
            "Gamma_error_pro": Gamma_error_pro,
            "Gamma_error_plug": Gamma_error_plug,
            "max_gamma_pro": max_gamma_pro,
            "max_gamma_plug": max_gamma_plug,
        }
        diagnostic_branin_simple[string] = dic_tmp

keys = [
    "IMSE_Z",
    "IMSE_Delta",
    "Delta_error",
    "J_error",
    "Jstar_error",
    "Gamma_error_pro",
    "Gamma_error_plug",
    "max_gamma_pro",
    "max_gamma_plug",
]


def plot_diagnostic(keys_dict, diag_dict, col_dict):
    for i in range(9):
        plt.subplot(3, 3, 1 + i)
        for kk in keys_dict:
            abs_tmp = np.abs(diag_dict[kk][keys[i]])
            tmp = [x if (x > 1e-50 and x < 1e50) else np.nan for x in abs_tmp]
            plt.plot(tmp, label=kk, color=col_dict[kk])
        plt.yscale("log")
        plt.xlim([0, 99])
        plt.title(keys[i].replace("_", " "))
        plt.legend()
    # plt.tight_layout()
    plt.show()


# Diagnostics

plot_diagnostic(diagnostic.keys(), diagnostic)
plot_diagnostic([str(i + 1) for i in range(5)], diagnostic_branin_simple)


fusion_diag = dict()
fusion_diag["aIMSE"] = diagnostic["aIMSE"]
fusion_diag["2stepaIMSE"] = diagnostic["2stepaIMSE"]
col_dict = {"aIMSE": "red", "2stepaIMSE": "blue"}

for k in [str(i) for i in range(1, 6)]:
    fusion_diag[k] = diagnostic_branin_simple[k]
    col_dict[k] = "lightgrey"


plot_diagnostic(fusion_diag.keys(), fusion_diag, col_dict)


with open("diagnostic.pkl", "rb") as f:
    diagnostic = dill.load(f)

with open("diagnostic_branin_simple.pkl", "rb") as f:
    diagnostic_branin_simple = dill.load(f)


def plot_enrichment(branin1, branin2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.contour(xmg, ymg, branin1.predict(XY).reshape(50, 50))
    plt.scatter(branin1.gp.X_train_[:, 0], branin1.gp.X_train_[:, 1])
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$u$")
    plt.title(r"Maximum variance enrichment")
    plt.subplot(1, 2, 2)
    plt.contour(xmg, ymg, branin2.predict(XY).reshape(50, 50))
    plt.scatter(branin2.gp.X_train_[:60, 0], branin2.gp.X_train_[:60, 1])
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$u$")
    plt.title(r"aIMSE enrichment")
    plt.tight_layout()
    plt.savefig(graphics_folder + "enrichment.png")
    plt.show()


plot_enrichment(branin, branin_aug)

#  EOF ----------------------------------------------------------------------
