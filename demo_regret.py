#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pyDOE
from sklearn.gaussian_process.kernels import Matern

from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d
import robustGP.tools as tools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from functools import partial
import seaborn as sns

sns.set_theme()


# Initialisation of the problem
def initialize_branin():
    """
    Create new instance of AdaptiveStrategy of the Branin 2d function
    with LHS as initial design
    """
    NDIM = 2
    bounds = np.asarray([(0, 1)] * NDIM)
    initial_design = pyDOE.lhs(
        n=NDIM, samples=5 * NDIM, criterion="maximin", iterations=50
    )
    branin = AdaptiveStrategy(bounds, branin_2d)
    branin.fit_gp(
        initial_design,
        branin.evaluate_function(initial_design),
        Matern(np.ones(NDIM)),
        n_restarts_optimizer=20,
    )
    branin.set_idxU([1], ndim=2)
    return branin


bounds = np.asarray([[0, 1], [0, 1]])
branin = initialize_branin()
# For plots
x, y = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
(XY, (xmg, ymg)) = tools.pairify((x, y))

xl, yl = np.linspace(0, 1, 500), np.linspace(0, 1, 500)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
Jtrue = branin.function(XYl).reshape(500, 500)


def plot_truth():
    plt.subplot(2, 2, 1)
    plt.contourf(xmgl, ymgl, Jtrue)
    theta_star = xl[Jtrue.argmin(0)]
    plt.plot(theta_star, yl, "w.")

    Jpred = branin.predict(XYl).reshape(500, 500)
    plt.subplot(2, 2, 2)

    theta_star_hat = xl[Jpred.argmin(0)]
    mstar, sstar = branin.predict(np.vstack([theta_star_hat, yl]).T, return_std=True)
    regret_add = Jpred - mstar[np.newaxis, :]
    regret_rel = Jpred / mstar[np.newaxis, :]
    plt.contourf(xmgl, ymgl, Jpred - mstar[np.newaxis, :])
    plt.plot(theta_star_hat, yl, "w.")

    plt.subplot(2, 2, 3)
    plt.plot(yl, mstar, label="GP")
    plt.fill_between(yl, mstar + sstar, mstar - sstar, color="lightgray", alpha=0.7)
    plt.plot(yl, Jtrue.min(0), label="truth")
    plt.legend()
    plt.tight_layout()
    plt.show()


theta_star = xl[Jtrue.argmin(0)]
Jpred = branin.predict(XYl).reshape(500, 500)
theta_star_hat = xl[Jpred.argmin(0)]
mstar, sstar = branin.predict(np.vstack([theta_star_hat, yl]).T, return_std=True)
regret_add = Jpred - mstar[np.newaxis, :]
regret_rel = Jpred / mstar[np.newaxis, :]
plt.subplot(1, 2, 1)
plt.contour(xmgl, ymgl, regret_add)
plt.subplot(1, 2, 2)
plt.contour(xmgl, ymgl, regret_rel)
plt.show()


def predvar_Delta(arg, X):
    m, va = arg.predict_GPdelta(X, alpha=1.0, beta=0.0)
    return va


def optimize_and_adjustment(fun, bounds):
    optim = opt.optimize_with_restart(fun, bounds, nrestart=25)
    x1, x2 = branin.separate_input(np.atleast_2d(optim[0]))
    set_input = branin.create_input(x2)
    x1_star = branin.get_conditional_minimiser(x2).x
    _, _, both = robustGP.gptools.gp_to_delta(
        branin,
        set_input(x1).flatten(),
        set_input(x1_star).flatten(),
        alpha=1.0,
        beta=0.0,
        return_var=True,
    )
    if both[0] < both[1]:
        return np.c_[x1_star, x2], optim[1], both
    else:
        return optim[0], optim[1], both


IMSE_total_pv = np.empty((50, 0))
for _ in range(10):
    branin = initialize_branin()
    global IMSE
    IMSE = []
    # predvar_AR = enrich.OneStepEnrichment(bounds)
    # predvar_AR.set_criterion(
    #     predvar_Delta,
    #     maxi=True,
    # )  #
    # predvar_AR.set_optim(optimize_and_adjustment)

    pred_var = enrich.OneStepEnrichment(bounds)
    pred_var.set_criterion(ac.prediction_variance, maxi=True)  #
    pred_var.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
    branin.set_enrichment(pred_var)
    # branin.set_enrichment(predvar_AR)
    branin.run(Niter=50, callback=plot_GP_prediction)
    IMSE_total_pv = np.c_[IMSE_total_pv, IMSE]

plt.plot(IMSE_total, "k", alpha=0.4)
plt.plot(IMSE_total_pv, "b", alpha=0.4)
plt.yscale("log")
plt.show()


def plot_GP_prediction(admethod, i, alpha=1.0, shift=0):
    global IMSE
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title(r"Prediction $m_\Delta$")
    pred, var = admethod.predict_GPdelta_product(x, y, alpha=1.0)
    # ivpc = ac.variance_probability_coverage((pred, np.sqrt(var)), None, 0).reshape(
    #     20, 20
    # )

    ax1.contourf(xmg, ymg, pred.reshape(20, 20))
    ax1.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
    ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
    ax1.set_aspect("equal")
    ax2 = plt.subplot(2, 2, 2)
    ax2.contourf(xmg, ymg, var.reshape(20, 20))
    ax2.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
    ax2.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
    ax2.set_aspect("equal")
    ax2.set_title(r"$\sigma_\Delta$")
    IMSE.append(var.mean())
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title(r"Prediction $m_Z$")
    mZ, sZ = admethod.predict(XY, return_std=True)
    ax3.contourf(xmg, ymg, mZ.reshape(20, 20))
    ax3.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
    ax3.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
    ax3.set_aspect("equal")
    ax4 = plt.subplot(2, 2, 4)
    ax4.contourf(xmg, ymg, sZ.reshape(20, 20))
    ax4.set_aspect("equal")
    ax4.set_title(r"$\sigma_Z$")
    plt.tight_layout()
    plt.savefig(f"/home/victor/robustGP/robustGP/dump/predvar_{i+shift:02d}.png")
    # with open("/home/victor/robustGP/robustGP/dump/imse_ivpc.txt", "a+") as f:
    #     f.write(f"{i+shift}, {var.mean()}, {ivpc.mean()}\n")
    plt.close()


plot_GP_prediction(branin, i=0, alpha=1.0)

IMSE = []


def augmented_IMSE_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
    def function_(arg):
        m, va = arg.predict_GPdelta(integration_points, alpha=alpha, beta=beta)
        return va

    return ac.augmented_design(arg, X, scenarios, function_, {})


integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=20)

# augIMSE = augmented_IMSE_Delta(branin, XY, None, integration_points, 1.3, 0)
branin = initialize_branin()
aIMSE_Delta = enrich.OneStepEnrichment(bounds)
aIMSE_Delta.set_criterion(
    augmented_IMSE_Delta,
    maxi=False,
    scenarios=None,
    integration_points=integration_points,
    alpha=1,
    beta=0,
)  #
aIMSE_Delta.set_optim(opt.optimize_with_restart, **{"nrestart": 5})
branin.set_enrichment(aIMSE_Delta)
branin.run(Niter=5, callback=plot_GP_prediction)
plt.plot(IMSE_total, "k", alpha=0.4)
plt.plot(IMSE_total_pv, "b", alpha=0.4)
plt.plot(IMSE, "r")
plt.yscale("log")
plt.show()

aimse = np.genfromtxt("/home/victor/robustGP/robustGP/dump/imse.txt", delimiter=",")

## IVPC


def augmented_IVPC_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
    def function_(arg):
        m, va = arg.predict_GPdelta(integration_points, alpha=alpha, beta=beta)
        s = np.sqrt(va)
        return ac.variance_probability_coverage((m, s), None, 0)

    return ac.augmented_design(arg, X, scenarios, function_, {})


bounds = np.asarray([[0, 1], [0, 1]])
branin_2 = initialize_branin()
aIVPC_Delta = enrich.OneStepEnrichment(bounds)
aIVPC_Delta.set_criterion(
    augmented_IVPC_Delta,
    maxi=False,
    scenarios=None,
    integration_points=integration_points,
    alpha=1.3,
    beta=0,
)  #
aIVPC_Delta.set_optim(opt.optimize_with_restart, **{"nrestart": 5})
branin_2.set_enrichment(aIVPC_Delta)
branin_2.run(Niter=5, callback=partial(plot_GP_prediction, shift=15))
