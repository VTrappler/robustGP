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
# import seaborn as sns

plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"],
    'image.cmap': u'viridis',
    'figure.figsize': [4, 4],
    'savefig.dpi': 400
})
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
graphics_folder = '/home/victor/collab_article/adaptive/figures/'


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
        n_restarts_optimizer=40,
    )
    branin.set_idxU([1])
    return branin


bounds = np.asarray([[0, 1], [0, 1]])
branin = initialize_branin()
# For plots
x, y = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
(XY, (xmg, ymg)) = tools.pairify((x, y))

global opt1
global opt2
global opt3
opt1 = []
opt2 = []
opt3 = []


def callback_stepwise(admethod, i, minimum=0.839753, prefix=""):
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title(r"Prediction $m_Z$")
    ax1.contourf(xmg, ymg, admethod.predict(XY).reshape(20, 20))
    ax1.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
    ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
    ax1.set_aspect("equal")
    ax2 = plt.subplot(2, 2, 2)
    ax2.contourf(
        xmg, ymg, -(admethod.enrichment.criterion(admethod, XY).reshape(20, 20))
    )
    ax2.set_aspect("equal")
    ax2.set_title("Criterion")

    ax3 = plt.subplot(2, 2, 3)
    opt1.append(((admethod.get_global_minimiser().fun - minimum) ** 2))
    opt2.append(((admethod.get_best_so_far() - minimum) ** 2))
    ax3.plot(opt1)
    ax3.plot(opt2)
    ax3.set_yscale("log")

    ax4 = plt.subplot(2, 2, 4)
    opt3.append(ac.prediction_variance(admethod, XY).mean())
    ax4.plot(opt3)
    ax4.set_yscale("log")
    ax4.set_title(r"IMSE")
    plt.tight_layout()
    plt.savefig(f"/home/victor/robustGP/robustGP/dump/{prefix}_stepwise{i:02d}.png")
    plt.close()


### Exploration

def maximum_prediction_variance():
    global opt1, opt2, opt3
    """Maximum of prediction variance"""
    branin = initialize_branin()
    pred_var = enrich.OneStepEnrichment(bounds)
    pred_var.set_criterion(ac.prediction_variance, maxi=True)  #
    pred_var.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
    branin.set_enrichment(pred_var)
    opt1 = []
    opt2 = []
    opt3 = []
    branin.run(
        Niter=50, callback=None
    )  # lambda ad, i: callback_stepwise(ad, i, prefix="predvar"))
    return branin


def reduction_aIMSE():
    global opt1, opt2, opt3
    """reduction of the augmented IMSE"""
    branin = initialize_branin()
    int_points = pyDOE.lhs(2, 100)  # Generate integration points
    augIMSE = enrich.OneStepEnrichment(bounds)
    augIMSE.set_criterion(
        ac.augmented_IMSE, maxi=False, scenarios=None, integration_points=int_points
    )
    augIMSE.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
    branin.set_enrichment(augIMSE)
    opt1 = []
    opt2 = []
    opt3 = []
    branin.run(Niter=25, callback=partial(callback_stepwise, prefix="augIMSE"))
    return branin


def infill_criterion():
    """ Infill criterion TBD"""
    infill_sobol = enrich.OneStepEnrichment(bounds)
    infill_sobol.set_criterion()
    infill_sobol.set_optim()


## Optimisation
def example_EGO():
    """ EGO """
    global opt1, opt2, opt3
    branin = initialize_branin()
    EGO = enrich.OneStepEnrichment(bounds)
    EGO.set_criterion(ac.expected_improvement, maxi=True)  #
    EGO.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
    branin.set_enrichment(EGO)
    opt1 = []
    opt2 = []
    opt3 = []
    branin.run(Niter=50, callback=partial(callback_stepwise, prefix="EGO"))
    return branin


## Contour/ Level set estimation

def reliability_index():
    """ reliability index"""
    global opt1, opt2, opt3
    branin = initialize_branin()
    reliability = enrich.OneStepEnrichment(bounds)
    reliability.set_criterion(ac.reliability_index_rho, maxi=True, T=1.3)  #
    reliability.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
    branin.set_enrichment(reliability)
    opt1, opt2, opt3 = [], [], []
    branin.run(Niter=50, callback=partial(callback_stepwise, prefix="rho"))
    return branin


def example_augmented_IVPC():
    branin = initialize_branin()
    augIVPC = enrich.OneStepEnrichment(bounds)
    augIVPC.set_criterion(
        ac.augmented_IVPC,
        scenarios=None,
        maxi=True,
        integration_points=pyDOE.lhs(2, 100),
        T=1.3,
    )  #
    augIVPC.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
    branin.set_enrichment(augIVPC)
    branin.run(Niter=50, callback=partial(callback_stepwise, prefix="augIVPCmaxi"))


# AK-MCS
import robustGP.sampling.samplers as sam
def example_AKMCS():
    branin = initialize_branin()
    AKMCS = enrich.AKMCSEnrichment(bounds)
    AKMCS.set_pdf(ac.margin_indicator, T=1.3, eta=0.05)
    AKMCS.set_sampler(sam.sampling_from_indicator, Nsamples=2000)
    AKMCS.set_clustering(sam.clustering, Kclusters=15)
    branin.set_enrichment(AKMCS)


    def callback_AKMCS(admethod, i):
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title(r"Prediction $m_Z$")
        ax1.contourf(xmg, ymg, admethod.predict(XY).reshape(20, 20))
        K = admethod.enrichment.Kclusters
        ax1.plot(admethod.gp.X_train_[:-K, 0], admethod.gp.X_train_[:-K, 1], "ob")
        ax1.plot(admethod.gp.X_train_[-K:, 0], admethod.gp.X_train_[-K:, 1], "or")
        ax1.set_aspect("equal")
        ax2 = plt.subplot(1, 2, 2)
        ax2.contourf(xmg, ymg, ac.probability_coverage(admethod, XY, 1.3).reshape(20, 20))
        ax2.set_aspect("equal")
        ax2.set_title("Probability of coverage")

        # ax3 = plt.subplot(2, 2, (3, 4))
        # opt1.append(np.log((admethod.get_global_minimiser().fun) ** 2))
        # opt2.append(np.log((admethod.get_best_so_far()) ** 2))
        # ax3.plot(opt1)
        # ax3.plot(opt2)

        plt.savefig(f"/home/victor/robustGP/robustGP/dump/im{i:02d}.png")
        plt.close()


    branin.run(Niter=5, callback=callback_AKMCS)
    return branin


## PEI_algo


def callback_PEI(admethod, i):
    xl, yl = np.linspace(0, 1, 500), np.linspace(0, 1, 500)
    (XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
    Jtrue = branin.function(XYl).reshape(500, 500)

    m, s = admethod.predict(XY, return_std=True)
    m = m.reshape(50, 50)
    s = s.reshape(50, 50)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title(r"Prediction $m_Z$")
    ax1.contourf(xmg, ymg, m)
    ax1.scatter(admethod.gp.X_train_[:, 0], admethod.gp.X_train_[:, 1], c="yellow", s=3, marker='x')
    # ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r", s=3)
    ax1.set_aspect("equal")
    ax1.set_ylabel(r"$u$")
    ax1.set_xlabel(r"$\theta$")

    ax3 = plt.subplot(2, 2, 3)
    ax3.contourf(
        xmg, ymg, -(admethod.enrichment.criterion(admethod, XY).reshape(50, 50))
    )
    # ax3.set_aspect("equal")
    ax3.set_title("PEI Criterion")
    ax3.set_ylabel(r"$u$")
    ax3.set_xlabel(r"$\theta$")

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title(r"Variance $\sigma^2_Z$")
    ax2.contourf(xmg, ymg, s**2)
    ax2.scatter(admethod.gp.X_train_[:, 0], admethod.gp.X_train_[:, 1],
                c="yellow", s=3, marker='x')
    # ax2.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r", s=3, marker='x')
    # ax2.set_aspect("equal")
    ax2 = plt.subplot(2, 2, 3)
    ax2.contourf(
        xmg, ymg, -(admethod.enrichment.criterion(admethod, XY).reshape(50, 50))
    )
    # ax2.set_aspect("equal")
    ax2.set_title("Criterion")
    ax2.set_ylabel(r"$u$")
    ax2.set_xlabel(r"$\theta$")

    Jpred = admethod.predict(XY).reshape(50, 50)
    theta_star_hat = x[Jpred.argmin(0)]
    mstar, sstar = admethod.predict(np.vstack([theta_star_hat, y]).T, return_std=True)
    ax1.plot(theta_star_hat, y, ".", color='red', alpha=0.9)

    # diff = mstar - Jtrue.min(0)
    # L2_error.append(np.sum(diff ** 2))
    # ax3 = plt.subplot(2, 2, 3)
    # ax3.plot(L2_error)
    # ax3.set_title(r"L2 error on $J^*$")
    # ax3.set_yscale("log")

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(y, mstar, label=r"$m_{Z^*}$", color='red')
    ax4.fill_between(y, mstar + sstar, mstar - sstar, color="lightgray", alpha=0.7)
    ax4.plot(yl, Jtrue.min(0), label=r"$J^*$")
    ax4.set_title(r"Prediction of $J^*$")
    ax4.set_xlabel(r"$\theta$")
    ax4.set_ylim([0.7, 1.6])
    # ax4.set_ylabel(r"")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"/home/victor/robustGP/robustGP/dump/PEI_{i}.png")
    plt.close()


def PEI_example():
    branin = initialize_branin()
    PEI_proc = enrich.OneStepEnrichment(bounds)
    PEI_proc.set_criterion(ac.PEI, maxi=True)  #
    PEI_proc.set_optim(opt.optimize_with_restart, **{"nrestart": 15, "bounds": bounds})
    branin.set_enrichment(PEI_proc)
    branin.run(Niter=30, callback=callback_PEI)
    return branin


# L2_total_predvar = np.empty((50, 0))

# for _ in range(10):
#     xl, yl = np.linspace(0, 1, 500), np.linspace(0, 1, 500)
#     (XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))
#     Jtrue = branin.function(XYl).reshape(500, 500)
#     branin = initialize_branin()
#     pred_var = enrich.OneStepEnrichment(bounds)
#     pred_var.set_criterion(ac.prediction_variance, maxi=True)  #
#     pred_var.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
#     branin.set_enrichment(pred_var)

#     # PEI_proc = enrich.OneStepEnrichment(bounds)
#     # PEI_proc.set_criterion(ac.PEI, maxi=True)  #
#     # PEI_proc.set_optim(opt.optimize_with_restart, **{"nrestart": 10})
#     # branin.set_enrichment(PEI_proc)

#     global L2_error
#     L2_error = []
#     branin.run(Niter=50, callback=callback_PEI)

#     # L2_total_predvar = np.empty((50, 0))
#     L2_total_predvar = np.c_[L2_total_predvar, L2_error]

# plt.plot(L2_total, color="k", alpha=0.2)
# plt.plot(L2_total.mean(1))
# plt.plot(L2_total_predvar, color="k", alpha=0.2)
# plt.plot(L2_total_predvar.mean(1))
# plt.yscale("log")
# plt.show()

# intU = pyDOE.lhs(1, 200)
# bounds = np.asarray([[0, 1], [0, 1]])
# branin = initialize_branin()
# x1 = np.linspace(0, 1, 100)
# plt.subplot(2, 2, (1, 3))
# plt.contourf(xmg, ymg, branin.predict(XY).reshape(20, 20))
# plt.subplot(2, 2, 2)
# m, c = predict_meanGP(branin, intU)(np.atleast_2d(x1), return_cov=True)
# plt.plot(m)
# plt.plot(m + np.sqrt(c))
# plt.plot(m - np.sqrt(c))
# plt.subplot(2, 2, 4)
# plt.plot(projected_EI(branin, x1, intU))
# plt.tight_layout()
# plt.show()


# EIVAR = enrich.TwoStepEnrichment(bounds)
# EIVAR.set_optim1(opt.optimize_with_restart, bounds=np.atleast_2d(branin.bounds[0]))
# EIVAR.set_optim2(opt.optimize_with_restart)
# EIVAR.set_criterion_step1(ac.projected_EI, maxi=True, intU=intU)
# EIVAR.set_criterion_step2(ac.augmented_VAR, maxi=True, intU=intU)
# branin.set_enrichment(EIVAR)


# def callback_EIVAR(admethod, i):
#     ax1 = plt.subplot(2, 2, 1)
#     ax1.set_title(r"Prediction $m_Z$")
#     ax1.contourf(xmg, ymg, admethod.predict(XY).reshape(20, 20))
#     ax1.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], c="b")
#     ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], c="r")
#     ax1.set_aspect("equal")
#     ax2 = plt.subplot(2, 2, 4)
#     Xnext = admethod.enrichment.run_stage1(admethod)[0]
#     ax2.contourf(
#         xmg, ymg, -(admethod.enrichment.criterion2(admethod, XY, Xnext).reshape(20, 20))
#     )
#     ax2.set_aspect("equal")
#     ax2.set_title("Augmented Var")
#     ax3 = plt.subplot(2, 2, 2)
#     m, c = ac.predict_meanGP(admethod, intU)(np.atleast_2d(x1), return_cov=True)
#     c = ac.predvar_meanGP(admethod, intU)(np.atleast_2d(x1))
#     ax3.plot(x1, m, "k")
#     ax3.plot(x1, m + np.sqrt(c), color="k")
#     ax3.plot(x1, m - np.sqrt(c), "k")
#     ax3.set_title(r"Projected process")
#     # ax3bis.plot(x1, c)
#     ax4 = plt.subplot(2, 2, 3)
#     ax4.plot(ac.projected_EI(admethod, x1, intU))
#     ax4.set_title(r"Projected EI")
#     plt.tight_layout()
#     plt.savefig(f"/home/victor/robustGP/robustGP/dump/EIVAR2_stepwise{i:02d}.png")
#     plt.close()


# branin.run(Niter=50, callback=callback_EIVAR)


if __name__ == '__main__':
    PEI_example()
