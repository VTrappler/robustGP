#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pyDOE
from sklearn.gaussian_process.kernels import Matern

from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d
from robustGP.tools import pairify
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from functools import partial


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
    branin.set_idxU([1])
    return branin


bounds = np.asarray([[0, 1], [0, 1]])
branin = initialize_branin()
# For plots
x, y = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
(XY, (xmg, ymg)) = pairify((x, y))

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


## Exploration
# Maximum of prediction variance
branin = initialize_branin()
pred_var = enrich.OneStepEnrichment(bounds)
pred_var.set_criterion(ac.prediction_variance, maxi=True)  #
pred_var.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
branin.set_enrichment(pred_var)
opt1 = []
opt2 = []
opt3 = []
branin.run(Niter=50, callback=lambda ad, i: callback_stepwise(ad, i, prefix="predvar"))

# reduction of the augmented IMSE
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


## Optimisation
#  EGO
branin = initialize_branin()
EGO = enrich.OneStepEnrichment(bounds)
EGO.set_criterion(ac.expected_improvement, maxi=True)  #
EGO.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
branin.set_enrichment(EGO)
opt1 = []
opt2 = []
opt3 = []
branin.run(Niter=50, callback=partial(callback_stepwise, prefix="EGO"))


## Contour/ Level set estimation
#   reliability index
branin = initialize_branin()
reliability = enrich.OneStepEnrichment(bounds)
reliability.set_criterion(ac.reliability_index_rho, maxi=True, T=1.3)  #
reliability.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
branin.set_enrichment(reliability)
opt1, opt2, opt3 = [], [], []
branin.run(Niter=50, callback=partial(callback_stepwise, prefix="rho"))

# AK-MCS
import robustGP.sampling.samplers as sam

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


## PEI_algo

branin = initialize_branin()
PEI_proc = enrich.OneStepEnrichment(bounds)
PEI_proc.set_criterion(ac.PEI, maxi=True)  #
PEI_proc.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
branin.set_enrichment(PEI_proc)
opt1 = []
opt2 = []
opt3 = []
import seaborn as sns
sns.set_theme()
branin.run(Niter=50, callback=partial(callback_stepwise, prefix="PEI"))

