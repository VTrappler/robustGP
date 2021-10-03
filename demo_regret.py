#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pyDOE
from sklearn.gaussian_process.kernels import Matern
import cma
from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt
from functools import partial
plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"],
    'image.cmap': u'viridis',
    'figure.figsize': [5.748031686730317, 3.552478950810724],
    'savefig.dpi': 200
})
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
graphics_folder = '/home/victor/collab_article/adaptive/figures/'


x, y = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
(XY, (xmg, ymg)) = tools.pairify((x, y))


# Initialisation of the problem
def initialize_branin(initial_design):
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


def predvar_Delta(arg, X, alpha):
    m, va = arg.predict_GPdelta(X, alpha=alpha, beta=0.0)
    return va


def optimize_and_adjustment(fun, bounds, arg, alpha):
    optim = opt.optimize_with_restart(fun, bounds, nrestart=25)
    x1, x2 = arg.separate_input(np.atleast_2d(optim[0]))
    set_input = arg.create_input(x2)
    x1_star = arg.get_conditional_minimiser(x2).x
    _, _, both = robustGP.gptools.gp_to_delta(
        arg,
        set_input(x1).flatten(),
        set_input(x1_star).flatten(),
        alpha=alpha,
        beta=0.0,
        return_var=True,
    )
    print(both)
    if both[0] < both[1]:
        print('adjustment')
        return np.c_[x1_star, x2], optim[1], both
    else:
        return optim[0], optim[1], both


# IMSE_predvar = {}
for alpha in [# 0.0, 1.0, 
              1.0]:
    IMSE_total_1 = np.empty((100, 0))
    print(f"alpha = {alpha}")
    for _ in range(10):
        branin_ = initialize_branin()
        global IMSE_alpha
        IMSE_alpha = []
        # predvar_AR = enrich.OneStepEnrichment(bounds)
        # predvar_AR.set_criterion(
        #     partial(predvar_Delta, alpha=alpha),
        #     maxi=True,
        # )  #
        # predvar_AR.set_optim(partial(optimize_and_adjustment, arg=branin_, alpha=alpha))

        pred_var = enrich.OneStepEnrichment(bounds)
        pred_var.set_criterion(ac.prediction_variance, maxi=True)  #
        pred_var.set_optim(opt.optimize_with_restart, **{"nrestart": 20})
        branin_.set_enrichment(pred_var)
        branin_.set_enrichment(predvar_AR)
        branin_.run(Niter=100, callback=partial(plot_GP_prediction,
                                                alpha=alpha, mutable=IMSE_alpha))
        IMSE_total_1 = np.c_[IMSE_total_1, IMSE_alpha]
    # IMSE_predvar[str(alpha)] = IMSE_total

IMSE_adj = {'1.0': IMSE_total_1,
            '2.0': IMSE_total}
plt.subplot(1, 2, 1)
for i, (k, v) in enumerate(IMSE_adj.items()):
    plt.plot(v, color=c[1+i], alpha=0.3)
    plt.plot(v.mean(1), color=c[1 + i], lw=2)
plt.yscale('log')
plt.subplot(1, 2, 2)
c = ['b', 'r', 'g']
for i, (k, v) in enumerate(IMSE_predvar.items()):
    plt.plot(v, color=c[i], alpha=0.3)
    plt.plot(v.mean(1), color=c[i], lw=2)
plt.yscale('log')
# plt.plot(IMSE_total, alpha=0.3, color='m')
plt.show()

for i, (k, v) in enumerate(IMSE_predvar.items()):
    # plt.plot(v, color=c[i], alpha=0.3)
    plt.plot(np.diff(np.log(v.mean(1))), color=c[i], lw=2)


IMSE_alpha = {'adjusted': IMSE_adj,
              'vanilla': IMSE_predvar}


with open("/home/victor/robustGP/IMSE_alpha.data", "wb") as f:
    dill.dump(IMSE_alpha, f)


plt.plot(IMSE_total, "k", alpha=0.4)
plt.plot(IMSE_alpha13_total, "k", alpha=0.4)

plt.plot(IMSE_total_pv, "b", alpha=0.4)
plt.yscale("log")
plt.show()
global IMSE_alph
IMSE_alpha13 = []


def plot_GP_prediction(admethod, i, alpha=2, shift=0, mutable=IMSE_alpha13):
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title(r"Prediction $m_\Delta$")
    pred, var = admethod.predict_GPdelta_product(x, y, alpha=alpha)
    # ivpc = ac.variance_probability_coverage((pred, np.sqrt(var)), None, 0).reshape(
    #     20, 20
    # )

    ax1.contourf(xmg, ymg, pred.reshape(50, 50))
    ax1.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], s=2, c="b")
    ax1.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], s=2, c="r")
    ax1.set_aspect("equal")
    ax2 = plt.subplot(2, 2, 2)
    ax2.contourf(xmg, ymg, var.reshape(50, 50))
    ax2.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], s=2, c="b")
    ax2.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], s=2, c="r")
    ax2.set_aspect("equal")
    ax2.set_title(r"$\sigma_\Delta$")
    mutable.append(var.mean())
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title(r"Prediction $m_Z$")
    mZ, sZ = admethod.predict(XY, return_std=True)
    ax3.contourf(xmg, ymg, mZ.reshape(50, 50))
    ax3.scatter(admethod.gp.X_train_[:-1, 0], admethod.gp.X_train_[:-1, 1], s=2, c="b")
    ax3.scatter(admethod.gp.X_train_[-1, 0], admethod.gp.X_train_[-1, 1], s=2, c="r")
    ax3.set_aspect("equal")
    ax4 = plt.subplot(2, 2, 4)
    ax4.contourf(xmg, ymg, sZ.reshape(50, 50))
    ax4.set_aspect("equal")
    ax4.set_title(r"$\sigma_Z$")
    plt.tight_layout()
    plt.savefig(f"/home/victor/robustGP/robustGP/dump/new_aIMSE_{i+shift:02d}.png")
    # with open("/home/victor/robustGP/robustGP/dump/imse_ivpc.txt", "a+") as f:
    #     f.write(f"{i+shift}, {var.mean()}, {ivpc.mean()}\n")
    plt.close()


plot_GP_prediction(branin, i=0, alpha=1.0)



def augmented_IMSE_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
    """
    Computes the averaged integrated imse, based on the scenarios
    """
    if isinstance(integration_points, int):
        integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=50)

    function_ = lambda arg: arg.predict_GPdelta(integration_points, alpha=alpha, beta=beta)[1]
    return ac.augmented_design(arg, X, scenarios, function_, {})


integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=20)

# augIMSE = augmented_IMSE_Delta(branin, XY, None, integration_points, 1.3, 0)
branin = initialize_branin()
aIMSE_Delta = enrich.OneStepEnrichment(bounds)
aIMSE_Delta.set_optim(opt.optimize_with_restart, **{"nrestart": 5})


global IMSE_aug
IMSE_aug = []
for i in range(5):
    integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=20)
    aIMSE_Delta.set_criterion(
        augmented_IMSE_Delta,
        maxi=False,
        scenarios=None,
        integration_points=integration_points,
        alpha=2,
        beta=0,
    )  #
    branin.set_enrichment(aIMSE_Delta)

    branin.run(Niter=1, callback=partial(plot_GP_prediction, shift=5 + i, mutable=IMSE_aug))


branin = initialize_branin()
opts = cma.CMAOptions()
opts['bounds'] = list(zip(*bounds))
opts['maxfevals'] = 150
aIMSE_Delta.set_optim(cma.fmin2, **{'x0': np.array([0.5, 0.5]),
                                    'sigma0': 0.3,
                                    'options': opts})
aIMSE_Delta.set_criterion(augmented_IMSE_Delta,
                          maxi=False,
                          scenarios=None,
                          integration_points=200,
                          alpha=2, beta=0)  #
branin.set_enrichment(aIMSE_Delta)
IMSE_r = []
branin.run(Niter=2, callback=partial(plot_GP_prediction, alpha=2.0, shift=13, mutable=IMSE_r))




# plt.plot(IMSE_total, "k", alpha=0.4)
# plt.plot(IMSE_total_pv, "b", alpha=0.4)
plt.plot(IMSE_alpha13_total, 'k', lw=0.5)
plt.plot(IMSE_aug, "r")
plt.yscale("log")
plt.show()

import dill

to_dill = {'branin': branin,
           'IMSE_pv': IMSE_total_pv,
           'IMSE_adjustment': IMSE_total,
           'IMSE_aIMSE': IMSE,
           'alpha': 1}

with open("/home/victor/robustGP/aIMSE_comp.pkl", "wb") as f:
    dill.dump(to_dill, f)

with open("/home/victor/robustGP/aIMSE_comp.pkl", "rb") as f:
    bbb = dill.load(f)
    

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



## Augmented IMSE

import cma



def augmented_IMSE_Delta(arg, X, scenarios, integration_points, alpha, beta=0):
    def function_(arg):
        m, va = arg.predict_GPdelta(integration_points, alpha=alpha, beta=beta)
        return va

    return ac.augmented_design(arg, X, scenarios, function_, {})


integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=50)

# augIMSE = augmented_IMSE_Delta(branin, XY, None, integration_points, 1.3, 0)
branin_aIMSE = initialize_branin()
aIMSE_Delta = enrich.OneStepEnrichment(bounds)

# aIMSE_Delta.set_optim(opt.optimize_with_restart, **{"nrestart": 5})
opts = cma.CMAOptions()
opts['bounds'] = list(zip(*bounds))
opts['maxfevals'] = 200
aIMSE_Delta.set_optim(cma.fmin2, **{'x0': np.array([0.5, 0.5]),
                                    'sigma0': 0.3,
                                    'options': opts})

global IMSE_aug
# IMSE_aug = []
for i in range(10):
    integration_points = pyDOE.lhs(2, 200, criterion="maximin", iterations=20)
    aIMSE_Delta.set_criterion(
        augmented_IMSE_Delta,
        maxi=False,
        scenarios=None,
        integration_points=integration_points,
        alpha=2,
        beta=0,
    )  #
    branin_aIMSE.set_enrichment(aIMSE_Delta)
    branin_aIMSE.run(Niter=1, callback=partial(plot_GP_prediction, shift=len(IMSE_aug) + i, mutable=IMSE_aug))
print(IMSE_aug)
# plt.plot(IMSE_total, "k", alpha=0.4)
# plt.plot(IMSE_total_pv, "b", alpha=0.4)
# plt.plot(IMSE_alpha13_total, 'k', lw=0.5)
plt.plot(IMSE_aug, "r")
plt.yscale("log")
plt.show()





xl, yl = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))

Jtrue = branin.function(XYl).reshape(100, 100)
Gamma_t = (Jtrue - 2.0 * Jtrue.min(0) <= 0).mean(1)
np.sum((Gamma_t - (pred <= 0).mean(1))**2)

pred, var = branin_placeholder.predict_GPdelta_product(xl, yl, alpha=2.0)
pred = pred.reshape(100, 100)
plt.subplot(2, 1, 1)
plt.contourf(xmgl, ymgl, pred)
plt.subplot(2, 1, 2)
plt.plot((pred <= 0).mean(1))
plt.tight_layout()
plt.show()



aIMSE_alpha = {}
aIMSE_alpha['2.0'] = []

L2_err = np.empty((100))
for i in tqdm(range(100)):
    branin_placeholder = initialize_branin()
    gp_rem = rm_obs_gp(branin_aIMSE.gp, 10, i)
    branin_placeholder.gp = gp_rem
    # plot_GP_prediction(branin_placeholder, i, alpha=2, shift=0, mutable=aIMSE_alpha['2.0'])
    pred, var = branin_placeholder.predict_GPdelta_product(xl, yl, alpha=2.0)
    pred = pred.reshape(100, 100)
    L2_err[i] = np.sum((Gamma_t - (pred <= 0).mean(1))**2)


data_to_dill = {}
data_to_dill['branin'] = branin_aIMSE
data_to_dill['aIMSE_alpha'] = aIMSE_alpha
data_to_dill['L2_error'] = L2_err

with open("/home/victor/robustGP/branin_aIMSE.data", "wb") as f:
    dill.dump(data_to_dill, f)

with open("/home/victor/robustGP/branin_aIMSE.data", "rb") as f:
    bbb = dill.load(f)


maimse, saimse = bbb['branin'].predict(XY, return_std=True)
mDaimse, sDaimse = bbb['branin'].predict_GPdelta(XY, alpha=1.3)

def training_labels():
    plt.scatter(xtr[:, 0], xtr[:, 1], s=4, c='white', alpha=0.6)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
    plt.gca().set_aspect('equal')


plt.figure(figsize=(5, 4))
plt.subplot(2, 2, 1)
plt.contourf(xmg, ymg, maimse.reshape(50, 50))
training_labels()
plt.title(r'GP prediction $m_Z$')
plt.subplot(2, 2, 2)
plt.contourf(xmg, ymg, (saimse**2).reshape(50, 50))
training_labels()
plt.title(r'GP variance $\sigma^2_Z$')
training_labels()
plt.subplot(2, 2, 3)
plt.title(r'GP prediction $m_{\Delta_{\alpha}}$')
plt.contourf(xmg, ymg, mDaimse.reshape(50, 50))
training_labels()
plt.subplot(2, 2, 4)
plt.title(r'GP variance $\sigma^2_{\Delta_{\alpha}}$')
plt.contourf(xmg, ymg, sDaimse.reshape(50, 50))
training_labels()
plt.tight_layout()
plt.savefig('/home/victor/acadwriting/Slides/Figures/aimse.png', dpi=400)
plt.close()



augmented_IMSE_alpha = {}
branin_ph = initialize_branin()
augmented_IMSE_alpha['branin'] = branin_ph
for alpha in tqdm([0.0, 1.0, 2.0, 3.0]):
    print(str(alpha))
    augmented_IMSE_alpha[str(alpha)] = augmented_IMSE_Delta(branin_ph,
                                                            XY, None,
                                                            integration_points, alpha, 0)
with open("/home/victor/robustGP/aIMSE_alpha.data", "wb") as f:
    dill.dump(augmented_IMSE_alpha, f)

ax1 = plt.subplot(2, 2, 1)
ax1.set_title(r"Prediction $m_\Delta$")
pred, var = branin_aIMSE.predict_GPdelta_product(x, y, alpha=alpha)
# ivpc = ac.variance_probability_coverage((pred, np.sqrt(var)), None, 0).reshape(
#     20, 20
# )

plt.rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern Roman']})
plt.rcParams.update(params)
plt.rcParams["font.family"] = "serif"
ax1.contourf(xmg, ymg, pred.reshape(20, 20))
ax1.scatter(branin_aIMSE.gp.X_train_[:, 0], branin_aIMSE.gp.X_train_[:, 1], c="b")
ax1.set_aspect("equal")
ax2 = plt.subplot(2, 2, 2)
ax2.contourf(xmg, ymg, var.reshape(20, 20))
ax2.scatter(branin_aIMSE.gp.X_train_[:-1, 0], branin_aIMSE.gp.X_train_[:-1, 1], c="b")
ax2.scatter(branin_aIMSE.gp.X_train_[-1, 0], branin_aIMSE.gp.X_train_[-1, 1], c="r")
ax2.set_aspect("equal")
ax2.set_title(r"$\sigma_\Delta$")
ax3 = plt.subplot(2, 2, 3)
ax3.set_title(r"Prediction $m_Z$")
mZ, sZ = branin_aIMSE.predict(XY, return_std=True)
ax3.contourf(xmg, ymg, mZ.reshape(20, 20))
ax3.scatter(branin_aIMSE.gp.X_train_[:-1, 0], branin_aIMSE.gp.X_train_[:-1, 1], c="b")
ax3.scatter(branin_aIMSE.gp.X_train_[-1, 0], branin_aIMSE.gp.X_train_[-1, 1], c="r")
ax3.set_aspect("equal")
ax4 = plt.subplot(2, 2, 4)
ax4.contourf(xmg, ymg, sZ.reshape(20, 20))
ax4.set_aspect("equal")
ax4.set_title(r"$\sigma^2_Z$")
plt.tight_layout()
# plt.savefig(f"/home/victor/robustGP/robustGP/dump/predvar_{i+shift:02d}.png")
# with open("/home/victor/robustGP/robustGP/dump/imse_ivpc.txt", "a+") as f:
#     f.write(f"{i+shift}, {var.mean()}, {ivpc.mean()}\n")
plt.show()

plt.subplot(1, 3, 1)
plt.contourf(xmg, ymg, sZ.reshape(20, 20))
plt.title(r"$\sigma^2_Z$")
plt.subplot(1, 3, 2)
plt.contourf(xmg, ymg, var.reshape(20, 20))
plt.scatter(branin_aIMSE.gp.X_train_[:-1, 0], branin_aIMSE.gp.X_train_[:-1, 1], c="b")
plt.scatter(branin_aIMSE.gp.X_train_[-1, 0], branin_aIMSE.gp.X_train_[-1, 1], c="r")
plt.title(r"$\sigma^2_\Delta$")
plt.subplot(1, 3, 3)
plt.contourf(xmg, ymg, aa.reshape(20, 20))
plt.scatter(branin_aIMSE.gp.X_train_[:, 0], branin_aIMSE.gp.X_train_[:, 1], c="b")
plt.title(r'augmented IMSE')
plt.tight_layout()
for ax in plt.gcf().axes:
    ax.set_aspect('equal')
plt.show()




#  EOF ----------------------------------------------------------------------
