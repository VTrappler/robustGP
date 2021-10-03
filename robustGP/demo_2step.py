#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports: General tools
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
graphics_folder = '/home/victor/acadwriting/Slides/Figures/'


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


bounds = np.asarray([[0, 1], [0, 1]])
# For plots
x, y = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
(XY, (xmg, ymg)) = tools.pairify((x, y))
xl, yl = np.linspace(0, 1, 500), np.linspace(0, 1, 500)
(XYl, (xmgl, ymgl)) = tools.pairify((xl, yl))


def optimiser_search(cr):
    x = np.linspace(0, 1, 50)
    res = cr(x)
    print(f'first phase: {x[res.argmin()]}, {res.min()}')
    return x[res.argmin()], res.min()


def optimiser_1D(cr):
    optim = opt.optimize_with_restart(cr, np.array([[0, 1]]), 5)
    print(f'first phase: {optim[0]}, {optim[1]}')
    return optim


def optimiser_2_search(cr, bounds):
    res = cr(XY)
    print(f'second phase: {XY[res.argmin()]}, {res.min()}')
    return XY[res.argmin()], res.min()


def optimiser_2D(cr, bounds):
    optim = opt.optimize_with_restart(cr, bounds, 5)
    print(f'second phase: {optim[0]}, {optim[1]}')
    return optim


def optimiser_1D_cma(cr):
    opts = cma.CMAOptions()
    opts['bounds'] = [(0,), (1,)]
    opts['maxfevals'] = 30
    return partial(cma.fmin2, x0=np.array([0.5]),
                   sigma0=0.3,
                   options=opts)(cr)


def optimiser_2D_cma(cr, bounds):
    opts = cma.CMAOptions()
    opts['bounds'] = list(zip(*bounds))
    opts['maxfevals'] = 150
    opts['verbose'] = 1
# two_steps.set_optim2(cma.fmin2, **{'x0':
#                                    'sigma0': 0.3,
#                                    'options': opts})
    return partial(cma.fmin2, x0=np.array([0.5, 0.5]),
                   sigma0=0.3,
                   options=opts)(cr)


def criterion_1(gp, X):
    inp = tools.construct_input(np.atleast_2d(X).T,
                                np.atleast_2d(np.linspace(0, 1, 10)).T,
                                idx2=[1],
                                product=True)
    m, s = gp.predict(inp, return_std=True)
    s = s.reshape(50, 10)
    return -(s**2).sum(1)


def criterion_2(gp, X, Xnext):
    # inp = tools.construct_input(np.atleast_2d(Xnext).T,
    #                             np.atleast_2d(np.linspace(0, 1, 10)).T,
    #                             idx2=[1],
    #                             product=True)
    m, s = gp.predict(X, return_std=True)
    # s = s.reshape(len(X), 10)
    return -(s**2)


def augmented_IMSE(gp, X, Xnext):
    integration_points = tools.construct_input(np.atleast_2d(Xnext).T,
                                               np.atleast_2d(np.linspace(0, 1, 15)).T,
                                               idx2=[1],
                                               product=True)

    def function_(arg):
        m, va = arg.predict_GPdelta(integration_points, alpha=2.0)
        return va**2
    return ac.augmented_design(gp, X, None, function_, {})


def augmented_IVPC(gp, X, Xnext):
    integration_points = tools.construct_input(np.atleast_2d(Xnext).T,
                                               np.atleast_2d(np.linspace(0, 1, 15)).T,
                                               idx2=[1],
                                               product=True)

    def function_(arg):
        m, s = arg.predict_GPdelta(integration_points, alpha=2.0)
        _, varcov = pro_var_coverage(m, s)
        return varcov
    return ac.augmented_design(gp, X, None, function_, {})


branin = initialize_branin()
two_steps = enrich.TwoStepEnrichment(bounds)
two_steps.set_criterion_step1(partial(LB_, nu=None))
two_steps.set_criterion_step2(augmented_IMSE)
two_steps.set_optim1(optimiser_1D)
two_steps.set_optim2(optimiser_2D_cma)
branin.set_enrichment(two_steps)
branin.run(Niter=15, callback=callback)

branin_other = initialize_branin()
two_steps = enrich.TwoStepEnrichment(bounds)
two_steps.set_criterion_step1(partial(LB_other, nu=None))
two_steps.set_criterion_step2(augmented_IMSE)
two_steps.set_optim1(optimiser_1D)
two_steps.set_optim2(optimiser_2D_cma)
branin_other.set_enrichment(two_steps)
branin_other.run(Niter=90, callback=partial(callback))


branin_ivpc = initialize_branin()
two_steps = enrich.TwoStepEnrichment(bounds)
two_steps.set_criterion_step1(partial(LB_other, nu=None))
two_steps.set_criterion_step2(augmented_IVPC)
two_steps.set_optim1(optimiser_1D)
two_steps.set_optim2(optimiser_2D_cma)
branin_ivpc.set_enrichment(two_steps)
branin_ivpc.run(Niter=10, callback=partial(callback, filename=None))



plt.plot(branin.gp.X_train_[10:, 0], branin.gp.X_train_[10:, 1], '.r')
plt.show()


def nu_ev(i):
    return np.exp(-i / 50.0)


def pro_var_coverage(mdel, sdel):
    pi = scipy.stats.norm.cdf(-mdel / sdel)
    return pi, pi * (1 - pi)


def LB_(arg, X, nu=None):
    if nu is None:
        i = len(arg.gp.X_train_) - 10
        nu = nu_ev(i)
    inte = np.random.uniform(size=20)
    inp = tools.construct_input(np.atleast_2d(X).T,
                                np.atleast_2d(inte).T,
                                idx2=[1],
                                product=True)
    mdel, vdel = arg.predict_GPdelta(inp, alpha=2)
    pro, var = pro_var_coverage(mdel, np.sqrt(vdel))
    LB = (pro.reshape(len(X), 20).mean(1) - nu * np.sqrt(var).reshape(len(X), 20).mean(1))
    return -LB


def LB_other(arg, X, nu=None):
    if nu is None:
        i = len(arg.gp.X_train_) - 10
        nu = nu_ev(i)
    inte = np.random.uniform(size=20)
    inp = tools.construct_input(np.atleast_2d(X).T,
                                np.atleast_2d(inte).T,
                                idx2=[1],
                                product=True)
    mdel, vdel = arg.predict_GPdelta(inp, alpha=2)
    low, mid, hi = [(mdel + mul * nu * np.sqrt(vdel)).reshape(len(X), 20) for mul in [-1, 0, 1]]
    return (low < 0).mean(1)


def callback(arg, i, filename):
    mdel, sdel = arg.predict_GPdelta(XY, alpha=2)
    n_added_points = len(arg.gp.X_train_) - 10

    nu = nu_ev(i)
    low, mid, hi = [(mdel + mul * nu * np.sqrt(sdel)).reshape(50, 50) for mul in [-1, 0, 1]]

    pro, var = pro_var_coverage(mdel, sdel)
    pro = pro.reshape(50, 50)
    var = var.reshape(50, 50)

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 2, 1)
    plt.contourf(xmg, ymg, mdel.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], '.w')
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], '.r')
    plt.title(r'$m_{\Delta}$')

    plt.subplot(3, 2, 3)
    plt.contourf(xmg, ymg, sdel.reshape(50, 50))
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], '.w')
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], '.r')
    plt.title(r'$\sigma^2_{\Delta}$')

    plt.subplot(3, 2, 2)
    plt.contourf(xmg, ymg, var)
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], '.w')
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], '.r')
    plt.title('Coverage variance')

    plt.subplot(3, 2, 4)
    plt.contourf(xmg, ymg, pro)
    plt.plot(arg.gp.X_train_[:10, 0], arg.gp.X_train_[:10, 1], '.w')
    plt.plot(arg.gp.X_train_[10:, 0], arg.gp.X_train_[10:, 1], '.r')
    plt.title('Prob of coverage')

    plt.subplot(3, 2, 5)
    # plt.plot(pro.mean(1))
    # plt.plot(pro.mean(1) + np.sqrt(var.mean(1)))
    # plt.plot(pro.mean(1) - np.sqrt(var.mean(1)))
    # plt.axvline((pro.mean(1) - np.sqrt(var.mean(1))).argmax())
    plt.plot((mid < 0).mean(1))
    plt.plot((hi < 0).mean(1))
    plt.plot((low < 0).mean(1))
    plt.axvline(((low < 0).mean(1)).argmax())
    plt.tight_layout()

    if isinstance(filename, str):
        fname = filename
    elif filename is None:
        fname = f'/home/victor/robustGP/2step_IVPC_{n_added_points}.png'
    else:
        fname = filename(n_added_points)
    plt.savefig(fname)
    plt.close()


branin2 = initialize_branin(10)
aIMSE_6 = criterion_2_aug(branin2, XY, [0.6])
aIMSE_3 = criterion_2_aug(branin2, XY, [0.3])
aIMSE_10 = criterion_2_aug(branin2, XY, [1.0])

m, s = branin2.predict(XY, return_std=True)
md, sd = branin2.predict_GPdelta(XY, alpha=2)

pro, var = pro_var_coverage(md, sd)

plt.subplot(3, 2, 1)
plt.title(r'GP prediction: $m_Z$')
plt.contourf(xmg, ymg, m.reshape(50, 50))
plt.plot(branin2.gp.X_train_[:, 0], branin2.gp.X_train_[:, 1], 'w.')

plt.subplot(3, 2, 2)
plt.title(r'GP std : $\sigma_Z$')
plt.contourf(xmg, ymg, s.reshape(50, 50))
plt.plot(branin2.gp.X_train_[:, 0], branin2.gp.X_train_[:, 1], 'w.')


plt.subplot(3, 2, 3)
plt.title(r'$m_{\Delta_{\alpha}}$')
plt.contourf(xmg, ymg, md.reshape(50, 50))
plt.plot(branin2.gp.X_train_[:, 0], branin2.gp.X_train_[:, 1], 'w.')

plt.subplot(3, 2, 4)
plt.title(r'$\sigma_{\Delta_{\alpha}}$')
plt.contourf(xmg, ymg, sd.reshape(50, 50))
plt.plot(branin2.gp.X_train_[:, 0], branin2.gp.X_train_[:, 1], 'w.')

plt.subplot(3, 2, 5)
plt.title(r'a-IMSE, $\tilde{\theta} = 0.6$')
plt.contourf(xmg, ymg, aIMSE_6.reshape(50, 50), vmin=0.027, vmax=0.104)
plt.plot(branin2.gp.X_train_[:, 0], branin2.gp.X_train_[:, 1], 'w.')
plt.axvline(0.6, color='r')
plt.subplot(3, 2, 6)
plt.title(r'a-IMSE, $\tilde{\theta} = 0.3$')
plt.contourf(xmg, ymg, aIMSE_10.reshape(50, 50), vmin=0.027, vmax=0.104)
plt.plot(branin2.gp.X_train_[:, 0], branin2.gp.X_train_[:, 1], 'w.')
plt.axvline(0.3, color='r')
plt.tight_layout()
plt.show()



plt.subplot(1, 2, 1)
plt.plot(pro.reshape(50, 50).mean(1))
plt.plot((pro + np.sqrt(var)).reshape(50, 50).mean(1))
plt.plot((pro - np.sqrt(var)).reshape(50, 50).mean(1))
plt.subplot(1, 2, 2)
plt.plot(var.reshape(50, 50).mean(1))
plt.show()


plt.subplot(1, 3, 1);plt.contourf(xmg, ymg, aIMSE_3.reshape(50,50), vmin=0.012, vmax=0.104)
plt.subplot(1, 3, 2);plt.contourf(xmg, ymg, aIMSE_6.reshape(50,50), vmin=0.012, vmax=0.104)
plt.subplot(1, 3, 3);plt.contourf(xmg, ymg, aIMSE_10.reshape(50,50), vmin=0.012, vmax=0.104)
plt.colorbar()
plt.show()
