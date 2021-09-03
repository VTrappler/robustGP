#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports: General tools
import numpy as np
import matplotlib.pyplot as plt
import scipy

from matplotlib import cm
import matplotlib.colors
from functools import partial

import pyDOE
# Imports: scikit-learn
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans

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
    'figure.figsize': [6, 4],
    'savefig.dpi': 400
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
    else:
        initial_design = initial_design
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


def prob_coverage(m, s, T):
    return scipy.stats.norm.cdf((T - m) / s)


def sampler(Nsamples, branin, T):
    n = 0
    samples = np.empty((0, 2))
    while n < Nsamples:
        cands = scipy.random.uniform(0, 1, 2 * 100).reshape(100, 2)
        m, v = branin.predict_GPdelta(cands, alpha=1.3)
        pi = prob_coverage(m, np.sqrt(v), T)
        for x, pr in zip(cands, np.logical_and((pi <= 0.975), (pi >= 0.025))):
            if pr:
                samples = np.vstack([samples, x])
        n = len(samples)
    return samples


def cluster_and_closest(samples, ncluster=8):
    kmeans = KMeans(n_clusters=ncluster).fit(samples)
    closest_samples = np.empty((ncluster, 2))
    for i, cen in enumerate(kmeans.cluster_centers_):
        closest_samples[i] = samples[np.sum((samples - cen)**2, 1).argmin()]
    return kmeans.labels_, kmeans.cluster_centers_, closest_samples


def adjustment(arg, closest_samples):
    X1, X2 = arg.separate_input(closest_samples)
    to_add = np.empty((len(X1), 2))
    for i, (x1, x2) in enumerate(zip(X1, X2)):
        set_input = arg.create_input(x2)
        x1_star = arg.get_conditional_minimiser(x2).x
        _, _, cov = robustGP.gptools.gp_to_delta(
            arg,
            set_input(x1).flatten(),
            set_input(x1_star).flatten(),
            alpha=1.3,
            beta=0,
            return_var=True,
        )
        if cov[0] > cov[1]:
            to_add[i] = closest_samples[i]
        else:
            to_add[i] = np.array([x1_star, x2]).squeeze()
    return to_add


def left_plot(branin, mean, std, new=0):
    plt.contourf(xmg, ymg, mean.reshape(50, 50))
    if new == 0:
        plt.scatter(branin.gp.X_train_[:, 0], branin.gp.X_train_[:, 1], c="white", s=3, marker='x')
    else:
        plt.scatter(branin.gp.X_train_[:, 0], branin.gp.X_train_[:, 1], c="white", s=3, marker='x')
        # plt.plot(branin.gp.X_train_[new:, 0], branin.gp.X_train_[new:, 1], 'y')
    plt.contour(xmgl, ymgl, Delta_true, levels=[0], colors='yellow')
    plt.contour(xmg, ymg, prob_coverage(mean, std, 0).reshape(50, 50),
                levels=[0.5], colors='lime', alpha=0.9)
    plt.plot(np.nan, np.nan, color='yellow', label=r'$J = \alpha J^*$')
    plt.plot(np.nan, np.nan, color='lime', label=r'$m_{\Delta_{\alpha}} = 0$')
    leg = plt.legend(loc='upper right')
    for text in leg.get_texts():
        plt.setp(text, color = 'w')

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
    plt.title(r'GP prediction')


cmap = matplotlib.colors.ListedColormap([cm.Pastel1(0), cm.Pastel1(1)])


def right_plot_bck(pi, title=''):
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
    margin = np.logical_and(pi > 0.025, pi < 0.975)
    plt.contourf(xmg, ymg, margin.reshape(50, 50),
                 cmap=cmap)
    plt.title(title)


if __name__ == '__main__':
    branin = initialize_branin(10)
    Jtrue = branin.function(XYl).reshape(500, 500)
    Jstar = Jtrue.min(0)
    alpha = 1.3
    Delta_true = Jtrue - alpha * Jstar
    mean, std2 = branin.predict_GPdelta(XY, alpha=alpha)
    std = np.sqrt(std2)
    pi = prob_coverage(mean, std, 0)
    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'Margin of uncertainty')


    plt.tight_layout()
    plt.savefig(graphics_folder + 'AKMCS_1.png')
    plt.close()

    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'Samples')
    samples = sampler(800, branin, 0)
    plt.scatter(samples[:, 0], samples[:, 1], c='white', s=5)
    plt.tight_layout()
    plt.savefig(graphics_folder + 'AKMCS_2.png')
    plt.close()

    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'Clustering using KMeans')
    labels, center, closest = cluster_and_closest(samples, 8)
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cm.tab10, s=5)
    to_add = adjustment(branin, closest)
    # plt.plot(to_add[:, 0], to_add[:, 1], '*', c='yellow')

    plt.tight_layout()
    plt.savefig(graphics_folder + 'AKMCS_3.png')
    plt.close()


    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'Clustering using KMeans')
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cm.tab10, s=5)
    plt.plot(center[:, 0], center[:, 1], '*', c='white')
    plt.plot(closest[:, 0], closest[:, 1], '*', c='red')
    to_add = adjustment(branin, closest)
    plt.plot(to_add[:, 0], to_add[:, 1], '*', c='yellow')

    plt.tight_layout()
    plt.savefig(graphics_folder + 'AKMCS_4.png')
    plt.close()

    branin.add_points(to_add)
    mean, std2 = branin.predict_GPdelta(XY, alpha=alpha)
    std = np.sqrt(std2)
    pi = prob_coverage(mean, std, 0)

    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std, new=8)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'AKMCS')
    samples = sampler(800, branin, 0)
    labels, center, closest = cluster_and_closest(samples, 8)
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cm.tab10, s=5)
    plt.plot(center[:, 0], center[:, 1], '*', c='white')
    plt.plot(closest[:, 0], closest[:, 1], '*', c='red')
    to_add = adjustment(branin, closest)
    plt.plot(to_add[:, 0], to_add[:, 1], '*', c='yellow')


    plt.tight_layout()
    plt.savefig(graphics_folder + 'AKMCS_5.png')
    plt.close()

    branin.add_points(to_add)
    mean, std2 = branin.predict_GPdelta(XY, alpha=alpha)
    std = np.sqrt(std2)
    pi = prob_coverage(mean, std, 0)

    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std, new=8)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'AKMCS')

    plt.tight_layout()
    samples = sampler(800, branin, 0)
    labels, center, closest = cluster_and_closest(samples, 8)
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cm.tab10, s=5)
    plt.plot(center[:, 0], center[:, 1], '*', c='white')
    plt.plot(closest[:, 0], closest[:, 1], '*', c='red')
    to_add = adjustment(branin, closest)
    plt.plot(to_add[:, 0], to_add[:, 1], '*', c='yellow')

    plt.savefig(graphics_folder + 'AKMCS_6.png')
    plt.close()


    branin.add_points(to_add)
    mean, std2 = branin.predict_GPdelta(XY, alpha=alpha)
    std = np.sqrt(std2)
    pi = prob_coverage(mean, std, 0)

    # -------
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std, new=8)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'AKMCS')

    plt.tight_layout()
    samples = sampler(800, branin, 0)
    labels, center, closest = cluster_and_closest(samples, 8)
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cm.tab10, s=5)
    plt.plot(center[:, 0], center[:, 1], '*', c='white')
    plt.plot(closest[:, 0], closest[:, 1], '*', c='red')
    to_add = adjustment(branin, closest)
    plt.plot(to_add[:, 0], to_add[:, 1], '*', c='yellow')

    plt.savefig(graphics_folder + 'AKMCS_7.png')
    plt.close()

    for i in range(5):
        print('finally there')
        branin.add_points(to_add)
        samples = sampler(800, branin, 0)
        labels, center, closest = cluster_and_closest(samples, 5)
        to_add = adjustment(branin, closest)

    mean, std2 = branin.predict_GPdelta(XY, alpha=alpha)
    std = np.sqrt(std2)
    pi = prob_coverage(mean, std, 0)
    plt.subplot(1, 2, 1)
    left_plot(branin, mean, std, new=8)
    plt.subplot(1, 2, 2)
    right_plot_bck(pi, 'AKMCS')
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cm.tab10, s=5)
    plt.plot(center[:, 0], center[:, 1], '*', c='white')
    plt.plot(closest[:, 0], closest[:, 1], '*', c='red')
    to_add = adjustment(branin, closest)
    plt.plot(to_add[:, 0], to_add[:, 1], '*', c='yellow')
    plt.tight_layout()
    plt.savefig(graphics_folder + 'AKMCS_8.png')

    plt.close()
