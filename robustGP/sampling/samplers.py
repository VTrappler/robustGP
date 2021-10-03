#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans


def sampling_from_indicator(indicator, gp, bounds, Nsamples, Ncandidates, **args):
    samples = np.empty((Nsamples, len(bounds)))
    ns = 0
    while ns < Nsamples:
        cands = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(Ncandidates, len(bounds)))
        ss = filter(lambda x: indicator(gp, np.atleast_2d(x), **args), cands)
        for i, s in enumerate(ss):
            try:
                samples[ns + i, :] = s
            except IndexError:
                break
        ns = i + ns
    return samples


def clustering(Kclusters, samples, **argskmeans):
    kmean = KMeans(n_clusters=Kclusters).fit(samples, **argskmeans)
    closest = np.empty((Kclusters, samples.shape[1]))
    for i, kmcenter in enumerate(kmean.cluster_centers_):
        closest[i] = samples[np.sum((samples - kmcenter) ** 2, 1).argmin()]
    return closest, kmean

