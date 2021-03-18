#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Enrichment:
    """Documentation for Enrichment"""

    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, gp):
        pass


class OptimEnrichment(Enrichment):
    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, gp):
        pass


class OneStepEnrichment(OptimEnrichment):
    """Documentation for OneStepEnrichment"""

    def __init__(self, bounds):
        super(OneStepEnrichment, self).__init__(bounds)

    def set_optim(self, optimiser, **params):
        if optimiser is None:
            self.optimiser = lambda cr: ac.optimize_with_restart(
                cr, self.bounds, **params
            )
        else:
            self.optimiser = lambda cr: optimiser(cr, self.bounds, **params)

    def set_criterion(self, criterion, maxi=False, **args):
        if maxi:
            self.criterion = lambda gp, X: -criterion(gp, X, **args)
        else:
            self.criterion = lambda gp, X: criterion(gp, X, **args)

    def run(self, gp):
        return self.optimiser(lambda X: self.criterion(gp, np.atleast_2d(X)))


class TwoStepEnrichment(Enrichment):
    """Documentation for TwoStepEnrichment"""

    # TODO: write enrichment
    def __init__(self, bounds):
        super(TwoStepEnrichment, self).__init__(bounds)
        self.bounds = bounds


class AKMCSEnrichment(Enrichment):
    """Documentation for AKMCSEnrichment"""

    def __init__(self, bounds):
        super(AKMCSEnrichment, self).__init__(bounds)

    def set_pdf(self, pdf, **args):
        self.pdf = lambda arg, X: pdf(arg, X, **args)

    def set_sampler(self, sampler, Nsamples, **args):
        self.sampler = sampler
        self.Nsamples = Nsamples

    def sample_candidates(self, admethod):
        return self.sampler(self.pdf, admethod, self.bounds, self.Nsamples, 100)

    def set_clustering(self, clustering, Kclusters):
        self.clustering = clustering
        self.Kclusters = Kclusters

    def run(self, admethod):
        samples = self.sample_candidates(admethod)
        sKmeans = self.clustering(self.Kclusters, samples)
        return sKmeans[0], sKmeans[1].cluster_centers_, samples
