#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import robustGP.optimisers as opt

import scipy.stats


class Enrichment:
    """Documentation for Enrichment"""

    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, gp):
        pass


class InfillEnrichment(Enrichment):
    def __init__(self, bounds):
        super(InfillEnrichment, self).__init__(bounds)

    # def set_infill_criterion


class MonteCarloEnrich(InfillEnrichment):
    def __init__(self, dim, bounds, sampler):
        super(MonteCarloEnrich, self).__init__(bounds)
        self.sampler = sampler
        self.dim = dim

    def run(self, gp):
        return scipy.stats.uniform.rvs(size=(1, self.dim)), "MC"


class OptimEnrichment(Enrichment):
    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, gp):
        pass


class OneStepEnrichment(OptimEnrichment):
    """Documentation for OneStepEnrichment"""

    def __init__(self, bounds):
        super(OneStepEnrichment, self).__init__(bounds)
        self.set_optim(None)

    def set_optim(self, optimiser, **params):
        if optimiser is None:
            self.optimiser = lambda cr: opt.optimize_with_restart(
                cr, self.bounds, **params
            )
        else:
            self.optimiser = lambda cr: optimiser(cr, **params)

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

    def set_optim1(self, optimiser, **params):
        self.optimiser1 = lambda cr: optimiser(cr, **params)

    def set_optim2(self, optimiser, **params):
        if optimiser is None:
            self.optimiser2 = lambda cr: opt.optimize_with_restart(
                cr, self.bounds, **params
            )
        else:
            self.optimiser2 = lambda cr: optimiser(cr, self.bounds, **params)

    def set_criterion_step1(self, criterion, maxi=False, **args):
        if maxi:
            self.criterion1 = lambda gp, X: -criterion(gp, X, **args)
        else:
            self.criterion1 = lambda gp, X: criterion(gp, X, **args)

    def set_criterion_step2(self, criterion, maxi=False, **args):
        if maxi:
            self.criterion2 = lambda gp, X, Xnext: -criterion(gp, X, Xnext, **args)
        else:
            self.criterion2 = lambda gp, X, Xnext: criterion(gp, X, Xnext, **args)

    def run_stage1(self, gp):
        return self.optimiser1(lambda X: self.criterion1(gp, np.atleast_2d(X)))

    def run_stage2(self, gp, Xnext):
        return self.optimiser2(lambda X: self.criterion2(gp, np.atleast_2d(X), Xnext))

    def run(self, gp):
        Xnext = self.run_stage1(gp)[0]
        return self.optimiser2(lambda X: self.criterion2(gp, np.atleast_2d(X), Xnext))


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
