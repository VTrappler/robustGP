#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np



class Enrichment():
    """Documentation for Enrichment

    """
    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, gp):
        pass


class OneStepEnrichment(Enrichment):
    """Documentation for OneStepEnrichment

    """
    def __init__(self, bounds):
        super(OneStepEnrichment, self).__init__(bounds)

    def set_optim(self, optimiser, **params):
        if optimiser is None:
            self.optimiser = lambda cr: ac.optimize_with_restart(cr, self.bounds, **params)
        else:
            self.optimiser = lambda cr: optimiser(cr, self.bounds, **params)

    def set_criterion(self, criterion, maxi=False, **args):
        if maxi:
            self.criterion = lambda gp, X: -criterion(gp, X, **args)
        else:
            self.criterion = lambda gp, X: criterion(gp, X, **args)

       
    def run(self, gp):
        return self.optimiser(
            lambda X: self.criterion(gp, np.atleast_2d(X))
        )
        
class TwoStepEnrichment(Enrichment):
    """Documentation for TwoStepEnrichment

    """
    # TODO: write enrichment
    def __init__(self, args):
        super(TwoStepEnrichment, self).__init__()
        self.args = args
        

class AKMCSEnrichment(Enrichment):
    """Documentation for AKMCSEnrichment

    """
    def __init__(self, sampler):
        super(AKMCSEnrichment, self).__init__()
        self.sampler = sampler
        
        
        



