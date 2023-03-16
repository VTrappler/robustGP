import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import tqdm

sys.path.append("..")
from functools import partial
import pyDOE
from sklearn.gaussian_process.kernels import Matern

from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d
import robustGP.tools as tools
import robustGP.gptools
import robustGP.acquisition.acquisition as ac
import robustGP.enrichment.Enrichment as enrich
import robustGP.optimisers as opt


def initialize_branin(initial_design=None):
    """
    Create new instance of AdaptiveStrategy of the Branin 2d function
    with LHS as initial design
    """
    NDIM = 2
    bounds = np.asarray([(0, 1)] * NDIM)
    if initial_design is None:
        initial_design = 5 * NDIM
    if isinstance(initial_design, int):
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
xs, ys = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
(XYs, (xmgs, ymgs)) = tools.pairify((xs, ys))

xsmall, ysmall = np.linspace(0, 1, 4), np.linspace(0, 1, 4)
(XYsmall, (xmgsmall, ymgsmall)) = tools.pairify((xsmall, ysmall))


branin = initialize_branin(XYsmall)


def augmented_IMSE(arg, X, scenarios, integration_points, verbose=True):
    if callable(integration_points):
        int_points = integration_points()
    else:
        int_points = integration_points

    def function_(arg):
        m, sd = arg.predict(int_points, return_std=True)
        return sd**2

    return ac.augmented_design(arg, X, scenarios, function_, {}, verbose=verbose)


def augmented_IMSE_Delta(
    arg, X, scenarios, integration_points, alpha, beta=0, verbose=True
):
    if callable(integration_points):
        int_points = integration_points()
    else:
        int_points = integration_points

    def function_(arg):
        m, va = arg.predict_GPdelta(int_points, alpha=alpha, beta=beta)
        return va

    return ac.augmented_design(arg, X, scenarios, function_, {}, verbose=verbose)


def save_at_each_iteration(filename):
    def decorator(function):
        def saving_function(XY, *args, **kwargs):
            for xy in tqdm.tqdm(XY):
                response = function(xy.reshape(-1, 2), *args, **kwargs)
                with open(filename, "a+") as fhandle:
                    to_write = np.atleast_2d(np.hstack([xy[0], xy[1], response]))
                    # print(f"{xy[0]}, {xy[1]}, {response}")
                    np.savetxt(fhandle, to_write, delimiter=",")
            return None

        return saving_function

    return decorator


integration_points = pyDOE.lhs(2, 100, criterion="maximin", iterations=50)


@save_at_each_iteration("aIMSE.txt")
def function_aIMSE(XY):
    return augmented_IMSE(
        branin, XY, scenarios=None, integration_points=integration_points
    )


@save_at_each_iteration("aIMSE_Delta.txt")
def function_aIMSE_Delta(XY):
    return augmented_IMSE_Delta(
        branin, XY, scenarios=None, integration_points=integration_points, alpha=2
    )


# )
# aIMSE_Delta = augmented_IMSE_Delta(
#     branin, XYs, scenarios=None, integration_points=integration_points, alpha=2
# )
if __name__ is "__main__":
    function_aIMSE(XYs)
    function_aIMSE_Delta(XYs)
