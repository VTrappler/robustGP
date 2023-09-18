import sys
sys.path.append('..')
import numpy as np
import tqdm
import argparse
import pyDOE
from sklearn.gaussian_process.kernels import Matern

from robustGP.SURmodel import AdaptiveStrategy
from robustGP.test_functions import branin_2d
import robustGP.tools as tools
import robustGP.acquisition.acquisition as ac


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


initial_design = np.genfromtxt("tmp/initial_design.txt")
branin = initialize_branin(initial_design)

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


def augmented_IVPC_Delta(
    arg, X, scenarios, integration_points, alpha, beta=0, verbose=True
):
    if callable(integration_points):
        int_points = integration_points()
    else:
        int_points = integration_points

    def function_(arg):
        m, va = arg.predict_GPdelta(int_points, alpha=alpha, beta=beta)
        s = np.sqrt(va)
        return ac.variance_probability_coverage((m, s), None, 0)

    return ac.augmented_design(arg, X, scenarios, function_, {}, verbose=verbose)


def save_at_each_iteration(filename, header=None):
    def decorator(function):
        def saving_function(XY, *args, **kwargs):
            print(filename)
            if header is not None:
                with open(filename, "w+") as fhandle:
                    to_write = f"#{header}\n"
                    print(f"-- {to_write}")
                    fhandle.write(to_write)
            for xy in tqdm.tqdm(XY):
                response = function(xy.reshape(-1, 2), *args, **kwargs)
                with open(filename, "a+") as fhandle:
                    to_write = np.atleast_2d(np.hstack([xy[0], xy[1], response]))
                    np.savetxt(fhandle, to_write, delimiter=",")
            return None

        return saving_function

    return decorator


integration_points = pyDOE.lhs(2, 50, criterion="maximin", iterations=50)

@save_at_each_iteration("tmp/aIMSE_Z.txt", header="aIMSE_Z, XY")
def function_aIMSE(XY):
    return augmented_IMSE(
        branin, XY, scenarios=None, integration_points=integration_points
    )


@save_at_each_iteration("tmp/aIMSE_Delta_1.txt", header="XYs, alpha=1.0")
def function_aIMSE_Delta_1(XY):
    return augmented_IMSE_Delta(
        branin,
        XY,
        scenarios=None,
        integration_points=integration_points,
        alpha=1.0,
    )

@save_at_each_iteration("tmp/aIVPC_Delta_1.txt", header="XYs, alpha=1.0")
def function_aIVPC_Delta_1(XY):
    return augmented_IVPC_Delta(
        branin,
        XY,
        scenarios=None,
        integration_points=integration_points,
        alpha=1.0,
    )

@save_at_each_iteration("tmp/aIVPC_Delta_2.txt", header="XYs, alpha=2.0")
def function_aIVPC_Delta_2(XY):
    return augmented_IVPC_Delta(
        branin,
        XY,
        scenarios=None,
        integration_points=integration_points,
        alpha=2.0,
    )


@save_at_each_iteration("tmp/aIVPC_Delta_3.txt", header="XYs, alpha=3.0")
def function_aIVPC_Delta_3(XY):
    return augmented_IVPC_Delta(
        branin,
        XY,
        scenarios=None,
        integration_points=integration_points,
        alpha=3.0,
    )

evaluate_IVPC = [function_aIVPC_Delta_1, function_aIVPC_Delta_2, function_aIVPC_Delta_3]

# np.savetxt('tmp/initial_design.txt', branin.gp.X_train_)
# print("aIMSE_Z ----")
# function_aIMSE(XY)
# print("aIMSE_Δ, 1.0----")
# function_aIMSE_Delta_1(XYs)


for i, eval_func in enumerate(evaluate_IVPC):
    print(f"aIVPC_Δ, α={i+1}.0 ----")
    eval_func(XYs)