# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import sys

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.packages import importr
from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern

numpy2ri.activate()
rgasp = importr("RobustGaSP")
base = importr("base")


def blockPrint():
    # Disable prints
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    # Restore prints
    sys.stdout = sys.__stdout__


class RGASP:
    """
    Robust Gaussian Process regression using tensor product kernels.
    Uses a python wrapper of https://github.com/cran/RobustGaSP/ .

    Attributes:
        kmodel: The rgasp model.
    """

    def __init__(
        self,
        D_matrices_train_train: list,
        y_train: np.ndarray,
        S_matrices_train_train: list = None,
        num_restarts: int = 3,
        verbose: int = 0,
    ):
        """
        Args:
            D_matrices_train_train: The list of train distance matrices that will be substituted in RBF kernels.
            y_train: The train output scalars.
            S_matrices_train_train: The optional list of scalar distance matrices that will be substituted in Matern52 kernels.
            num_restarts: The number of restarts for the optimization of hyperparameters.
            verbose: The verbose level. If verbose=2, prints all messages during the optimization.

        """
        # set_seed = robjects.r('set.seed') # Useless as RobustGaSP sets its own seeds...
        y_train = y_train.reshape((-1, 1))
        num_exp = len(D_matrices_train_train)
        num_matern = (
            len(S_matrices_train_train) if S_matrices_train_train is not None else 0
        )
        kernel_type = ["pow_exp"] * num_exp + ["matern_5_2"] * num_matern

        placeholder = np.zeros(
            (D_matrices_train_train[0].shape[0], num_exp + num_matern)
        )
        for i in range(num_exp):
            placeholder[:, i] = np.linspace(
                0, np.max(D_matrices_train_train[i]), D_matrices_train_train[i].shape[0]
            )
        for j in range(num_matern):
            placeholder[:, j + num_exp] = np.linspace(
                0, np.max(S_matrices_train_train[j]), S_matrices_train_train[j].shape[0]
            )

        if S_matrices_train_train is not None:
            R0 = D_matrices_train_train + S_matrices_train_train
        else:
            R0 = D_matrices_train_train

        if verbose < 2:
            blockPrint()
        self.kmodel = rgasp.rgasp(
            placeholder,
            y_train,
            kernel_type=kernel_type,
            R0=R0,
            isotropic=False,
            alpha=2.0,
            nugget_est=True,
            num_initial_values=num_restarts,
        )
        if verbose < 2:
            enablePrint()

    def predict(self, D_matrices_test_train: list, S_matrices_test_train: list = None):
        """
        Once the model is trained, use this function to predict scalar values for new test inputs and their associated uncertainties.

        Args:
            D_matrices_test_train: The list of distance matrices between test and train input graphs.
            S_matrices_test_train: The optional list of distance matrices between test and train scalar inputs.
        Returns:
            y_pred_test (np.ndarray): The array containing all predicted outputs.
            uq_test (np.ndarray): An array with 3 lines corresponding to the lower and upper bound of the 95% posterior credible interval and the standard deviation.
        """
        num_exp = len(D_matrices_test_train)
        num_matern = (
            len(S_matrices_test_train) if S_matrices_test_train is not None else 0
        )
        placeholder_test = np.random.randn(
            D_matrices_test_train[0].shape[0], num_exp + num_matern
        )
        if S_matrices_test_train is not None:
            R0_test = D_matrices_test_train + S_matrices_test_train
        else:
            R0_test = D_matrices_test_train
        y_pred_test, *uq_test = rgasp.predict(self.kmodel, placeholder_test, r0=R0_test)
        return y_pred_test, uq_test

    def get_parameters(self):
        # Returns a dict containing the range, variance, nugget and mean hyperparameters of the model.
        dollar = base.__dict__["@"]
        params = {
            "range": 1 / dollar(self.kmodel, "beta_hat"),
            "variance": dollar(self.kmodel, "sigma2_hat"),
            "nugget": dollar(self.kmodel, "nugget"),
            "mean": dollar(self.kmodel, "theta_hat"),
        }
        return params
