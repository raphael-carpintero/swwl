# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import itertools
import os
import sys

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.svm import SVC

from .kernel import custom_kernel_exp, custom_kernel_just_scalars, custom_kernel_rbf


class SVC_precomputed_distances:
    """
    Support vector classification using tensor product kernels. Inputs are precomputed distance matrices.
    """

    def __init__(
        self,
        D_matrices_train_train: list,
        y_train: np.ndarray,
        S_matrices_train_train: list = None,
        type_kernel: str = "exp",
        svm_param_grid: dict = {"C": np.logspace(-3, 3, num=10)},
        gammas: np.ndarray = np.logspace(-3, 3, num=10),
        scalar_lengthscales: np.ndarray = np.logspace(-3, 3, num=10),
        nugget: float = 1e-7,
        verbose: int = 0,
        cv_splits: int = 5,
        cv_seed: int = 0,
    ):
        """
        Args:
            D_matrices_train_train: The list of train distance matrices that will be substituted in the exp of rbf kernels.
            y_train: The train output scalars.
            S_matrices_train_train: The optional list of scalar distance matrices that will be substituted in Matern52 kernels.
            type_kernel: The type of distance substitution kernel ('exp' of 'rbf').
            svm_param_grid: The dict containing the SVM param grid. By default, only the 'C' parameter is tuned.
            gammas: The array of possible values for precision parameter of the exponential/RBF kernel.
            scalar_lengthscales: The array of possible values for any lengthscale parameter in a Matern52 kernel for scalars.
            nugget: The default nugget parameter.
            verbose: The verbose level. If verbose=2, prints all validation scores during the hyperparameter optimization.
            cv_splits: The number of splits for the inner cross validation loop.
            cv_seed: The seed used to split train/valid sets using a StratifiedKFold.
        """

        self.D_matrices_train_train = D_matrices_train_train
        self.S_matrices_train_train = S_matrices_train_train
        self.num_exp = len(D_matrices_train_train)
        self.num_matern = (
            len(S_matrices_train_train) if S_matrices_train_train is not None else 0
        )
        self.y_train = y_train.ravel()
        self.custom_kernel = (
            custom_kernel_rbf if type_kernel == "rbf" else custom_kernel_exp
        )
        # kernel_param_grid = Precisions (gammas) + Lengthscales + variance + nugget
        # where variance and nugget are fixed here.
        kernel_param_grid = (
            [gammas] * self.num_exp
            + [scalar_lengthscales] * self.num_matern
            + [[1.0]]
            + [[nugget]]
        )
        self.custom_grid_search_cv(
            svm_param_grid, kernel_param_grid, cv_splits, verbose=verbose, seed=cv_seed
        )

    def predict(self, D_matrices_test_train: list, S_matrices_test_train: list = None):
        """
        Once the model is trained, use this function to predict scalar outputs for new test inputs.

        Args:
            D_matrices_test_train: The list of distance matrices between test and train input graphs.
            S_matrices_test_train: The optional list of distance matrices between test and train scalar inputs.
        Returns:
            y_pred_test (np.ndarray): The array containing all predicted outputs.
            uq_test (None): No confidence intervals for the predictions.
        """
        K = self.custom_kernel(
            self.best_kernel_params,
            D_matrices_test_train,
            S_matrices=S_matrices_test_train,
            add_nugget=False,
        )
        y_pred_test = self.kmodel.predict(K)
        return y_pred_test, None

    def custom_grid_search_cv(
        self,
        svm_param_grid: dict,
        kernel_param_grid: list,
        n_splits: int,
        verbose: int = 0,
        seed: int = 0,
    ):
        """
        Lengthscale/precision parameters as well as the SVM parameters are tuned using a custom grid search based on the sklearn grid search.
        A Stratified KFold is used to separate train/valid sets.

        Args:
            svm_param_grid: The dict containing the SVM param grid. By default, only the 'C' parameter is tuned.
            kernel_param_grid: The hyperparameters of the kernel (lengthscales, variance, nugget) in the form of a list of liste of possible values for each parameter.
            n_splits: The number of splits for the inner cross validation loop.
            verbose: The verbose level. If verbose=2, prints all validation scores during the hyperparameter optimization.
            seed: The seed used to split train/valid sets using a StratifiedKFold.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = SVC(kernel="precomputed")
        results = []
        for train_index, valid_index in cv.split(
            self.D_matrices_train_train[0], self.y_train
        ):
            split_results = []
            all_params = []

            # Two loops for the hyperparameter grid: 1- kernel parameters 2- SVM parameters.
            for kparams in itertools.product(*kernel_param_grid):
                kparams = list(kparams)
                kernel_params = {
                    "precision_embeddings": kparams[: self.num_exp],
                    "variance": kparams[self.num_exp + self.num_matern],
                    "nugget": kparams[self.num_exp + self.num_matern + 1],
                }
                if self.S_matrices_train_train is not None:
                    kernel_params["lengthscales_scalars"] = kparams[
                        self.num_exp : self.num_exp + self.num_matern
                    ]

                K = self.custom_kernel(
                    kernel_params,
                    self.D_matrices_train_train,
                    S_matrices=self.S_matrices_train_train,
                    add_nugget=True,
                )

                for p in list(ParameterGrid(svm_param_grid)):
                    sc = _fit_and_score(
                        clone(model),
                        K,
                        self.y_train,
                        scorer=make_scorer(accuracy_score),
                        train=train_index,
                        test=valid_index,
                        verbose=0,
                        parameters=p,
                        fit_params=None,
                    )
                    split_results.append(sc["test_scores"])
                    all_params.append((kernel_params, p))
                    if verbose == 2:
                        print(f"{kernel_params}, C={p} -> Acc={sc}")
            results.append(split_results)

        results = np.array(results)
        final_results = results.mean(axis=0)
        best_idx = np.argmax(final_results)
        best_model = clone(model).set_params(**all_params[best_idx][1])
        K_train = self.custom_kernel(
            all_params[best_idx][0],
            self.D_matrices_train_train,
            S_matrices=self.S_matrices_train_train,
            add_nugget=True,
        )
        best_model.fit(K_train, self.y_train)
        self.best_cv_result = final_results[best_idx]
        self.kmodel = best_model
        self.best_kernel_params = all_params[best_idx][0]
        self.best_svm_params = all_params[best_idx][1]

    def get_best_cv_result(self):
        return self.best_cv_result

    def get_parameters(self):
        return self.best_kernel_params, self.best_svm_params


class SVC_precomputed_Gram:
    """
    Support vector classification using tensor product kernels. Inputs are precomputed Gram matrices.
    """

    def __init__(
        self,
        K_matrix_train_train: np.ndarray,
        y_train: np.ndarray,
        S_matrices_train_train: list = None,
        svm_param_grid: dict = {"C": np.logspace(-3, 3, num=10)},
        scalar_lengthscales: np.ndarray = np.logspace(-3, 3, num=10),
        nugget: float = 1e-7,
        verbose: int = 0,
        cv_splits: int = 5,
        cv_seed: int = 0,
    ):
        """
        Args:
            K_matrix_train_train: The train Gram matrix.
            y_train: The train output scalars.
            S_matrices_train_train: The optional list of scalar distance matrices that will be substituted in Matern52 kernels.
            svm_param_grid: The dict containing the SVM param grid. By default, only the 'C' parameter is tuned.
            scalar_lengthscales: The array of possible values for any lengthscale parameter in a Matern52 kernel for scalars.
            nugget: The default nugget parameter.
            verbose: The verbose level. If verbose=2, prints all validation scores during the hyperparameter optimization.
            cv_splits: The number of splits for the inner cross validation loop.
            cv_seed: The seed used to split train/valid sets using a StratifiedKFold.
        """
        self.K_matrix_train_train = K_matrix_train_train
        self.S_matrices_train_train = S_matrices_train_train
        self.y_train = y_train.ravel()

        self.num_matern = (
            len(S_matrices_train_train) if S_matrices_train_train is not None else 0
        )
        self.y_train = y_train.ravel()
        kernel_param_grid = (
            [scalar_lengthscales] * self.num_matern + [[1.0]] + [[nugget]]
        )

        self.custom_grid_search_cv(
            svm_param_grid, kernel_param_grid, cv_splits, verbose=verbose, seed=cv_seed
        )

    def predict(
        self, K_matrix_test_train: np.ndarray, S_matrices_test_train: list = None
    ):
        """
        Once the model is trained, use this function to predict scalar outputs for new test inputs.

        Args:
            K_matrix_test_train: The Gram matrix between test and train input graphs.
            S_matrices_test_train: The optional list of distance matrices between test and train scalar inputs.
        Returns:
            y_pred_test (np.ndarray): The array containing all predicted outputs.
            uq_test (None): No confidence intervals for the predictions.
        """
        if S_matrices_test_train is not None:
            K = custom_kernel_just_scalars(
                self.best_kernel_params, S_matrices_test_train, add_nugget=False
            )
            K = K * K_matrix_test_train
        else:
            K = K_matrix_test_train

        y_pred_test = self.kmodel.predict(K)
        return y_pred_test, None

    def custom_grid_search_cv(
        self,
        svm_param_grid: dict,
        kernel_param_grid: list,
        n_splits: int,
        verbose: int = 0,
        seed: int = 0,
    ):
        """
        SVM parameters are tuned using a custom grid search based on the sklearn grid search.
        If there are scalars inputs, their associated lengthscale parameters are also tuned.
        A Stratified KFold is used to separate train/valid sets.

        Args:
            svm_param_grid: The dict containing the SVM param grid. By default, only the 'C' parameter is tuned.
            kernel_param_grid: The hyperparameters of the kernel (lengthscales, variance, nugget) in the form of a list of liste of possible values for each parameter.
            n_splits: The number of splits for the inner cross validation loop.
            verbose: The verbose level. If verbose=2, prints all validation scores during the hyperparameter optimization.
            seed: The seed used to split train/valid sets using a StratifiedKFold.
        """

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = SVC(kernel="precomputed")
        results = []
        for train_index, test_index in cv.split(
            self.K_matrix_train_train, self.y_train
        ):
            split_results = []
            all_params = []  # list of dict (the same for every split).

            # run over the kernels first.
            for kparams in itertools.product(*kernel_param_grid):
                kparams = list(kparams)
                kernel_params = {
                    "variance": kparams[self.num_matern],
                    "nugget": kparams[self.num_matern + 1],
                }
                if self.S_matrices_train_train is not None:
                    kernel_params["lengthscales_scalars"] = kparams[: self.num_matern]
                    K = custom_kernel_just_scalars(
                        kernel_params, self.S_matrices_train_train, add_nugget=True
                    )
                    K = K * self.K_matrix_train_train
                else:
                    K = self.K_matrix_train_train

                for p in list(ParameterGrid(svm_param_grid)):
                    sc = _fit_and_score(
                        clone(model),
                        K,
                        self.y_train,
                        scorer=make_scorer(accuracy_score),
                        train=train_index,
                        test=test_index,
                        verbose=0,
                        parameters=p,
                        fit_params=None,
                    )
                    split_results.append(sc["test_scores"])
                    all_params.append((kernel_params, p))
                    if verbose == 2:
                        print(f"{kernel_params}, C={p} -> Acc={sc}")
            results.append(split_results)

        results = np.array(results)
        final_results = results.mean(axis=0)
        best_idx = np.argmax(final_results)
        best_model = clone(model).set_params(**all_params[best_idx][1])

        self.best_kernel_params = all_params[best_idx][0]
        self.best_svm_params = all_params[best_idx][1]
        self.best_cv_result = final_results[best_idx]

        if self.S_matrices_train_train is not None:
            K_train = custom_kernel_just_scalars(
                self.best_kernel_params, self.S_matrices_train_train, add_nugget=True
            )
            K_train = K_train * self.K_matrix_train_train
        else:
            K_train = self.K_matrix_train_train
        best_model.fit(K_train, self.y_train)
        self.kmodel = best_model

    def get_best_cv_result(self):
        return self.best_cv_result

    def get_parameters(self):
        return self.best_kernel_params, self.best_svm_params
