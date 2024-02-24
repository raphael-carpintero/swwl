# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numpy as np
from sklearn.gaussian_process.kernels import Matern


def matern(D: np.ndarray, ls: float = 1.0):
    # If D is a NxN matrix, returns the NxN matrix where the Matern52 function with lengthscale l has been applied elementwise.
    return Matern(nu=5 / 2, length_scale=ls)(0, D.ravel()[:, np.newaxis]).reshape(
        D.shape
    )


def custom_kernel_exp(
    hparams: dict, D_matrices: list, S_matrices: list = None, add_nugget: bool = True
):
    """
    Given H distance matrices D_1, ...,D_H and L scalar matrices S_1, ..., S_L,
    the precision parameters p_1, ..., p_H for the exponential kernels,
    the Matern52 lengthscale parameters l_1, ..., l_L, and the optional nugget eps,
    this function returns the tensorized kernel
    Prod_i exp(-p_i D_i)  Prod_j Matern52(S_j |l_j) + nugget*I
    where all operations are performed elementwise.
    """
    K = np.ones((D_matrices[0].shape))
    for i in range(len(D_matrices)):
        K = K * np.exp(-hparams["precision_embeddings"][i] * D_matrices[i])
    if S_matrices is not None:
        for scal in range(len(S_matrices)):
            K = K * matern(S_matrices[scal], l=hparams["lengthscales_scalars"][scal])
    K = hparams["variance"] * K
    if add_nugget:
        K = K + hparams["nugget"] * np.eye(len(K))
    return K


def custom_kernel_rbf(
    hparams: dict, D_matrices: list, S_matrices: list = None, add_nugget: bool = True
):
    """
    Given H distance matrices D_1, ...,D_H and L scalar matrices S_1, ..., S_L,
    the precision parameters p_1, ..., p_H for the RBF kernels,
    the Matern52 lengthscale parameters l_1, ..., l_L, and the optional nugget eps,
    this function returns the tensorized kernel
    Prod_i exp(-p_i D_i^2) * Prod_j Matern52(S_j |l_j) + nugget*I
    where all operations are performed elementwise.
    """
    K = np.ones((D_matrices[0].shape))
    for i in range(len(D_matrices)):
        K = K * np.exp(-hparams["precision_embeddings"][i] * D_matrices[i] ** 2)
    if S_matrices is not None:
        for scal in range(len(S_matrices)):
            K = K * matern(S_matrices[scal], l=hparams["lengthscales_scalars"][scal])
    K = hparams["variance"] * K
    if add_nugget:
        K = K + hparams["nugget"] * np.eye(len(K))
    return K


def custom_kernel_just_scalars(hparams, S_matrices, add_nugget=True):
    """
    Given H Gram matrices K_1, ...,K_H and L scalar matrices S_1, ..., S_L,
    the Matern52 lengthscale parameters l_1, ..., l_L, and the optional nugget eps,
    this function returns the tensorized kernel
    Prod_i K_i * Prod_j Matern52(S_j |l_j) + nugget*I
    where all operations are performed elementwise.
    """
    K = np.ones((S_matrices[0].shape))
    for scal in range(len(S_matrices)):
        K = K * matern(S_matrices[scal], l=hparams["lengthscales_scalars"][scal])
    K = hparams["variance"] * K
    if add_nugget:
        K = K + hparams["nugget"]
    return K
