# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numpy as np
from ot import dist, emd2, fused_gromov_wasserstein2, sinkhorn
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances


def distance_matrices_scalars(X_scalars1: np.ndarray, X_scalars2: np.ndarray = None):
    """
    Function to compute the distance matrices between scalar inputs. Scalars are compared using absolute values.

    Args:
        X_scalars1: A np.ndarray of shape (N, L) that give the L scalars for all the N inputs.
        X_scalars2: A np.ndarray of shape (N2, L) that give the L scalars for all the N2 inputs. If None, automatically takes embeddings2 = embeddings1.
    Returns:
        D (list): A list of length L where D[l] is the distance matrix of shape (N,N2) bewteen the l-th input scalars.
    """
    D = []
    if X_scalars2 is None:
        for j in range(X_scalars1.shape[1]):
            D.append(squareform(pdist(X_scalars1[:, j : j + 1])))
    else:
        for j in range(X_scalars1.shape[1]):
            D.append(
                pairwise_distances(X_scalars2[:, j : j + 1], X_scalars1[:, j : j + 1])
            )
    return D


def distance_matrices_swwl(embeddings1: np.ndarray, embeddings2: np.ndarray = None):
    """
    Function to compute the SWWL distance matrix between all pairs (embedding1, embedding2) for embedding1 in embeddings1 and embedding2 in embeddings2.
    If embeddings2 is None, distances between all pairs of train embeddings1 are computed.

    Args:
        embeddings1: SWWL graph embeddings given as a np.ndarray of shape (N, PQ).
        embeddings2: SWWL graph embeddings given as a np.ndarray of shape (N2, PQ). If None, automatically takes embeddings2 = embeddings1.
    Returns:
        D (list): A list containing a unique value which is the distance matrix of shape (N,N2).
    """
    if embeddings2 is None:
        return [squareform(pdist(embeddings1))]
    else:
        return [pairwise_distances(embeddings2, embeddings1)]


def distance_matrices_aswwl(
    num_wl_iterations: int, embeddings1: np.ndarray, embeddings2: np.ndarray = None
):
    """
    Function to compute the ASWWL distance matrices between all pairs (embedding1, embedding2) for embedding1 in embeddings1 and embedding2 in embeddings2.
    If embeddings2 is None, distances between all pairs of train embeddings1 are computed.
    For each iteration, a distance matrix is computed separately.

    Args:
        num_wl_iterations: The number of wl iterations H.
        embeddings1: ASWWL graph embeddings given as a np.ndarray of shape (N, (H+1)*PQ).
        embeddings2: ASWWL graph embeddings given as a np.ndarray of shape (N2, (H+1)*PQ). If None, automatically takes embeddings2 = embeddings1.
    Returns:
        D (list): A list containing a distance matrix of shape (N,N2) for each h in {0, ..., H+1}.
    """
    dim_embeddings_one_it = embeddings1.shape[-1] // (num_wl_iterations + 1)
    D = []

    if embeddings2 is None:
        for i in range(0, num_wl_iterations + 1):
            D.append(
                squareform(
                    pdist(
                        embeddings1[
                            :,
                            i * dim_embeddings_one_it : (i + 1) * dim_embeddings_one_it,
                        ]
                    )
                )
            )
    else:
        for i in range(0, num_wl_iterations + 1):
            D.append(
                pairwise_distances(
                    embeddings2[
                        :, i * dim_embeddings_one_it : (i + 1) * dim_embeddings_one_it
                    ],
                    embeddings1[
                        :, i * dim_embeddings_one_it : (i + 1) * dim_embeddings_one_it
                    ],
                )
            )
    return D


def distance_matrices_wwl(
    embeddings1: list,
    embeddings2: list = None,
    sinkhorn_ER: bool = False,
    sinkhorn_lambda: float = 1e-2,
):
    """
    Function to compute the WWL distance matrix between all pairs (embedding1, embedding2) for embedding1 in embeddings1 and embedding2 in embeddings2.
    If embeddings2 is None, distances between all pairs of train embeddings1 are computed.

    Args:
        embeddings1: Continuous WL embeddings given as a list of length N composed of arrays of shape (n_i, (H+1)*d).
        embeddings2: Continuous WL embeddings given as a list of length N2 composed of arrays of shape (n_i, (H+1)*d).. If None, automatically takes embeddings2 = embeddings1.
        sinkhorn_ER: Wether tu use Sinkhorn iterations to compute regularized Wasserstein distances. Otherwise, uses the classical Wasserstein distance.
        sinkhorn_lambda: The lambda parameter for Sinkhorn iterations (if sinkhorn_ER = True).
    Returns:
        D (list): A list containing a unique value which is the distance matrix of shape (N,N2).
    """

    if embeddings2 is None:
        N = len(embeddings1)
        D = np.zeros((N, N))
        for i in range(N):
            mu = embeddings1[i]
            for j in range(i + 1, N):
                nu = embeddings1[j]
                costs = dist(mu, nu, metric="euclidean")
                if sinkhorn_ER:
                    mat = sinkhorn(
                        np.ones(len(mu)) / len(mu),
                        np.ones(len(nu)) / len(nu),
                        costs,
                        sinkhorn_lambda,
                        numItermax=50,
                    )
                    D[i, j] = np.sum(np.multiply(mat, costs))
                    D[j, i] = D[i, j]
                else:
                    D[i, j] = emd2([], [], costs)
                    D[j, i] = D[i, j]
    else:
        N_train, N_test = len(embeddings1), len(embeddings2)
        D = np.zeros((N_test, N_train))
        for i in range(N_test):
            mu = embeddings2[i]
            for j in range(N_train):
                nu = embeddings1[j]
                costs = dist(mu, nu, metric="euclidean")
                if sinkhorn:
                    mat = sinkhorn(
                        np.ones(len(mu)) / len(mu),
                        np.ones(len(nu)) / len(nu),
                        costs,
                        sinkhorn_lambda,
                        numItermax=50,
                    )
                    D[i, j] = np.sum(np.multiply(mat, costs))
                else:
                    D[i, j] = emd2([], [], costs)
    return [D]


def distance_matrices_fgw(
    embeddings1: list, embeddings2: list = None, alphas: list = [0.5]
):
    """
    Function to compute the FGW distance matrix between all pairs (embedding1, embedding2) for embedding1 in embeddings1 and embedding2 in embeddings2.
    If embeddings2 is None, distances between all pairs of train embeddings1 are computed.
    If several alpha hyperparameters are given, we output distance matrices for all such parameters.

    Args:
        embeddings1: A list of length N composed of tuples (attributes, shortest path matrix).
        embeddings2: A list of length N composed of tuples (attributes, shortest path matrix). If None, automatically takes embeddings2 = embeddings1.
        alphas: A list of alpha parameters for the FGW distance.
    Returns:
        D (list): A list of size len(alphas). For i in len(alphas), D[i] is a list containing a unique value which is a distance matrix of shape (N,N2).
    """
    if embeddings2 is None:
        N = len(embeddings1)
        D = np.zeros((len(alphas), N, N))
        for i in range(N):
            # print(f"We are here: {i}")
            C_mu = embeddings1[i][1]
            X_mu = embeddings1[i][0]
            for j in range(i + 1, N):
                C_nu = embeddings1[j][1]
                X_nu = embeddings1[j][0]

                M = dist(X_mu, X_nu, metric="sqeuclidean")
                for a, alpha in enumerate(alphas):
                    D[a, i, j] = fused_gromov_wasserstein2(
                        M,
                        C_mu,
                        C_nu,
                        np.ones(X_mu.shape[0]) / X_mu.shape[0],
                        np.ones(X_nu.shape[0]) / X_nu.shape[0],
                        loss_fun="square_loss",
                        alpha=alpha,
                        verbose=False,
                        log=False,
                    )
                    D[a, j, i] = D[a, i, j]

        return [[d] for d in D]
    else:
        N_train, N_test = len(embeddings1), len(embeddings2)
        D = np.zeros((len(alphas), N_test, N_train))
        for i in range(N_test):
            C_mu = embeddings2[i][1]
            X_mu = embeddings2[i][0]
            for j in range(N_train):
                C_nu = embeddings1[j][1]
                X_nu = embeddings1[j][0]
                M = dist(X_mu, X_nu, metric="sqeuclidean")
                for a, alpha in enumerate(alphas):
                    D[a, i, j] = fused_gromov_wasserstein2(
                        M,
                        C_mu,
                        C_nu,
                        np.ones(X_mu.shape[0]) / X_mu.shape[0],
                        np.ones(X_nu.shape[0]) / X_nu.shape[0],
                        loss_fun="square_loss",
                        alpha=alpha,
                        verbose=False,
                        log=False,
                    )
    return [[d] for d in D]
