# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

""" Main file to compute distance/Gram matrices for the regression datasets and (optionally) run the Gaussian process regression.

* Notes about distance/Gram matrix computation:
    - cannot be performed for "wwl" due to the huge computation time. Please use wwl_meshes_parallelized_step{1,2,3}.py instead.
    
* Notes about Gaussian process regression:
    - done with a python wrapper of the R package RobustGaSP (using rpy2),
    - determinitsic optimization of the hyperparameters,
    - does not work for kernel methods that manages Gram matrices instead of distances (like "propag"),
    - please use run_gp_regression.R after precomputing distance/Gram matrices with this file in order to reproduce the experiments.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from datasets_regression import (
    available_mesh_datasets,
    load_mesh_dataset,
    load_mesh_dataset_grakel,
    load_scalars,
)
from sklearn.metrics import mean_squared_error, r2_score
from utils import *

from graph_gp.distances import *
from graph_gp.distances import distance_matrices_aswwl, distance_matrices_swwl
from graph_gp.embeddings import compute_graph_embeddings
from graph_gp.encoders import ASWWL_Encoder, SWWL_Encoder
from graph_gp.models import RGASP

DEBUG_N_TRAIN, DEBUG_N_TEST = None, None
# For debugging purpose, you can try the methods with a few train/test samples.


def compute_distance_matrices(
    dataset_name: str,
    kernel: str,
    hparams: dict,
    config_params: dict,
    seed: int,
    compute_scalar_matrices: bool = True,
):
    """
    Function to load a dataset and compute the distance matrices between all input graphs for a given kernel.
    This method is only used for kernels that require the computation of a distance matrix. Use compute_gram_matrices for other kernels.
    The hparams dict should contain only one hyperparameter value, and the method outputs only the distance matrix (matrices) for this hyperparameter.
    Remark: wwl is not supported due to time/memory issues.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel. Must be in ['swwl', 'aswwl']
        hparams: The hyperparameter dict (eg {"num_wl_iterations": [3], "step_wl": 0}).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the embedding/distance step.
        compute_scalar_matrices: Whether to compute all distance matrices between scalar inputs or not.
    Returns:
        D_matrices (list): A list of np.ndarray of shape (N,N) where N=N_train+N_test. If the kernel is not asswl, this list is of size 1.
        S_matrices (list): A list of L np.ndarray of shape (N,N) where L is the number of input scalars.
        times (dict): The times needed to load the dataset, compute the embeddings and assemble the distance matrices.
    """

    # Step 0: load the dataset.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time_dataset = time.time()
    data_loader, X_scalars, _, dim_attributes, (_, _) = load_mesh_dataset(
        dataset_name,
        config_params["datasets"],
        batch_size=config_params["embeddings"]["batch_size_big"],
        N_train=DEBUG_N_TRAIN,
        N_test=DEBUG_N_TEST,
        fuse_train_test=True,
    )
    time_dataset = time.time() - start_time_dataset

    h = hparams["num_wl_iterations"][0]
    # Step 1: For the unique hyperparameter, compute the distance matrix (matrices if aswwl).
    # Step 1A: Compute embeddings.
    # Step 1B: Assemble distance matrices.
    if kernel == "swwl":
        start_time_embeddings = time.time()
        encoder_generator = torch.Generator()
        encoder_generator.manual_seed(seed)
        encoder = SWWL_Encoder(
            dim_attributes,
            h,
            hparams["num_projections"],
            hparams["num_quantiles"],
            step=hparams["step_wl"],
            generator=encoder_generator,
        ).to(device)
        graph_embeddings = compute_graph_embeddings(
            data_loader, encoder, device, wwl=False
        )
        time_embeddings = time.time() - start_time_embeddings

        start_time_distances = time.time()
        D_matrices = distance_matrices_swwl(graph_embeddings)
        time_distances = time.time() - start_time_distances

    elif kernel == "aswwl":
        start_time_embeddings = time.time()
        encoder_generator = torch.Generator()
        encoder_generator.manual_seed(seed)
        encoder = ASWWL_Encoder(
            dim_attributes,
            h,
            hparams["num_projections"],
            hparams["num_quantiles"],
            step=hparams["step_wl"],
            generator=encoder_generator,
        ).to(device)
        graph_embeddings = compute_graph_embeddings(
            data_loader, encoder, device, wwl=False
        )
        time_embeddings = time.time() - start_time_embeddings

        start_time_distances = time.time()
        D_matrices = distance_matrices_aswwl(h, graph_embeddings)
        time_distances = time.time() - start_time_distances
    else:
        print(f"{kernel} is not a recognized kernel.")
        sys.exit(1)
    # elif kernel=="wwl":
    #     start_time_embeddings = time.time()
    #     encoder = WWL_Encoder(dim_attributes, h, step=hparams["step_wl"])
    #     graph_embeddings = compute_graph_embeddings(data_loader, encoder, device, wwl=True)
    #     time_embeddings = (time.time() - start_time_embeddings)

    #     start_time_distances = time.time()
    #     D_matrices = distance_matrices_wwl(graph_embeddings)
    #     time_distances = (time.time() -start_time_distances)

    # elif kernel=="wwl_ER":
    #     start_time_embeddings = time.time()
    #     encoder = WWL_Encoder(dim_attributes, h, step=hparams["step_wl"])
    #     graph_embeddings = compute_graph_embeddings(data_loader, encoder, device, wwl=True)
    #     time_embeddings = (time.time() - start_time_embeddings)

    #     start_time_distances = time.time()
    #     D_matrices = distance_matrices_wwl(graph_embeddings, sinkhorn_ER=True, sinkhorn_lambda=float(config_params["embeddings"]["wwl_er_sinkhorn_lambda"]))
    #     time_distances = (time.time() -start_time_distances)

    S_matrices = (
        distance_matrices_scalars(X_scalars)
        if (X_scalars is not None and compute_scalar_matrices)
        else None
    )
    times = {
        "time_dataset": time_dataset,
        "time_embeddings": time_embeddings,
        "time_matrices": time_distances,
    }
    return D_matrices, S_matrices, times


def regression_precomputed_distances(
    regressor: str,
    D_matrices: list,
    y: np.ndarray,
    N_train: int,
    config_params: dict,
    seed: int,
    S_matrices: list = None,
    verbose: int = 0,
):
    """
    Function to train and test a kernel method regressor given precomputed distance matrices.
    Once the model is trained, it is tested on new inputs and we return Q2/RMSE scores.
    D_matrices correspond to one hyperparameter and is a list of length 1 except for the ASWWL kernel.

    Args:
        regressor: The name of the regression model. Must be in ['rgasp'].
        D_matrices: The list of distance matrices required for a single distance substitution kernel.
        y: The train+test output array of shape (N,1).
        N_train: The number of train inputs.
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the training step.
        S_matrices: The optional scalar matrices used in tensorized kernels.
        verbose: The verbose level. If verbose=1: prints final results.
    Returns:
        scores_out (dict): The dict containing the test scores.
        times_training (float): The total required to train and test the model for the given distance matrix.
    """

    y_train, y_test = y[:N_train], y[N_train:]

    if S_matrices is not None:
        S_matrices_train_train = [S[:N_train][:, :N_train] for S in S_matrices]
        S_matrices_test_train = [S[N_train:][:, :N_train] for S in S_matrices]
    else:
        S_matrices_train_train, S_matrices_test_train = None, None

    start_time_training = time.time()

    D_matrices_train_train = [D[:N_train][:, :N_train] for D in D_matrices]
    D_matrices_test_train = [D[N_train:][:, :N_train] for D in D_matrices]

    if regressor == "rgasp":
        model = RGASP(
            D_matrices_train_train,
            y_train,
            S_matrices_train_train=S_matrices_train_train,
            num_restarts=config_params["optimization"]["rgasp"]["restarts"],
            verbose=verbose,
        )
        y_pred_test, _ = model.predict(
            D_matrices_test_train, S_matrices_test_train=S_matrices_test_train
        )

        q2 = r2_score(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    else:
        print(
            f"{kernel} is not a recognized regressor for precomputed distance matrices."
        )
        sys.exit(1)

    time_training = time.time() - start_time_training

    if verbose > 0:
        print(f"Time training (s): {time_training}")
        print(f"Test scores: q2 = {q2}, test_rmse = {test_rmse}")

    scores_out = {"test_q2": q2, "test_rmse": test_rmse}
    return scores_out, time_training


def compute_gram_matrices(
    dataset_name: str,
    kernel: str,
    hparams: dict,
    config_params: dict,
    seed: int,
    compute_scalar_matrices: bool = True,
):
    """
    Function to load a dataset and compute the Gram matrices between all input graphs for a given kernel.
    This method is only used for the propag kernel. Use compute_distance_matrices for other kernels.
    The hparams dict should contain only one hyperparameter value, and the method outputs only the distance matrix (matrices) for this hyperparameter.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel. Must be in ['propag', 'ghopper']
        hparams: The hyperparameter dict (eg {"num_wl_iterations": [3], "step_wl": 0}).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the kernel step.
        compute_scalar_matrices: Whether to compute all distance matrices between scalar inputs or not.
    Returns:
        K_matrices (list): A list (of length 1) of np.ndarray of shape (N,N) where N=N_train+N_test.
        S_matrices (list): A list of L np.ndarray of shape (N,N) where L is the number of input scalars.
        times (dict): The total times needed to load the dataset and to compute the Gram matrices.
    """

    # Step 0: load the dataset.
    start_time_dataset = time.time()
    Gs, X_scalars = load_mesh_dataset_grakel(
        dataset_name,
        config_params["datasets"],
        N_train=DEBUG_N_TRAIN,
        N_test=DEBUG_N_TEST,
    )
    time_dataset = time.time() - start_time_dataset

    # Step 1: for each hyperparameter tuple, compute the Gram matrices directly.
    if kernel == "propag":
        t_max = hparams["pk_tmax"][0]
        w = hparams["pk_w"][0]
        start_time_gram = time.time()
        gk = PropagationAttr(normalize=True, t_max=t_max, w=w, random_state=seed)
        K = gk.fit_transform(G)
        time_gram = time.time() - start_time_gram
    else:
        print(f"{kernel} is not a recognized kernel.")
        sys.exit(1)

    S_matrices = (
        distance_matrices_scalars(X_scalars)
        if (X_scalars is not None and compute_scalar_matrices)
        else None
    )
    times = {
        "time_dataset": time_dataset,
        "time_embeddings": 0.0,
        "time_matrices": time_gram,
    }
    return [K], S_matrices, times


def load_matrices_if_exist(
    dataset_name: str,
    kernel: str,
    hparams: dict,
    config_params: dict,
    seed: int,
    verbose: int = 0,
    kind: str = "distances",
):
    """
    This functions is used to load Gram/distance matrices (+ their times) for a unique hyperparameter of the given kernel.
    If the matrices cannot be found, the method returns None, None.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel.
        hparams: The hyperparameter dict (eg {"num_wl_iterations": [3], "step_wl": 0}).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used during the embedding/kernel step.
        verbose: If verbose>0, prints errors to show what matrices were not found.
        kind: A hint to select wether to load 'distances' or 'gram' matrices.
    Returns:
        matrices (list): The list of distance matrices for all hyperparameters. all_D_matrices[i] is list of size 1 np.ndarray of shape (N,N) where N=N_train+N_test except for 'aswwl' where the length is >1.The dict containing the test scores (and the validation results if requested in the config dict).
        times (dict): The total times needed to load the dataset, to compute the embeddings and the Gram/distance matrices.
    """

    # Caution: the paths and filenames depend on the kernel and its associated hyperparameters.
    save_root_matrices_seed = prefix_filenames(
        config_params["results"][kind], dataset_name, kernel, seed=seed
    )
    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER"]:
        h = hparams["num_wl_iterations"][0]
        suffix_matrices_i = suffix_matrices_filenames(
            dataset_name,
            kernel,
            seed,
            H=h,
            P=hparams["num_projections"],
            Q=hparams["num_quantiles"],
            T=hparams["step_wl"],
        )
        matrices = load_matrices(save_root_matrices_seed, suffix_matrices_i)
        if matrices is None:
            if verbose > 0:
                print(f"Distance matrix for h={h} not found: computing matrices again.")
            return None, None
    elif kernel in ["propag"]:
        t = hparams["pk_tmax"][0]
        w = hparams["pk_w"][0]
        suffix_matrices_i = suffix_matrices_filenames(
            dataset_name,
            kernel,
            seed,
            t_max=hparams["pk_tmax"][i],
            w=hparams["pk_w"][j],
        )
        matrices = load_matrices(save_root_matrices_seed, suffix_matrices_i)
        if matrices is None:
            if verbose > 0:
                print(
                    f"Distance matrix for tmax={t}, w={w} not found: computing matrices again."
                )
            return None, None

    save_root_times_seed = prefix_filenames(
        config_params["results"]["times"], dataset_name, kernel, seed=seed
    )
    suffix_times = suffix_times_filenames(dataset_name, kernel, hparams, seed=seed)
    times = load_times(save_root_times_seed, suffix_times)
    if times is None:
        print("Caution: distance computation times were not found...")
        times = {"time_dataset": -1, "time_embeddings": -1, "time_matrices": -1}
    return matrices, times


if __name__ == "__main__":
    # *** Usage *** python run_regression.py --dataset=Rotor37 --kernel=swwl --regressor=gpy -H=3 -Q=100 -P=50 --seed 0 --out=0 --verbose=1
    available_kernels = ["swwl", "aswwl", "wwl", "wwl_ER", "propag"]
    available_regressors = ["rgasp"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=available_mesh_datasets()
    )
    parser.add_argument("--kernel", type=str, required=True, choices=available_kernels)
    parser.add_argument(
        "--regressor", type=str, choices=available_regressors, default="rgasp"
    )
    parser.add_argument(
        "-H",
        "--wl_iter",
        help="For wwl/swwl/aswwl: number of WL iterations.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-Q",
        "--quantiles",
        help="For swwl/aswwl: number of quantiles of the SW embeddings.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-P",
        "--projections",
        help="For swwl/aswwl: number of projections of the SW embeddings.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-T",
        "--wl_step",
        help="For wwl/swwl/aswwl: setp for WL iterations.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-pk_tmax", "--pk_tmax", help="For propag: tmax list.", type=int, default=1
    )
    parser.add_argument(
        "-pk_w", "--pk_w", help="For propag: tmax list", type=float, default=0.01
    )
    parser.add_argument(
        "--seeds",
        help="The seeds used for the kernel and the training.",
        nargs="+",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--out",
        help="Train only for a specific output. If None, no training is performed. If -1, all outs are considered.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        help="0: nothing printed, 1: general infos+results, 2: 1 and optimisation infos.",
        type=int,
        choices=[0, 1, 2],
        default=1,
    )

    args = parser.parse_args()
    with open("config.yml", "r") as f:
        config_params = yaml.safe_load(f)
    dataset_name = args.dataset
    kernel = args.kernel
    regressor = args.regressor
    verbose = args.verbose
    seeds = args.seeds

    hparams = {}
    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER"]:
        hparams["num_wl_iterations"] = [args.wl_iter]
        hparams["step_wl"] = args.wl_step
    if kernel in ["swwl", "aswwl"]:
        hparams["num_projections"] = args.projections
        hparams["num_quantiles"] = args.quantiles
    if kernel in ["propag"]:
        hparams["pk_tmax"] = [args.pk_tmax]
        hparams["pk_w"] = [args.pk_w]

    kind = "gram" if kernel in ["propag"] else "distances"
    compute_matrices = (
        compute_distance_matrices if kind == "distances" else compute_gram_matrices
    )
    if kind == "distances":
        regression = regression_precomputed_distances
    elif args.out is not None:
        print(
            "Warning: Not possible to perform regression with precomputed Gram matrices. Use the R code instead."
        )

    if verbose > 0:
        print("\n------------------------------------------------")
        print(f"Dataset: {dataset_name}")
        print(f"Task: Regression using {regressor}")
        print(f"kernel: {kernel}")
        print(f"Hyperparameters: {hparams}")

    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER"]:
        import torch
    elif kernel in ["propag"]:
        from grakel.kernels import PropagationAttr

    y, (N_train, N_test) = load_scalars(
        dataset_name,
        roots=config_params["datasets"],
        fuse_train_test=True,
        N_train=DEBUG_N_TRAIN,
        N_test=DEBUG_N_TEST,
    )
    if args.out == -1:
        outs = np.arange(y.shape[-1])
    elif args.out is None or kind == "gram":
        outs = []
    else:
        outs = [args.out]

    q2s = {out: [] for out in outs}
    test_rmses = {out: [] for out in outs}

    use_input_scalars = config_params["regression"]["use_input_scalars"]
    S_matrices = None
    if (
        use_input_scalars and config_params["matrices"]["load_if_exists"]
    ):  # Try to load scalar distance matrices if the option is True
        S_matrices = load_scalar_matrices(
            config_params["results"]["scalar_matrices"], dataset_name, format="npy"
        )

    for i, seed in enumerate(seeds):
        if verbose > 0:
            print("------------------------------------------------")
            print(f"***** seed: {seed} *****")

        # I- Distance/Gram matrix computation.
        if kernel in ["swwl", "aswwl", "propag"] or i == 0:
            # Deterministic embeddings => only one distance matrix required for seed=0.
            # Non-deterministic embeddings => distance matrices for all seeds.
            matrices = None
            if config_params["matrices"][
                "load_if_exists"
            ]:  # Try to load distance matrices if the option is True.
                matrices, times = load_matrices_if_exist(
                    dataset_name, kernel, hparams, config_params, seed, verbose=verbose
                )
            if matrices is None or (
                S_matrices is None and use_input_scalars
            ):  # If failed, compute again.
                if kernel in ["wwl", "wwl_ER"]:
                    print(
                        "Error: Please use wwl_meshes_parallelized{1,2,3}.py to compute distance matrices using wwl..."
                    )
                    sys.exit(1)

                matrices, S_matrices, times = compute_matrices(
                    dataset_name,
                    kernel,
                    hparams,
                    config_params,
                    seed,
                    compute_scalar_matrices=use_input_scalars,
                )
                if config_params[
                    "matrices"
                ][
                    "save"
                ]:  # If distances need to be saved, save times too (only if distance matrices were computed).
                    save_root_times_seed = prefix_filenames(
                        config_params["results"]["times"],
                        dataset_name,
                        kernel,
                        seed=seed,
                    )
                    create_if_not_exists(save_root_times_seed)
                    suffix_times = suffix_times_filenames(
                        dataset_name, kernel, hparams, seed=seed
                    )
                    save_times(save_root_times_seed, suffix_times, times)

                    save_root_matrices_seed = prefix_filenames(
                        config_params["results"]["distances"],
                        dataset_name,
                        kernel,
                        seed=seed,
                    )
                    create_if_not_exists(save_root_matrices_seed)

                    if kernel in ["swwl", "aswwl"]:
                        suffix_matrices = suffix_matrices_filenames(
                            dataset_name,
                            kernel,
                            seed,
                            H=hparams["num_wl_iterations"][0],
                            P=hparams["num_projections"],
                            Q=hparams["num_quantiles"],
                            T=hparams["step_wl"],
                        )
                    elif kernel in ["propag"]:
                        suffix_matrices = suffix_matrices_filenames(
                            dataset_name,
                            kernel,
                            seed,
                            t_max=hparams["pk_tmax"][0],
                            w=hparams["pk_w"][0],
                        )
                    save_matrices(
                        save_root_matrices_seed,
                        suffix_matrices,
                        matrices,
                        format="npy",
                        kind=kind,
                    )

                    if (
                        i == 0
                    ):  # No need to save output scalars and input scalar matrices multiple times.
                        create_if_not_exists(
                            os.path.join(
                                config_params["results"]["output_scalars"], dataset_name
                            )
                        )
                        save_output_scalars(
                            config_params["results"]["output_scalars"],
                            dataset_name,
                            y,
                            format="npy",
                        )
                        # Save output scalars to load them quickly in R.
                        if use_input_scalars:
                            create_if_not_exists(
                                os.path.join(
                                    config_params["results"]["scalar_matrices"],
                                    dataset_name,
                                )
                            )
                            save_scalar_matrices(
                                config_params["results"]["scalar_matrices"],
                                dataset_name,
                                S_matrices,
                                format="npy",
                            )

            elif verbose > 0:
                print("Loading distances was successful.")

            if verbose > 0:
                print(f"Time loading dataset (s): {times['time_dataset']}")
                print(f"Time embeddings (s): {times['time_embeddings']}")
                print(f"Time matrices (s): {times['time_matrices']}")

        # II- Regression
        for out in outs:
            if verbose > 0:
                print(f"\n (Out {out})")
            y_out = y[:, out : out + 1]

            scores_out, time_training = regression(
                regressor,
                matrices,
                y_out,
                N_train,
                config_params,
                seed,
                S_matrices=S_matrices,
                verbose=verbose,
            )
            scores_out["num_wl_iterations"] = hparams["num_wl_iterations"]
            scores_out["step_wl"] = hparams["step_wl"]

            if config_params["results"]["save_scores"]:
                save_root_scores_seed = prefix_filenames(
                    config_params["results"]["scores"], dataset_name, kernel, seed=seed
                )
                create_if_not_exists(save_root_scores_seed)
                save_root_times_seed = prefix_filenames(
                    config_params["results"]["times"], dataset_name, kernel, seed=seed
                )
                create_if_not_exists(save_root_times_seed)
                times["time_training"] = time_training

                suffix_scores = suffix_scores_filenames(
                    dataset_name, regressor, kernel, hparams, seed=seed
                )
                save_scores_for_one_output(
                    save_root_scores_seed, suffix_scores, scores_out, out=out
                )
                save_times_training_for_one_output(
                    save_root_times_seed, suffix_scores, times, out=out
                )
            q2s[out].append(scores_out["test_q2"])
            test_rmses[out].append(scores_out["test_rmse"])

    if len(outs) > 0:
        print("------------------------------------------------")
        print("***** Final results *****")
        save_root_scores = prefix_filenames(
            config_params["results"]["scores"], dataset_name, kernel
        )
        create_if_not_exists(save_root_scores)
        suffix_scores = suffix_scores_filenames(
            dataset_name, regressor, kernel, hparams, seed=None
        )
        for out in outs:
            scores_out = {
                "mean_q2_test": np.mean(q2s[out]),
                "std_q2": np.std(q2s[out]),
                "mean_test_rmse": np.mean(test_rmses[out]),
                "std_test_rmse": np.std(test_rmses[out]),
                "n_exp": len(q2s[out]),
            }
            print(
                f"Mean/Std out {out}: q2 = {np.mean(q2s[out]):.5f} +- {np.std(q2s[out]):.5f},  test_rmse = {np.mean(test_rmses[out]):.5f} +- {np.std(test_rmses[out]):.5f} ({len(q2s[out])} exp)"
            )
            save_scores_for_one_output(
                save_root_scores, suffix_scores, scores_out, out=out
            )
