# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import argparse
import sys
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, matthews_corrcoef
from utils import (
    create_if_not_exists,
    fuse_labels_with_grakel_graphs,
    load_matrices,
    load_times,
    prefix_filenames,
    read_just_node_labels,
    renumber_nodes_to_start_at_one,
    save_matrices,
    save_scores_for_one_output,
    save_times,
    save_times_training_for_one_output,
    suffix_matrices_filenames,
    suffix_scores_filenames,
    suffix_times_filenames,
)

from graph_gp.models import SVC_precomputed_distances, SVC_precomputed_Gram
from graph_gp.splitters import split_train_test_indices_stratified_fold


def compute_distance_matrices(
    dataset_name: str, kernel: str, hparams: dict, config_params: dict, seed: int
):
    """
    Function to load a dataset and compute the distance matrices between all input graphs for a given kernel.
    This method is only used for kernels that require the computation of a distance matrix. Use compute_gram_matrices for other kernels.
    The output is a list containing the matrices for all hyperparameters in the grid defined by hparams.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel. Must be in ["swwl", "aswwl", "wwl", "wwl_ER", "sgml", "fgw"].
        hparams: The hyperparameter dict (eg {"num_wl_iterations": [0,1,2,3], "step_wl": 0}).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the embedding/distance step.
    Returns:
        all_D_matrices (list): The list of distance matrices for all hyperparameters. all_D_matrices[i] is list of size 1 np.ndarray of shape (N,N) where N=N_train+N_test except for 'aswwl' where the length is >1.
        times (dict): The total times needed to load the dataset, to compute the embeddings and the distance matrices.
    """

    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER"]:
        # The swwl, asswl, wwl and wwl_ER kernels require the computation of WL iterations with torch.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_D_matrices = []
        time_embeddings, time_distances = 0.0, 0.0

        # Step 0: load the dataset
        start_time_dataset = time.time()
        data_loader, dim_attributes = load_classification_dataset(
            dataset_name,
            batch_size=config_params["embeddings"]["batch_size_small"],
            root_datasets=config_params["datasets"]["root_TUDatasets"],
        )
        time_dataset = time.time() - start_time_dataset

        # Step 1: For each hyperparameter compute the distance matrix.
        # Step 1A: Compute embeddings.
        # Step 1B: Assemble distance matrices.
        for h in hparams["num_wl_iterations"]:
            if kernel == "swwl":
                start_time_embeddings = time.time()
                encoder_generator = torch.Generator()
                encoder_generator.manual_seed(seed)
                if dataset_name == "Cuneiform":
                    encoder = SWWL_Encoder_ContinuousAndCategorical(
                        dim_attributes,
                        [4, 3],
                        h,
                        hparams["num_projections"],
                        hparams["num_quantiles"],
                        step=hparams["step_wl"],
                        generator=encoder_generator,
                    ).to(device)
                else:
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
                time_embeddings += time.time() - start_time_embeddings

                start_time_distances = time.time()
                D_matrices = distance_matrices_swwl(graph_embeddings)
                time_distances += time.time() - start_time_distances

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
                time_embeddings += time.time() - start_time_embeddings

                start_time_distances = time.time()
                D_matrices = distance_matrices_aswwl(h, graph_embeddings)
                time_distances += time.time() - start_time_distances

            elif kernel == "wwl":
                start_time_embeddings = time.time()
                if dataset_name == "Cuneiform":
                    encoder = WWL_Encoder_ContinuousAndCategorical(
                        dim_attributes, [4, 3], h, step=hparams["step_wl"]
                    )
                else:
                    encoder = WWL_Encoder(dim_attributes, h, step=hparams["step_wl"])
                graph_embeddings = compute_graph_embeddings(
                    data_loader, encoder, device, wwl=True
                )
                time_embeddings += time.time() - start_time_embeddings

                start_time_distances = time.time()
                D_matrices = distance_matrices_wwl(graph_embeddings)
                time_distances += time.time() - start_time_distances

            elif kernel == "wwl_ER":
                start_time_embeddings = time.time()
                if dataset_name == "Cuneiform":
                    encoder = WWL_Encoder_ContinuousAndCategorical(
                        dim_attributes, [4, 3], h, step=hparams["step_wl"]
                    )
                else:
                    encoder = WWL_Encoder(dim_attributes, h, step=hparams["step_wl"])
                graph_embeddings = compute_graph_embeddings(
                    data_loader, encoder, device, wwl=True
                )
                time_embeddings += time.time() - start_time_embeddings

                # The entropic regularization term strongly depends on lambda that is fixed. In practice, this hyperparameter should be tuned.
                start_time_distances = time.time()
                D_matrices = distance_matrices_wwl(
                    graph_embeddings,
                    sinkhorn_ER=True,
                    sinkhorn_lambda=float(
                        config_params["embeddings"]["wwl_er_sinkhorn_lambda"]
                    ),
                )
                time_distances += time.time() - start_time_distances

            all_D_matrices.append(D_matrices)
    elif kernel in ["sgml"]:
        (
            all_D_matrices,
            time_dataset,
            time_embeddings,
            time_distances,
        ) = compute_all_dataset_distances_sgml(
            dataset_name,
            hparams["num_wl_iterations"],
            "fuse" if dataset_name == "Cuneiform" else "attributes",
            config_params,
            seed=seed,
        )

    elif kernel in ["fgw"]:
        start_time_dataset = time.time()
        data_loader, dim_attributes = load_classification_dataset(
            dataset_name,
            batch_size=1,
            root_datasets=config_params["datasets"]["root_TUDatasets"],
        )
        time_dataset = time.time() - start_time_dataset

        start_time_embeddings = time.time()
        graph_embeddings = compute_graph_embeddings_fgw(data_loader, dim_attributes)
        time_embeddings = time.time() - start_time_embeddings

        start_time_distances = time.time()
        all_D_matrices = distance_matrices_fgw(
            graph_embeddings, alphas=hparams["fgw_alphas"]
        )
        time_distances = time.time() - start_time_distances
    else:
        print(f"{kernel} is not a recognized kernel.")
        sys.exit(1)

    times = {
        "time_dataset": time_dataset,
        "time_embeddings": time_embeddings,
        "time_matrices": time_distances,
    }
    return all_D_matrices, times


def classification_precomputed_distances(
    classifier: str,
    all_D_matrices: list,
    y: np.ndarray,
    config_params: dict,
    seed: int,
    verbose: int = 0,
):
    """
    Function to train and test a kernel method classifier given precomputed distance matrices.
    An inner cross-validation step is performed to select simultaneously
    - the distance matrix to keep in all_D_matrices (e.g. for varying numbers of WL iterations),
    - the distance substituion kernel hyperparameter (e.g. the range parameters of a RBF kernel),
    - the classifier hyperparameters (e.g. SVM regularization parameter).
    Once the model is trained, it is tested on new inputs and we return Accuracy/MCC scores.

    Args:
        classifier: The name of the classifier. Must be in ['svc', 'svc_rbf'].
        all_D_matrices: The list (of lists) of distance matrices for all hyperparameters.
        y: The train+test output array of shape (N,1).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the training step.
        verbose: The verbose level. If verbose=1: prints final results. If verbose>1: prints intermediate validation results.
    Returns:
        scores_out (dict): The dict containing the test scores (and the validation results if requested in the config dict).
        times_training (float): The total required to train and test the model for all hyperparameters.
    """

    # Train/test sets depend on the seed and are the same for all the methods we compare.
    indices_train, indices_test = split_train_test_indices_stratified_fold(
        config_params["datasets"]["classification_NSPLITS"],
        config_params["datasets"]["classification_stratified_split_seed"],
        seed,
        np.arange(len(y)),
        y=y,
    )
    y_train, y_test = y[indices_train], y[indices_test]

    models, valid_accuracies, test_accuracies, test_mccs = [], [], [], []
    start_time_training = time.time()

    # Step 2: Training.
    for i in range(len(all_D_matrices)):
        # Step 2A: for each distance matrix, train a SVC.
        D_matrices = all_D_matrices[i]
        D_matrices_train_train = [
            D[indices_train][:, indices_train] for D in D_matrices
        ]
        D_matrices_test_train = [D[indices_test][:, indices_train] for D in D_matrices]

        if classifier in ["svc"]:  # by default, svc = svc with exponential kernel
            svm_param_grid = {
                "C": np.logspace(
                    config_params["optimization"]["svc"]["grid_C_log"]["lower"],
                    config_params["optimization"]["svc"]["grid_C_log"]["upper"],
                    num=config_params["optimization"]["svc"]["grid_C_log"]["num"],
                )
            }
            gammas = np.logspace(
                config_params["optimization"]["svc"]["grid_gammas_log"]["lower"],
                config_params["optimization"]["svc"]["grid_gammas_log"]["upper"],
                num=config_params["optimization"]["svc"]["grid_gammas_log"]["num"],
            )
            model = SVC_precomputed_distances(
                D_matrices_train_train,
                y_train,
                type_kernel="exp",
                svm_param_grid=svm_param_grid,
                gammas=gammas,
                nugget=config_params["optimization"]["svc"]["nugget"],
                verbose=verbose,
                cv_splits=config_params["optimization"]["svc"]["cv_splits"],
                cv_seed=seed,
            )
        elif classifier == "svc_rbf":
            svm_param_grid = {
                "C": np.logspace(
                    config_params["optimization"]["svc"]["grid_C_log"]["lower"],
                    config_params["optimization"]["svc"]["grid_C_log"]["upper"],
                    num=config_params["optimization"]["svc"]["grid_C_log"]["num"],
                )
            }
            gammas = np.logspace(
                config_params["optimization"]["svc"]["grid_gammas_log"]["lower"],
                config_params["optimization"]["svc"]["grid_gammas_log"]["upper"],
                num=config_params["optimization"]["svc"]["grid_gammas_log"]["num"],
            )
            model = SVC_precomputed_distances(
                D_matrices_train_train,
                y_train,
                type_kernel="rbf",
                svm_param_grid=svm_param_grid,
                gammas=gammas,
                nugget=config_params["optimization"]["svc"]["nugget"],
                verbose=verbose,
                cv_splits=config_params["optimization"]["svc"]["cv_splits"],
                cv_seed=seed,
            )
        else:
            print(f"{classifier} is not a recognized classifier.")
            sys.exit(1)
        models.append(model)
        valid_acc = model.get_best_cv_result()
        valid_accuracies.append(valid_acc)
        y_pred_test, _ = model.predict(D_matrices_test_train)
        # Test scores are computed for ALL hyperparameters. However, only validation results are used to find the best hyperparameters.
        test_acc = accuracy_score(y_test, y_pred_test)
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        test_accuracies.append(test_acc)
        test_mccs.append(test_mcc)

        if verbose > 1:
            print(
                f"Valid {i}, valid_acc={valid_acc}, test_acc={test_acc}, test_mcc={test_mcc}"
            )
    time_training = time.time() - start_time_training

    # Step 2B: keep only the best parameters obtained using the validation scores.
    best_idx = np.argmax(valid_accuracies)
    best_model = models[best_idx]

    # Step 3: predict using the best model for test inputs.
    test_acc_for_best_valid = test_accuracies[best_idx]
    test_mcc_for_best_valid = test_mccs[best_idx]

    if verbose > 0:
        print(f"Time training (s): {time_training}")
        print(
            f"Test scores: test_acc = {test_acc_for_best_valid}, test_mcc = {test_mcc_for_best_valid}"
        )
        print(f"Model parameters: {best_model.get_parameters()}, best idx={best_idx}")
    scores_out = {
        "test_acc": test_acc_for_best_valid,
        "test_mcc": test_mcc_for_best_valid,
    }
    if config_params["results"]["save_valid_scores"]:
        scores_out["all_valid_accuracies"] = valid_accuracies
        scores_out["all_test_accuracies"] = test_accuracies
    return scores_out, time_training


def compute_gram_matrices(
    dataset_name: str, kernel: str, hparams: dict, config_params: dict, seed: int
):
    """
    Function to load a dataset and compute the Gram matrices between all input graphs for a given kernel.
    This method is only used for the ghopper and propag kernels. Use compute_distance_matrices for other kernels.
    The output is a list containing the matrices for all hyperparameters in the grid defined by hparams.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel. Must be in ['propag', 'ghopper'].
        hparams: The hyperparameter dict (eg {"num_wl_iterations": [0,1,2,3], "step_wl": 0}).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the kernel step.
    Returns:
        all_K_matrices (list): The list of Gram matrices for all hyperparameters. all_K_matrices[i] is list of size 1 np.ndarray of shape (N,N) where N=N_train+N_test.
        times (dict): The total times needed to load the dataset and to compute the Gram matrices.
    """
    all_K_matrices = []

    # Step 0: load the dataset.
    start_time_dataset = time.time()
    dataset = fetch_dataset(
        dataset_name,
        prefer_attr_nodes=True,
        verbose=False,
        data_home=config_params["datasets"]["root_TUDatasets_zip"],
    )
    G = dataset.data
    if (
        dataset_name == "Cuneiform"
    ):  # If the dataset is Cuneiform, fuse the node labels with their attributes.
        G = renumber_nodes_to_start_at_one(G)
        node_labels = read_just_node_labels(
            dataset_name, root_datasets=config_params["datasets"]["root_TUDatasets"]
        )
        G = fuse_labels_with_grakel_graphs(G, node_labels)
    time_dataset = time.time() - start_time_dataset

    # Step 1: for each hyperparameter tuple, compute the Gram matrices directly.
    start_time_gram = time.time()
    if kernel in ["propag"]:
        for t in hparams["pk_tmax"]:
            for w in hparams["pk_w"]:
                gk = PropagationAttr(normalize=True, t_max=t, w=w, random_state=seed)
                K = gk.fit_transform(G)
                all_K_matrices.append([K])
    elif kernel in ["ghopper"]:
        gk = GraphHopper(normalize=True, kernel_type="linear")
        K = gk.fit_transform(G)
        all_K_matrices.append([K])
    else:
        print(f"{kernel} is not a recognized kernel for precomputed distance matrices.")
        sys.exit(1)
    time_gram = time.time() - start_time_gram

    times = {
        "time_dataset": time_dataset,
        "time_embeddings": 0.0,
        "time_matrices": time_gram,
    }
    return all_K_matrices, times


def classification_precomputed_gram(
    classifier: str,
    all_K_matrices: list,
    y: np.ndarray,
    config_params: dict,
    seed: int,
    verbose: int = 0,
):
    """
    Function to train and test a kernel method classifier given precomputed Gram matrices.
    An inner cross-validation step is performed to select simultaneously
    - the Gram matrix to keep in all_K_matrices (e.g. for varying numbers of WL iterations),
    - the classifier hyperparameters (e.g. SVM regularization parameter).
    Once the model is trained, it is tested on new inputs and we return Accuracy/MCC scores.

    Args:
        classifier: The name of the classifier. Only 'svc' is supported here.
        all_K_matrices: The list (of lists) of Gram matrices for all hyperparameters.
        y: The train+test output array of shape (N,1).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used for the training step.
        verbose: The verbose level. If verbose=1: prints final results. If verbose>1: prints intermediate validation results.
    Returns:
        scores_out (dict): The dict containing the test scores (and the validation results if requested in the config dict).
        times_training (float): The total required to train and test the model for all hyperparameters.
    """

    # Train/test sets depend on the seed and are the same for all the methods we compare.
    indices_train, indices_test = split_train_test_indices_stratified_fold(
        config_params["datasets"]["classification_NSPLITS"],
        config_params["datasets"]["classification_stratified_split_seed"],
        seed,
        np.arange(len(y)),
        y=y,
    )
    y_train, y_test = y[indices_train], y[indices_test]

    models, valid_accuracies, test_accuracies, test_mccs = [], [], [], []
    start_time_training = time.time()

    # Step 2A: training.
    for i in range(len(all_K_matrices)):
        # Step 2A: for each distance matrix, train a SVC.
        K_matrix = all_K_matrices[i][0]

        K_matrix_train_train = K_matrix[indices_train][:, indices_train]
        K_matrix_test_train = K_matrix[indices_test][:, indices_train]

        svm_param_grid = {
            "C": np.logspace(
                config_params["optimization"]["svc"]["grid_C_log"]["lower"],
                config_params["optimization"]["svc"]["grid_C_log"]["upper"],
                num=config_params["optimization"]["svc"]["grid_C_log"]["num"],
            )
        }

        if classifier == "svc":
            model = SVC_precomputed_Gram(
                K_matrix_train_train,
                y_train,
                svm_param_grid=svm_param_grid,
                nugget=config_params["optimization"]["svc"]["nugget"],
                verbose=verbose,
                cv_splits=config_params["optimization"]["svc"]["cv_splits"],
                cv_seed=seed,
            )
        else:
            print(
                f"{kernel} is not a recognized classifier for precomputed Gram matrices."
            )
            sys.exit(1)
        models.append(model)
        valid_acc = model.get_best_cv_result()
        valid_accuracies.append(valid_acc)
        y_pred_test, _ = model.predict(K_matrix_test_train)
        # Test scores are computed for ALL hyperparameters. However, they are not used to find the best hyperparameters.
        test_acc = accuracy_score(y_test, y_pred_test)
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        test_accuracies.append(test_acc)
        test_mccs.append(test_mcc)

        if verbose > 1:
            print(
                f"Param number {i}: valid_acc={valid_acc}, test_acc={test_acc}, test_mcc={test_mcc}"
            )
            print(np.sum(y_pred_test), len(y_pred_test))
    time_training = time.time() - start_time_training

    # Step 2B: keep only the best parameters obtained using the validation scores.
    best_idx = np.argmax(valid_accuracies)
    best_model = models[best_idx]

    # Step 3: predict using the best model for test inputs.
    test_acc_for_best_valid = test_accuracies[best_idx]
    test_mcc_for_best_valid = test_mccs[best_idx]

    if verbose > 0:
        print(f"Time training (s): {time_training}")
        print(
            f"Test scores: test_acc = {test_acc_for_best_valid}, test_mcc = {test_mcc_for_best_valid}"
        )
        print(f"Model parameters: {best_model.get_parameters()}, best idx={best_idx}")
    scores_out = {
        "test_acc": test_acc_for_best_valid,
        "test_mcc": test_mcc_for_best_valid,
    }
    if config_params["results"]["save_valid_scores"]:
        scores_out["all_valid_accuracies"] = valid_accuracies
        scores_out["all_test_accuracies"] = test_accuracies
    return scores_out, time_training


def load_all_matrices_if_all_exist(
    dataset_name: str,
    kernel: str,
    hparams: dict,
    config_params: dict,
    seed: int,
    verbose: int = 0,
    kind: str = "distances",
):
    """
    This functions is used to load Gram/distance matrices (+ their times) for all hyperparameters of the given kernel.
    If the matrix cannot be found for some hyperparameter, the method returns None, None.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel.
        hparams: The hyperparameter dict (eg {"num_wl_iterations": [0,1,2,3], "step_wl": 0}).
        config_params: The configuration dict loaded from config.yml.
        seed: The seed used during the embedding/kernel step.
        verbose: The verbose level. If verbose>0, prints errors to show what matrices were not found.
        kind: A hint to select whether to load 'distances' or 'gram' matrices.
    Returns:
        all_matrices (list): The list of distance matrices for all hyperparameters. all_D_matrices[i] is list of size 1 np.ndarray of shape (N,N) where N=N_train+N_test except for 'aswwl' where the length is >1.The dict containing the test scores (and the validation results if requested in the config dict).
        times (dict): The total times needed to load the dataset, to compute the embeddings and the Gram/distance matrices.
    """

    # Caution: the paths and filenames depend on the kernel and its associated hyperparameters.
    save_root_matrices_seed = prefix_filenames(
        config_params["results"][kind], dataset_name, kernel, seed=seed
    )
    all_matrices = []
    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER", "sgml"]:
        for i, h in enumerate(hparams["num_wl_iterations"]):
            P = hparams["num_projections"] if kernel in ["swwl", "aswwl"] else None
            Q = hparams["num_quantiles"] if kernel in ["swwl", "aswwl"] else None
            T = (
                hparams["step_wl"]
                if kernel in ["swwl", "aswwl", "wwl", "wwl_ER"]
                else None
            )
            suffix_matrices_i = suffix_matrices_filenames(
                dataset_name, kernel, seed, H=h, P=P, Q=Q, T=T
            )
            matrices = load_matrices(save_root_matrices_seed, suffix_matrices_i)
            if matrices is None:
                if verbose > 0:
                    print(
                        f"Distance matrix for h={h} not found: computing all matrices again."
                    )
                return None, None
            all_matrices.append(matrices)
    elif kernel in ["fgw"]:
        for i, a in enumerate(hparams["fgw_alphas"]):
            suffix_matrices_i = suffix_matrices_filenames(
                dataset_name, kernel, seed, alpha=hparams["fgw_alphas"][i]
            )
            matrices = load_matrices(save_root_matrices_seed, suffix_matrices_i)
            if matrices is None:
                if verbose > 0:
                    print(
                        f"Distance matrix for alpha={a} not found: computing all matrices again."
                    )
                return None, None
            all_matrices.append(matrices)
    elif kernel in ["propag"]:
        for i, t in enumerate(hparams["pk_tmax"]):
            for j, w in enumerate(hparams["pk_w"]):
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
                            f"Distance matrix for tmax={t}, w={w} not found: computing all matrices again."
                        )
                    return None, None
                all_matrices.append(matrices)
    elif kernel in ["ghopper"]:
        suffix_matrices_i = suffix_matrices_filenames(dataset_name, kernel, seed)
        matrices = load_matrices(save_root_matrices_seed, suffix_matrices_i)
        if matrices is None:
            if verbose > 0:
                print(
                    f"Distance matrix for tmax={t}, w={w} not found: computing all matrices again."
                )
            return None, None
        all_matrices.append(matrices)

    save_root_times_seed = prefix_filenames(
        config_params["results"]["times"], dataset_name, kernel, seed=seed
    )
    suffix_times = suffix_times_filenames(dataset_name, kernel, hparams, seed=seed)
    times = load_times(save_root_times_seed, suffix_times)
    if times is None:
        print("Caution: distance computation times were not found...")
        times = {"time_dataset": -1, "time_embeddings": -1, "time_matrices": -1}
    return all_matrices, times


if __name__ == "__main__":
    # *** Usage ***  python run_classification.py --dataset=BZR --kernel=swwl --classifier=svc -H 3 -Q=20 -P=10 --seed 0 --verbose=1

    available_kernels = [
        "swwl",
        "aswwl",
        "wwl",
        "wwl_ER",
        "sgml",
        "fgw",
        "propag",
        "ghopper",
    ]
    available_classifiers = ["svc", "svc_rbf"]
    available_classification_datasets = [
        "BZR",
        "COX2",
        "ENZYMES",
        "PROTEINS",
        "Cuneiform",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=available_classification_datasets
    )
    parser.add_argument("--kernel", type=str, required=True, choices=available_kernels)
    parser.add_argument(
        "--classifier", type=str, choices=available_classifiers, default="svc"
    )
    parser.add_argument(
        "-H",
        "--wl_iter",
        help="For wwl/swwl/aswwl/sgml: number of WL iterations. If multiple values, they are hyper-optimized.",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "-Q",
        "--quantiles",
        help="For swwl/aswwl: number of quantiles of the SW embeddings.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-P",
        "--projections",
        help="For swwl/aswwl: number of projections of the SW embeddings.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-T",
        "--wl_step",
        help="For wwl/swwl/aswwl: step for WL iterations.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-alphas",
        "--fgw_alphas",
        help="For fgw: alpha parameter of FGW. If multiple values, they are hyper-optimized.",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "-pk_tmax",
        "--pk_tmax",
        help="For propag: tmax list. If multiple values, they are hyper-optimized.",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7],
    )
    parser.add_argument(
        "-pk_w",
        "--pk_w",
        help="For propag: w list. If multiple values, they are hyper-optimized.",
        nargs="+",
        type=float,
        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    parser.add_argument(
        "--seeds",
        help="The seeds used for the kernel and the training.",
        nargs="+",
        type=int,
        default=[0],
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
    classifier = args.classifier
    hparams = {}
    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER", "sgml"]:
        hparams["num_wl_iterations"] = args.wl_iter
        if kernel != "sgml":
            hparams["step_wl"] = args.wl_step
    if kernel in ["swwl", "aswwl"]:
        hparams["num_projections"] = args.projections
        hparams["num_quantiles"] = args.quantiles
    if kernel in ["fgw"]:
        hparams["fgw_alphas"] = args.fgw_alphas
    if kernel in ["propag"]:
        hparams["pk_tmax"] = args.pk_tmax
        hparams["pk_w"] = args.pk_w
    verbose = args.verbose
    seeds = args.seeds
    root_datasets = config_params["datasets"]["root_TUDatasets"]

    kind = "gram" if kernel in ["propag", "ghopper"] else "distances"
    classification = (
        classification_precomputed_distances
        if kind == "distances"
        else classification_precomputed_gram
    )
    compute_matrices = (
        compute_distance_matrices if kind == "distances" else compute_gram_matrices
    )

    if verbose > 0:
        print("\n------------------------------------------------")
        print(f"Dataset: {dataset_name}")
        print(f"Task: Classification using {classifier}")
        print(f"kernel: {kernel}")
        print(f"Hyperparameters: {hparams}")

    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER", "fgw"]:
        import torch
        from datasets_classification import load_classification_dataset, load_scalars

        from graph_gp.distances import (
            distance_matrices_aswwl,
            distance_matrices_fgw,
            distance_matrices_swwl,
            distance_matrices_wwl,
        )
        from graph_gp.embeddings import (
            compute_graph_embeddings,
            compute_graph_embeddings_fgw,
        )
        from graph_gp.encoders import (
            ASWWL_Encoder,
            SWWL_Encoder,
            SWWL_Encoder_ContinuousAndCategorical,
            WWL_Encoder,
            WWL_Encoder_ContinuousAndCategorical,
        )
    elif kernel in ["sgml"]:
        from lib_sgml.distance_sgml import compute_all_dataset_distances_sgml
        from lib_sgml.process_data import load_scalars
    elif kernel in ["propag", "ghopper"]:
        from grakel.datasets import fetch_dataset
        from grakel.kernels import GraphHopper, PropagationAttr

        def load_scalars(dataset_name, root_datasets=None):
            dataset = fetch_dataset(
                dataset_name,
                prefer_attr_nodes=True,
                verbose=False,
                data_home=root_datasets,
            )
            return dataset.target.reshape(-1, 1)

        root_datasets = config_params["datasets"]["root_TUDatasets_zip"]

    y = load_scalars(dataset_name, root_datasets=root_datasets)
    out = 0
    test_accuracies, test_mccs = [], []
    for i, seed in enumerate(seeds):
        if verbose > 0:
            print("------------------------------------------------")
            print(f"***** seed: {seed} *****")

        # I- Distance/Gram matrix computation.
        if kernel in ["swwl", "aswwl", "sgml", "propag"] or i == 0:
            # Deterministic embeddings => only one distance matrix required for seed=0.
            # Non-deterministic embeddings => distance matrices for all seeds.
            all_matrices = None
            if config_params["matrices"][
                "load_if_exists"
            ]:  # Try to load distance matrices if the option is True.
                all_matrices, times = load_all_matrices_if_all_exist(
                    dataset_name,
                    kernel,
                    hparams,
                    config_params,
                    seed,
                    verbose=verbose,
                    kind=kind,
                )
            if all_matrices is None:  # If failed, compute again
                all_matrices, times = compute_matrices(
                    dataset_name, kernel, hparams, config_params, seed
                )
                if config_params["matrices"]["save"]:
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
                        config_params["results"][kind], dataset_name, kernel, seed=seed
                    )
                    create_if_not_exists(save_root_matrices_seed)
                    if kernel in ["swwl", "aswwl", "wwl", "wwl_ER", "sgml"]:
                        for i, h in enumerate(hparams["num_wl_iterations"]):
                            P = (
                                hparams["num_projections"]
                                if kernel in ["swwl", "aswwl"]
                                else None
                            )
                            Q = (
                                hparams["num_quantiles"]
                                if kernel in ["swwl", "aswwl"]
                                else None
                            )
                            T = (
                                hparams["step_wl"]
                                if kernel in ["swwl", "aswwl", "wwl", "wwl_ER"]
                                else None
                            )
                            suffix_matrices_i = suffix_matrices_filenames(
                                dataset_name, kernel, seed, H=h, P=P, Q=Q, T=T
                            )
                            save_matrices(
                                save_root_matrices_seed,
                                suffix_matrices_i,
                                all_matrices[i],
                                kind=kind,
                            )
                    elif kernel in ["fgw"]:
                        for i, a in enumerate(hparams["fgw_alphas"]):
                            suffix_matrices_i = suffix_matrices_filenames(
                                dataset_name,
                                kernel,
                                seed,
                                alpha=hparams["fgw_alphas"][i],
                            )
                            save_matrices(
                                save_root_matrices_seed,
                                suffix_matrices_i,
                                all_matrices[i],
                                kind=kind,
                            )
                    elif kernel in ["propag"]:
                        for i, t in enumerate(hparams["pk_tmax"]):
                            for j, w in enumerate(hparams["pk_w"]):
                                suffix_matrices_i = suffix_matrices_filenames(
                                    dataset_name,
                                    kernel,
                                    seed,
                                    t_max=hparams["pk_tmax"][i],
                                    w=hparams["pk_w"][j],
                                )
                                save_matrices(
                                    save_root_matrices_seed,
                                    suffix_matrices_i,
                                    all_matrices[j + i * len(hparams["pk_w"])],
                                )
                    elif kernel in ["ghopper"]:
                        suffix_matrices_i = suffix_matrices_filenames(
                            dataset_name, kernel, seed
                        )
                        save_matrices(
                            save_root_matrices_seed,
                            suffix_matrices_i,
                            all_matrices[0],
                            kind=kind,
                        )
            elif verbose > 0:
                print("Loading distances was successful.")

            if verbose > 0:
                print(f"Time loading dataset (s): {times['time_dataset']}")
                print(f"Time embeddings (s): {times['time_embeddings']}")
                print(f"Time matrices (s): {times['time_matrices']}")

        # II- Classification.
        y_out = y[:, out : out + 1]
        scores_out, time_training = classification(
            classifier, all_matrices, y_out, config_params, seed, verbose=verbose
        )
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
                dataset_name, classifier, kernel, hparams, seed=seed
            )
            save_scores_for_one_output(
                save_root_scores_seed, suffix_scores, scores_out, out=out
            )
            save_times_training_for_one_output(
                save_root_times_seed, suffix_scores, times, out=out
            )
        test_accuracies.append(scores_out["test_acc"])
        test_mccs.append(scores_out["test_mcc"])

    print("------------------------------------------------")
    print("***** Final results *****")
    save_root_scores = prefix_filenames(
        config_params["results"]["scores"], dataset_name, kernel
    )
    create_if_not_exists(save_root_scores)
    suffix_scores = suffix_scores_filenames(
        dataset_name, classifier, kernel, hparams, seed=None
    )
    scores_out = {
        "mean_test_acc": np.mean(test_accuracies),
        "std_test_acc": np.std(test_accuracies),
        "mean_test_mcc": np.mean(test_mccs),
        "std_test_mcc": np.std(test_mccs),
        "n_exp": len(test_accuracies),
    }
    print(
        f"Mean/Std out: test_acc = {np.mean(test_accuracies):.5f} +- {np.std(test_accuracies):.5f},  test_mcc = {np.mean(test_mccs):.5f} +- {np.std(test_mccs):.5f} ({len(test_accuracies)} exp)"
    )
    save_scores_for_one_output(save_root_scores, suffix_scores, scores_out, out=out)
