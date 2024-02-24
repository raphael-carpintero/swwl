# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

""" 
This file gathers utility functions: save/load, file naming.
All result files are saved using the same nomenclature: 
- a prefix made of a global root + extra folders (namely the dataset name, the kernel) where all results are saved,
- a keyword depending on the saved object (for instance 'D' for distances, 'scores' for scores),
- a suffix made of information about the model and its hyperparameters (for instance '_wwl_H3_T1' for the swwl kernel with 3 wl iterations with step 1).

example: PATH_TO_RESULTS/distances/Rotor37_CM/swwl/exp0/D_Rotor37_CM_swwl_H3_P100_Q50_T1_seed4.npy 
        contains the distance matrice of the SWWL kernel applied to Rotor37_CM for the experiment 4, 100 projections and 50 quantiles.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd


def create_if_not_exists(folder: str):
    # If the folder does not exist, create it.
    if not os.path.exists(folder):
        os.makedirs(folder)


def suffix_scores_filenames(
    dataset_name: str, training_model: str, kernel: str, hparams: dict, seed: int = None
):
    """
    Function to build the path where score filenames need to be saved according to the dataset, the kernel, the training model and the kernel hyperparameters.
    Final path = prefix + keyword + suffix + format.

    Args:
        dataset_name: The name of the dataset.
        training_model: The model used for training (e.g. 'svc_exp' or 'rgasp').
        kernel: The kernel or distance function (e.g. 'swwl' or 'fgw').
        hparams: The dict containing at least all hyperparameters of the kernel.
        seed: The seed used for the training. Choose seed=None if this is a path to the merged results for all experiments.
    Returns:
        suffix (str): The string suffix used to save the score filename.
    """
    suffix = f"_{dataset_name}_{training_model}_{kernel}"

    if kernel == "swwl" or kernel == "aswwl":
        H = hparams["num_wl_iterations"]
        T = hparams["step_wl"]
        P = hparams["num_projections"]
        Q = hparams["num_quantiles"]
        if len(H) == 1:
            suffix += f"_H{H[0]}_P{P}_Q{Q}_T{T}"
        else:
            suffix += f"_P{P}_Q{Q}"
    elif kernel == "wwl" or kernel == "wwl_ER":
        H = hparams["num_wl_iterations"]
        T = hparams["step_wl"]
        if len(H) == 1:
            suffix += f"_H{H[0]}_T{T}"
    elif kernel == "fgw":
        alphas = hparams["fgw_alphas"]
        if len(alphas) == 1:
            suffix += f"_alpha{alphas[0]:.3f}"
    elif kernel == "sgml":
        H = hparams["num_wl_iterations"]
        if len(H) == 1:
            suffix += f"_H{H}"
    elif kernel == "propag":
        w = hparams["pk_w"]
        t_max = hparams["pk_tmax"]
        if len(t_max) == 1:
            suffix += f"_tmax{t_max[0]}"
        if len(w) == 1:
            suffix += f"_w{w[0]}"
    if seed is not None:
        suffix += f"_seed{seed}"
    return suffix


def suffix_times_filenames(
    dataset_name: str, kernel: str, hparams: dict, seed: int = None
):
    """
    Function to build the path to where times filenames need to be saved according to the dataset, the kernel, and the kernel hyperparameters.
    Final path = prefix + keyword + suffix + format.

    Args:
        dataset_name: The name of the dataset.
        kernel: The kernel or distance function (e.g. 'swwl' or 'fgw').
        hparams: The dict containing at least all hyperparameters of the kernel.
        seed: The seed used for the training. Choose seed=None if this is a path to the merged times for all experiments.
    Returns:
        suffix (str): The string suffix used to save the times filename.
    """
    suffix = f"_{dataset_name}_{kernel}"

    if kernel == "swwl" or kernel == "aswwl":
        H = hparams["num_wl_iterations"]
        T = hparams["step_wl"]
        P = hparams["num_projections"]
        Q = hparams["num_quantiles"]
        if len(H) == 1:
            suffix += f"_H{H[0]}_P{P}_Q{Q}_T{T}"
        else:
            suffix += f"_P{P}_Q{Q}"
    elif kernel == "wwl" or kernel == "wwl_ER":
        H = hparams["num_wl_iterations"]
        T = hparams["step_wl"]
        if len(H) == 1:
            suffix += f"_H{H[0]}_T{T}"
    elif kernel == "fgw":
        alphas = hparams["fgw_alphas"]
        if len(alphas) == 1:
            suffix += f"_alpha{alphas[0]:.3f}"
    elif kernel == "sgml":
        H = hparams["num_wl_iterations"]
        if len(H) == 1:
            suffix += f"_H{H}"
    elif kernel == "propag":
        w = hparams["pk_w"]
        t_max = hparams["pk_tmax"]
        if len(t_max) == 1:
            suffix += f"_tmax{t_max[0]}"
        if len(w) == 1:
            suffix += f"_w{w[0]}"
    if seed is not None:
        suffix += f"_seed{seed}"
    return suffix


def suffix_matrices_filenames(
    dataset_name: str,
    kernel: str,
    seed: int,
    H: int = None,
    P: int = None,
    Q: int = None,
    T: int = None,
    t_max: int = None,
    w: float = None,
    alpha: float = None,
):
    """
    Function to build the path to distance/gram matrices filenames according to the dataset and the kernel. The hyperparameters depend on the kernel used.
    Final path = prefix + keyword + suffix + format.

    Args:
        dataset_name: The name of the dataset.
        kernel: The kernel or distance function (e.g. 'swwl' or 'fgw').
        seed: The seed used for the training.
        H: The number of continuous WL iterations (for SWWL/ASWWL/WWL/SGML only).
        P: The number of projections (for SWWL/ASWWL only).
        Q: The number of quantiles (for SWWL/ASWWL only).
        T: The step of continuous WL iterations (for SWWL/ASWWL/WWL only).
        t_max: The number of propagation iterations (for propag only).
        w: The bin width (for propag only).
        alpha: The balance parameter between Wasserstein and Gromov Wasserstein (for FGW only).
    Returns:
        suffix (str): The string suffix used to save the distance/gram matrices filename.
    """
    suffix = f"_{dataset_name}_{kernel}"
    if kernel == "swwl" or kernel == "aswwl":
        suffix += f"_H{H}_P{P}_Q{Q}_T{T}"
    elif kernel == "wwl" or kernel == "wwl_ER":
        suffix += f"_H{H}_T{T}"
    elif kernel == "fgw":
        suffix += f"_alpha{alpha:.3f}"
    elif kernel == "sgml":
        suffix += f"_H{H}"
    elif kernel == "propag":
        suffix += f"_tmax{t_max}_w{w}"
    suffix += f"_seed{seed}"
    return suffix


def prefix_filenames(root: str, dataset_name: str, kernel: str, seed: int = None):
    """
    Function to build the path to result filenames according to the dataset and the kernel.
    Final path = prefix + keyword + suffix + format.

    Args:
        root: The common root folder path for result files.
        dataset_name: The name of the dataset.
        kernel: The kernel or distance function (e.g. 'swwl' or 'fgw').
        seed: The seed used for the training. Choose seed=None if this is a path to the merged results for all experiments.
    Returns:
        prefix (str): The string prefix used to save the results.
    """
    if seed is not None:
        return os.path.join(root, dataset_name, kernel, f"exp{seed}")
    else:
        return os.path.join(root, dataset_name, kernel)


def save_scores_for_one_output(
    root_filenames: str, suffix_scores: str, scores_out: str, out: int = 0
):
    """
    Function to save scores corresponding to predictions for all test values of a single output in a .pkl format.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The folder where the file needs to be saved.
        suffix_scores: The string suffix used to save the results.
        scores_out: Any dict representing the test scores.
        out: The number of the output (0 if there is a single prediction task).
    """
    filename_scores = "scores" + suffix_scores + f"_out{out}.pkl"
    with open(os.path.join(root_filenames, filename_scores), "wb") as f:
        pickle.dump(scores_out, f)


def load_scores_for_one_output(root_filenames: str, suffix_scores: str, out: int = 0):
    """
    Function to load scores corresponding to predictions for all test values of a single output.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The folder where the file is saved.
        suffix_scores: The string suffix used to save the results.
        out: The number of the output (0 if there is a single prediction task).
    Returns:
        scores_out (dict): The dict representing the test scores. If not found, returns None.
    """
    filename_scores = "scores" + suffix_scores + f"_out{out}.pkl"
    file_scores = os.path.join(root_filenames, filename_scores)
    scores_out = None
    if os.path.exists(file_scores):
        with open(file_scores, "rb") as f:
            scores_out = pickle.load(f)
    return scores_out


def save_times_training_for_one_output(
    root_filenames: str, suffix_times: str, times_out: dict, out: int = 0
):
    """
    Function to save times of the training step (GP, SVM) corresponding to predictions for all test values of a single output in a .pkl format.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The folder where the file needs to be saved.
        suffix_times: The string suffix used to save the results.
        times_out: Any dict representing the times of the training step.
        out: The number of the output (0 if there is a single prediction task).
    """
    filename_times = "times_training" + suffix_times + f"_out{out}.pkl"
    with open(os.path.join(root_filenames, filename_times), "wb") as f:
        pickle.dump(times_out, f)


def load_times_training_for_one_output(
    root_filenames: str, suffix_times: str, out: int = 0
):
    """
    Function to load times of the training step (GP, SVM) corresponding to predictions for all test values of a single output.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The folder where the file is saved.
        suffix_times: The string suffix used to save the results.
        out: The number of the output (0 if there is a single prediction task).
    Returns:
        times_out (dict): The dict representing the training times. If not found, returns None.
    """
    filename_times = "times_training" + suffix_times + f"_out{out}.pkl"
    file_times = os.path.join(root_filenames, filename_times)
    times_out = None
    if os.path.exists(file_times):
        with open(file_times, "rb") as f:
            times_out = pickle.load(f)
    return times_out


def save_times(root_filenames: str, suffix_times: str, times: dict):
    """
    Function to save times of the embedding step (embeddings+ distances/Gram matrices) in a .pkl format.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The folder where the file needs to be saved.
        suffix_times: The string suffix used to save the results.
        times_out: Any dict representing the times of the embedding step.
    """
    filename_times = "times_matrices" + suffix_times + ".pkl"
    with open(os.path.join(root_filenames, filename_times), "wb") as f:
        pickle.dump(times, f)


def load_times(root_filenames: str, suffix_times: str):
    """
    Function to load times of the embedding step (embeddings+ distances/Gram matrices).
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The folder where the file is saved.
        suffix_times: The string suffix used to save the results.
    Returns:
        times_out (dict): The dict representing the embedding+distance/Gram matrices times. If not found, returns None.
    """
    filename_times = "times_matrices" + suffix_times + f".pkl"
    file_times = os.path.join(root_filenames, filename_times)
    times = None
    if os.path.exists(file_times):
        with open(file_times, "rb") as f:
            times = pickle.load(f)
    return times


def save_scalar_matrices(
    root: str, dataset_name: str, S_matrices: list, format: str = "npy"
):
    """
    Function to save scalar matrices in a .npy or .csv format. All matrices are stacked and saved in a single file.

    Args:
        root: The root folder of results.
        dataset_name: The name of the dataset.
        S_matrices: The list of scalar matrices to save in the file.
        format: The format is '.npy' or '.csv'.
    """
    if format == "npy":
        filename_S = os.path.join(root, dataset_name, f"S_{dataset_name}.npy")
        S_arr = np.stack(S_matrices)
        with open(filename_S, "wb") as f:
            np.save(f, S_arr)
    elif format == "csv":
        filename_S = os.path.join(root, dataset_name, f"S_{dataset_name}.csv")
        S_arr = np.concatenate(S_matrices)
        df = pd.DataFrame(S_arr)
        df.to_csv(filename_S, header=False, index=False)
    else:
        print(f"Save format {format} not recognized")
        sys.exit(1)


def load_scalar_matrices(root: str, dataset_name: str, format: str = "npy"):
    """
    Function to load scalar matrices saved in a .npy or .csv file.

    Args:
        root: The root folder of results.
        dataset_name: The name of the dataset.
        format: The format is '.npy' or '.csv'.
    Returns:
        S_matrices: The list of scalar matrices.
    """
    S_matrices = None
    if format == "npy":
        filename_S = os.path.join(root, dataset_name, f"S_{dataset_name}.npy")
        if os.path.exists(filename_S):
            with open(filename_S, "rb") as f:
                S_arr = np.load(f)
                S_matrices = [S for S in S_arr]
        return S_matrices
    elif format == "csv":
        filename_S = os.path.join(root, dataset_name, f"S_{dataset_name}.csv")
        if os.path.exists(filename_S):
            S_matrices_arr = pd.read_csv(filename_S, header=None).to_numpy()
            n_matrices = S_matrices_arr.shape[0] // S_matrices_arr.shape[1]
            S_matrices = np.split(S_matrices_arr, n_matrices)
        return S_matrices
    else:
        print(f"Save format {format} not recognized")
        sys.exit(1)


def save_matrices(
    root_filenames: str,
    suffix_matrices: str,
    matrices: list,
    format: str = "npy",
    kind: str = "distances",
):
    """
    Function to save distance/Gram matrices in a .npy or .csv format. If there are several matrices, they are stacked and saved in a single file.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The root folder prefix.
        suffix_matrices: The string suffix used to save the results.
        matrices: The list of distance/Gram matrices to save in the file.
        format: The format is '.npy' or '.csv'.
        kind: A hint to choose the keyword 'D' for distances and 'K' for Gram matrices.
    """
    letter = "K" if kind == "gram" else "D"
    if format == "npy":
        filename = os.path.join(root_filenames, letter + suffix_matrices + f".npy")
        arr = np.stack(matrices)
        with open(filename, "wb") as f:
            np.save(f, arr)
    elif format == "csv":
        filename = os.path.join(root_filenames, letter + suffix_matrices + f".csv")
        arr = np.concatenate(matrices)
        df = pd.DataFrame(arr)
        df.to_csv(filename, header=False, index=False)
    else:
        print(f"Save format {format} not recognized")
        sys.exit(1)


def load_matrices(
    root_filenames: str,
    suffix_matrices: str,
    format: str = "npy",
    kind: str = "distances",
):
    """
    Function to load distance/Gram matrices saved in a .npy or .csv format.
    Final path = prefix + keyword + suffix + format.

    Args:
        root_filenames: The root folder prefix.
        suffix_matrices: The string suffix used to save the results.
        format: The format is '.npy' or '.csv'.
        kind: A hint to choose the keyword 'D' for distances and 'K' for Gram matrices.
    Returns:
        matrices: The list of distance/Gram matrices.
    """
    letter = "K" if kind == "gram" else "D"
    if format == "npy":
        filename = os.path.join(root_filenames, letter + suffix_matrices + f".npy")
        matrices = None
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                arr = np.load(f)
                matrices = [M for M in arr]
        return matrices
    elif format == "csv":
        filename = os.path.join(root_filenames, letter + suffix_matrices + f".csv")
        matrices_arr = pd.read_csv(filename, header=None).to_numpy()
        n_matrices = matrices_arr.shape[0] // matrices_arr.shape[1]
        matrices = np.split(matrices_arr, n_matrices)
        return matrices
    else:
        print(f"Save format {format} not recognized")
        sys.exit(1)


def save_output_scalars(
    root: str, dataset_name: str, y: np.ndarray, format: str = "npy"
):
    """
    Function to save output scalars in a .npy or .csv format.

    Args:
        root: The root folder prefix.
        dataset_name: The name of the dataset.
        y: The array containing the output scalars.
        format: The format is '.npy' or '.csv'.
    """
    if format == "npy":
        filename_y = os.path.join(root, dataset_name, f"y_{dataset_name}.npy")
        with open(filename_y, "wb") as f:
            np.save(f, y)
    elif format == "csv":
        filename_y = os.path.join(root, dataset_name, f"y_{dataset_name}.csv")
        df = pd.DataFrame(y)
        df.to_csv(filename_y, header=False, index=False)
    else:
        print(f"Save format {format} not recognized")
        sys.exit(1)


# Auxiliary functions used to fuse attributes with the node label information when handling graphs for Grakel kernels.
# Only used for the Cuneiform dataset.
def read_just_node_labels(dataset_name: str, root_datasets: str = "./"):
    """
    Function to read the node labels of all graphs. The data should be in the following format: https://chrsmrrs.github.io/datasets/docs/format/.

    Args:
        dataset_name: The name of the dataset.
        root_datasets: The root where datasets are saved.
    Returns:
        node_labels (list): The list of node labels for each graph. node_labels[i] is a  np.ndarray of shape (n_i, M) where the i-th graph has n_i nodes, and there are M labels.
    """
    graph_sizes = []
    filename_graph_indicator = os.path.join(
        root_datasets, dataset_name, "raw", f"{dataset_name}_graph_indicator.txt"
    )
    with open(filename_graph_indicator, "r") as f:
        lines = f.readlines()
        graph_indicator = np.array([int(x.strip()) for x in lines])
        for i in range(max(graph_indicator)):
            graph_sizes.append(graph_indicator[graph_indicator == i + 1].shape[0])
        nb_graph = len(graph_sizes)

    filename = os.path.join(
        root_datasets, dataset_name, "raw", f"{dataset_name}_node_labels.txt"
    )
    print(filename)
    with open(filename, "r") as f:
        lines = f.readlines()
        labels_size = len(lines[0].replace(" ", "").split(","))
        node_labels = [np.zeros((x, labels_size)) for x in graph_sizes]
        k = 0
        for i in range(len(graph_sizes)):
            for j in range(graph_sizes[i]):
                node_labels[i][j, :] = [
                    float(x) for x in lines[k].replace(" ", "").split(",")
                ]
                k = k + 1
    return node_labels


def renumber_nodes_to_start_at_one(G: list):
    """
    Function to relabel the nodes of all graphs g in G to be {1, ..., n_g} if g has n_g nodes.

    Args:
        G: The list of graphs given as tuples (set of edges, dict asscociating attributes to all nodes, empty dict).
    Returns:
        G_bis (list): The transformed dataset.
    """
    G_bis = []
    for g in G:
        starting_vertex = np.min(list(g[1].keys()))
        # print(starting_vertex)
        g_bis_edges = {
            (e[0] - starting_vertex + 1, e[1] - starting_vertex + 1) for e in g[0]
        }
        nice = False
        for e in g_bis_edges:
            if e[0] == 1 or e[1] == 1:
                nice = True
        g_bis_features = {k - starting_vertex + 1: v for k, v in g[1].items()}

        G_bis.append((g_bis_edges, g_bis_features, g[2]))
    return G_bis


def fuse_labels_with_grakel_graphs(graphs, node_labels):
    """
    Function to fuse node labels to node attributes in all graphs g in G. Graphs nodes should be labeled {1, ..., n_g} in order to work correctly.

    Args:
        graphs: The list of graphs given as tuples (set of edges, dict asscociating attributes to all nodes, empty dict).
    Returns:
        G_bis (list): The transformed dataset.
    """
    l = []
    j = 0
    for g, labels in zip(graphs, node_labels):
        j += 1
        l.append(
            (
                g[0],
                {
                    i + 1: np.concatenate((g[1][i + 1], labels[i]))
                    for i in range(len(g[1]))
                },
                g[2],
            )
        )
    return l
