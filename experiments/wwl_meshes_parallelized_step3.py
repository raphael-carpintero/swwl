# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import argparse
import os

import numpy as np
import yaml
from datasets_regression import available_mesh_datasets, load_scalars
from utils import (
    create_if_not_exists,
    prefix_filenames,
    save_matrices,
    suffix_matrices_filenames,
)


def load_line(dataset_name: str, h: int, T: int, line_number: int, root: str = "./"):
    """
    Function to load a line saved in a .npy format using using wwl_meshes_parallelized_step2.py's function 'save_line'.

    Args:
        dataset_name: The name of the dataset.
        h: The number of continuous WL iterations.
        T: The step of continuous WL iterations.
        line_number: The number of the line to load.
        root: The root of line files.
    Returns:
        line_array (np.ndarray): The (N,1) array containing the loaded distance values for the current line.
    """
    root_dataset = os.path.join(root, dataset_name, f"H{h}_T{T}")
    filename = os.path.join(root_dataset, f"Line_{line_number}.npy")
    with open(filename, "rb") as f:
        line_array = np.load(f)
    return line_array


if __name__ == "__main__":
    available_mesh_datasets = available_mesh_datasets()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=available_mesh_datasets
    )
    parser.add_argument(
        "-H", "--wl_iter", help="Number of WL iterations", type=int, default=3
    )
    parser.add_argument(
        "-T", "--wl_step", help="Step for WL iterations", type=int, default=1
    )

    args = parser.parse_args()
    with open("config.yml", "r") as f:
        config_params = yaml.safe_load(f)
    dataset_name = args.dataset

    h = args.wl_iter
    T = args.wl_step
    kernel = args.kernel

    root_lines = config_params["results"]["wwl_tmp_lines"]

    y, (N_train, N_test) = load_scalars(
        dataset_name, roots=config_params["datasets"], fuse_train_test=True
    )
    N = N_train + N_test
    D = np.zeros((N, N))
    for i in range(0, N - 1):
        line_i = load_line(dataset_name, h, T, i, root=root_lines)
        if i % 100 == 0:
            print(i, line_i.shape)
        D[i, i + 1 :] = line_i
        D[i + 1 :, i] = line_i
        # For line i, we saved only distances between the ith input and inputs [i+1, ...N] as d(G_i,G_i)=0.

    seed = 0
    save_root_distances_seed = prefix_filenames(
        config_params["results"]["distances"], dataset_name, kernel, seed=seed
    )
    create_if_not_exists(save_root_distances_seed)
    suffix_distances = suffix_matrices_filenames(dataset_name, kernel, seed, H=h, T=T)
    save_matrices(save_root_distances_seed, suffix_distances, [D], kind="distances")
