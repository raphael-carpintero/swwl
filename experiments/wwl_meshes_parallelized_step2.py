# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import argparse
import os
import sys
import time

import numpy as np
import yaml
from datasets_regression import available_mesh_datasets, load_scalars
from ot import dist, emd2
from utils import create_if_not_exists


def load_embeddings(
    dataset_name: str, h: int, T: int, indices: np.ndarray, root: str = "./"
):
    """
    Function to load several continuous WL embeddings.

    Args:
        dataset_name: The name of the dataset.
        h: The number of continuous WL iterations.
        T: The step of continuous WL iterations.
        indices: The indices of embeddings to load.
        root: The root of continuous WL embeddings.
    """
    root_dataset = os.path.join(root, dataset_name, f"H{h}_T{T}")
    embeddings = []
    for i in indices:
        filename = f"Emb_{i}.npy"
        if not os.path.exists(os.path.join(root_dataset, filename)):
            print("This is a problem...", i)
            print("F", os.path.join(root_dataset, filename))
            continue  # DEBUG
        with open(os.path.join(root_dataset, filename), "rb") as f:
            emb = np.load(f)
        embeddings.append(emb)
    return embeddings


def wasserstein_dist(mu: np.ndarray, nu: np.ndarray):
    # Computes the Wasserstein distance between the two empirical distributions mu, nu.
    costs = dist(mu, nu, metric="euclidean")
    return emd2([], [], costs)


# Note that using a torch backend instead doesn't seem to significantly improve the computation time.


def save_line(
    dataset_name: str,
    line_array: np.ndarray,
    h: int,
    T: int,
    line_number: int,
    root="./",
):
    """
    Function to save a line in a .npy format.
    This line can be loaded using using wwl_meshes_parallelized_step3.py's function 'load_line'.

    Args:
        dataset_name: The name of the dataset.
        line_array: The line array to save.
        h: The number of continuous WL iterations.
        T: The step of continuous WL iterations.
        line_number: The number of the line to load.
        root: The root of line files.
    """
    root_dataset = os.path.join(root, dataset_name, f"H{h}_T{T}")
    create_if_not_exists(root_dataset)
    filename = os.path.join(root_dataset, f"Line_{line_number}.npy")
    with open(filename, "wb") as f:
        np.save(f, line_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=available_mesh_datasets()
    )
    parser.add_argument(
        "-H",
        "--wl_iter",
        help="Number of continuous WL iterations",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-T", "--wl_step", help="Step of continuous WL iterations", type=int, default=1
    )
    parser.add_argument("--line", help="Number of line", required=True, type=int)

    args = parser.parse_args()
    with open("config.yml", "r") as f:
        config_params = yaml.safe_load(f)
    dataset_name = args.dataset

    h = args.wl_iter
    T = args.wl_step
    line_number = args.line
    root_embeddings = config_params["results"]["wwl_tmp_embeddings"]
    root_lines = config_params["results"]["wwl_tmp_lines"]

    print("Line ", line_number, file=sys.stderr)

    with open("config.yml", "r") as f:
        config_params = yaml.safe_load(f)
    y, (N_train, N_test) = load_scalars(
        dataset_name, roots=config_params["datasets"], fuse_train_test=True
    )
    N = N_train + N_test
    indices_global = np.arange(N)

    all_embeddings = load_embeddings(
        dataset_name, h, T, indices_global, root=root_embeddings
    )

    current_i = line_number
    total_time = 0
    while current_i < N - 1:
        reference_embedding = all_embeddings[current_i]
        other_embeddings = all_embeddings[current_i + 1 :]
        # For line i, we compute only distances between the ith input and inputs [i+1, ...N] as d(G_i,G_i)=0.
        line_array = []
        start0 = time.time()
        for embedding in other_embeddings:
            d = wasserstein_dist(embedding, reference_embedding)  # torch
            line_array.append(d)

        end0 = time.time()
        total_time += end0 - start0
        print(
            f"Time line {current_i} ({len(other_embeddings)} distances): {end0-start0} s"
        )

        save_line(dataset_name, np.array(line_array), h, T, current_i, root=root_lines)
        current_i += 100

    print("Dataset:", dataset_name)
    print("N graphs:", N)
    print("Total time:", total_time)
