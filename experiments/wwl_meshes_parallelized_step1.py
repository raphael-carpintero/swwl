# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import argparse
import os
import time

import numpy as np
import torch
import yaml
from datasets_regression import available_mesh_datasets, load_mesh_dataset
from utils import (
    create_if_not_exists,
    prefix_filenames,
    save_times,
    suffix_times_filenames,
)

from graph_gp.embeddings import compute_graph_embeddings
from graph_gp.encoders import WWL_Encoder


def compute_embeddings_wwl(dataset_name: str, h: int, T: int, config_params: dict):
    """
    Function to compute continuous WL embeddings for a given dataset.

    Args:
        dataset_name: The name of the dataset.
        h: The number of continuous WL iterations.
        T: The step of continuous WL iterations.
        config_params: The configuration dict loaded from config.yml
    Returns:
        graph_embeddings (list): The list of continuous WL embeddings given as np.ndarray objects.
        times (dict): The total times needed to load the dataset and to compute the embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time_dataset = time.time()
    data_loader, X_scalars, _, dim_attributes, (_, _) = load_mesh_dataset(
        dataset_name,
        config_params["datasets"],
        batch_size=config_params["embeddings"]["batch_size_big"],
        N_train=None,
        N_test=None,
        fuse_train_test=True,
    )
    end_time_dataset = time.time()

    start_time_embeddings = time.time()
    encoder = WWL_Encoder(dim_attributes, h, step=T)
    graph_embeddings = compute_graph_embeddings(data_loader, encoder, device, wwl=True)
    end_time_embeddings = time.time()

    times = {
        "time_dataset": end_time_dataset - start_time_dataset,
        "time_embeddings": end_time_embeddings - start_time_embeddings,
    }
    return graph_embeddings, times


def save_embeddings_separately(
    dataset_name: str, embeddings: list, h: int, T: int, root: str = "./data/"
):
    """
    Function to save all continuous WL embeddings for a given dataset.

    Args:
        dataset_name: The name of the dataset.
        embeddings: The list of continuous WL embeddings given as np.ndarray objects.
        h: The number of continuous WL iterations.
        T: The step of continuous WL iterations.
        root: The root of continuous WL embeddings.
    """
    root_dataset = os.path.join(root, dataset_name, f"H{h}_T{T}")
    create_if_not_exists(root_dataset)
    for i in range(len(embeddings)):
        filename = f"Emb_{i}.npy"
        with open(os.path.join(root_dataset, filename), "wb") as f:
            np.save(f, embeddings[i])


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
        "-T",
        "--wl_step",
        help="For wwl/swwl/aswwl: setp for iterations of WL",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    with open("config.yml", "r") as f:
        config_params = yaml.safe_load(f)
    dataset_name = args.dataset

    h = args.wl_iter
    T = args.wl_step
    root_embeddings = config_params["results"]["wwl_tmp_embeddings"]

    print("wwl, ", dataset_name, h, T)
    embeddings, times = compute_embeddings_wwl(dataset_name, h, T, config_params)
    print("N =", len(embeddings))
    print("Time WL iterations:", times["time_embeddings"])

    save_embeddings_separately(dataset_name, embeddings, h, T, root=root_embeddings)
    hparams = {"num_wl_iterations": [h], "step_wl": T}
    save_root_times_seed = prefix_filenames(
        config_params["results"]["times"], dataset_name, "wwl", seed=0
    )
    suffix_times = suffix_times_filenames(dataset_name, "wwl", hparams, seed=0)
    save_times(save_root_times_seed, suffix_times, times)
