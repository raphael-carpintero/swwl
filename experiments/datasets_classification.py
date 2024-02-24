# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def load_classification_dataset(
    dataset_name: str, batch_size: int = 32, root_datasets: str = "./data/"
):
    """
    Function to load TU datasets (https://chrsmrrs.github.io/datasets/docs/datasets/) using torch geometric's DataLoader.
    The data loader gathers all graphs and their associated outputs.

    Args:
        dataset_name: The name of the dataset (for instance 'BZR', 'COX2', 'PROTEINS', 'ENZYMES', 'Cuneiform',...).
        batch_size: The batch size used in torch geometric's loaders.
        root_datasets: The path to the datasets. If data do not exist, they will be donwloaded and saved at this location.
    Returns:
        data_loader (DataLoader): the DataLoader object containing all graphs.
        dim_attributes (int): The dimension of the attributes of the nodes.
    """
    dataset = TUDataset(
        root=root_datasets, name=dataset_name, use_node_attr=True, use_edge_attr=False
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    # When using the option use_node_attr, the DataLoader concatenates attributes and one hot encoding of node degrees. We need to give the dimension of attributes manually.
    # If you choose another dataset, please add the correct dimension size here.
    if dataset_name == "PROTEINS":
        dim_attributes = 1
    elif dataset_name == "ENZYMES":
        dim_attributes = 18
    else:
        dim_attributes = 3
    return data_loader, dim_attributes


def load_scalars(dataset_name: str, root_datasets: str = "./data/"):
    """
    Function to load TU datasets (https://chrsmrrs.github.io/datasets/docs/datasets/) using torch geometric's DataLoader.
    Only scalar outputs are loaded by this function.

    Args:
        dataset_name: The name of the dataset (for instance 'BZR', 'COX2', 'PROTEINS', 'ENZYMES', 'Cuneiform',...).
        root_datasets: The path to the datasets. If data do not exist, they will be donwloaded and saved at this location.
    Returns:
        y (np.ndarray): A (N,1) array containing the output associated with each graph in the dataset.
    """
    dataset = TUDataset(
        root=root_datasets, name=dataset_name, use_node_attr=True, use_edge_attr=False
    )
    data_loader_y = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    y = []
    for data in data_loader_y:
        y += [data.y]
    y = torch.concat(y).cpu().numpy().reshape((-1, 1))
    return y
