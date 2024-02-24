# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import networkx as nx
import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, unbatch


def compute_graph_embeddings_fgw(data_loader: DataLoader, dim_attributes: int):
    """
    Function to compute the FGW graph 'embeddings'. Each 'embedding' is a tuple (attributes, shortest path matrix).

    Args:
        data_loader: A torch_geometric.loader.DataLoader object containing all graphs.
        dim_attributes: The dimension of the node attributes.
    Returns:
        graph_embeddings (list): A list containing the (attributes, shortest path matrix) tuples for all graphs.
    """

    # When the graph is not connected, we need to replace the 'inf' values by a default value (10x the highest value).
    def replace_inf(M):
        M[M == float("inf")] = 10 * np.max(M[M != float("inf")])
        return M

    graph_embeddings = []
    for data in data_loader:
        g_nx = to_networkx(data, to_undirected=True)
        graph_embeddings += [
            (
                data.x.cpu().numpy()[:, :dim_attributes],
                replace_inf(shortest_path(nx.adjacency_matrix(g_nx))),
            )
        ]
    return graph_embeddings


def compute_graph_embeddings(
    data_loader: DataLoader,
    encoder: torch.nn.Module,
    device: torch.device,
    wwl: bool = False,
):
    """
    Function to compute the graph embeddings of methods using Each 'embedding' is a tuple (attributes, shortest path matrix).

    Args:
        data_loader: A torch_geometric.loader.DataLoader object containing all graphs.
        encoder: The torch.nn encoder used to transform torch.Data into their embeddings.
        wwl: If this option is True, the encoder outputs continuous WL iterations of different sizes that cannot be stacked.
    Returns:
        graph_embeddings (list or np.ndarray): A list containing the embeddings if wwl is True. Otherwise, they are stacked in a np.ndarray.
    """

    # Computing embeddings
    graph_embeddings = []
    for data in data_loader:
        data = data.to(device)
        out = encoder(data)
        if wwl:
            for u in unbatch(out, data.batch):
                graph_embeddings += [u.cpu().numpy()]
        else:
            graph_embeddings += [out]

    if not wwl:
        graph_embeddings = torch.concat(graph_embeddings).cpu().numpy()

    return graph_embeddings
