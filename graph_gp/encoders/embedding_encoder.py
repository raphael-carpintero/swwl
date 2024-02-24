# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

# Remark: ContinuousAndCategorical variants are only used for the Cuneiform.

import numpy as np
import torch

from .sw_encoder import Anisotropic_SWEncoder, SWEncoder
from .wl_encoder import WLEncoder, WLEncoder_Categorical


class SWWL_Encoder(torch.nn.Module):
    """
    Encoder that computes the projected quantile embeddings of SWWL with the following steps:
    - transformation of the node attributes with the continuous WL iterations,
    - projection of the concatenated attributes onto the hypersphere with uniform random projections,
    - computation of the quantiles accross the nodes.

    Size of embeddings: num_projections*num_quantiles.
    """

    def __init__(
        self,
        dim_attributes: int,
        num_wl_iterations: int,
        num_projections: int,
        num_quantiles: int,
        step: int = 1,
        generator: torch.Generator = None,
    ):
        """
        Args:
            dim_attributes: The dimension of node attributes.
            num_wl_iterations: The number of continuous WL iterations.
            num_projections: The number of projections for the MC estimation of the sliced Wasserstein distance.
            num_quantiles: The number of quantiles for the estimation of 1-d Wasserstein distances.
            step: The setp of continuous WL iterations. step=1 -> no skips.
            generator: The torch random number generator (for reproducibility).
        """
        super().__init__()
        self.wl_encoder = WLEncoder(num_wl_iterations, step=step)
        size_after_wl_iterations = dim_attributes * (num_wl_iterations + 1)
        self.sw_encoder = SWEncoder(
            size_after_wl_iterations,
            num_projections,
            num_quantiles,
            generator=generator,
        )
        self.dim_attributes = dim_attributes

    def forward(self, data):
        x, edge_index, batch = (
            data.x[:, : self.dim_attributes],
            data.edge_index,
            data.batch,
        )
        x = self.wl_encoder(x, edge_index)
        x = self.sw_encoder(x, batch)
        return x


class ASWWL_Encoder(torch.nn.Module):
    """
    Encoder that computes the projected quantile embeddings of SWWL with the following steps:
    - transformation of the node attributes with the continuous WL iterations,
    - for each WL iteration:
        - projection of the concatenated attributes onto the hypersphere with uniform random projections,
        - computation of the quantiles accross the nodes.

    Size of embeddings: (num_wl_iterations+1)*num_projections*num_quantiles).
    """

    def __init__(
        self,
        dim_attributes: int,
        num_wl_iterations: int,
        num_projections: int,
        num_quantiles: int,
        step: int = 1,
        generator: torch.Generator = None,
    ):
        """
        Args:
            dim_attributes: The dimension of node attributes.
            num_wl_iterations: The number of continuous WL iterations.
            num_projections: The number of projections for the MC estimation of the sliced Wasserstein distance.
            num_quantiles: The number of quantiles for the estimation of 1-d Wasserstein distances.
            step: The setp of continuous WL iterations. step=1 -> no skips.
            generator: The torch random number generator (for reproducibility).
        """
        super().__init__()
        self.wl_encoder = WLEncoder(num_wl_iterations, step=step)
        self.sw_encoder = Anisotropic_SWEncoder(
            dim_attributes,
            num_wl_iterations,
            num_projections,
            num_quantiles,
            generator=generator,
        )
        self.dim_attributes = dim_attributes

    def forward(self, data):
        x, edge_index, batch = (
            data.x[:, : self.dim_attributes],
            data.edge_index,
            data.batch,
        )
        x = self.wl_encoder(x, edge_index)
        x = self.sw_encoder(x, batch)
        return x


class WWL_Encoder(torch.nn.Module):
    """
    Encoder that transforms graphs into a point clouds using the continuous WL iterations.

    Size of embeddings: (For the i-th graph with n_i nodes) n_i*(num_wl_iterations+1)*dim_attributes
    """

    def __init__(self, dim_attributes: int, num_wl_iterations: int, step: int = 1):
        """
        Args:
            dim_attributes: The dimension of node attributes.
            num_wl_iterations: The number of continuous WL iterations.
            step: The setp of continuous WL iterations. step=1 -> no skips.
        """
        super().__init__()
        self.wl_encoder = WLEncoder(num_wl_iterations, step=step)
        self.dim_attributes = dim_attributes

    def forward(self, data):
        x, edge_index = data.x[:, : self.dim_attributes], data.edge_index
        x = self.wl_encoder(x, edge_index)
        return x


class WWL_Encoder_ContinuousAndCategorical(torch.nn.Module):
    """
    Encoder that transforms graphs into a point clouds using the continuous WL iterations.
    Here, we take into account several one-hot encoded categorical labels for nodes.
    WL iterations are performed separately for continuous attributes and for each categorical variable.

    Size of embeddings: (For the i-th graph with n_i nodes) n_i*(num_wl_iterations+1)*(dim_attributes+num_labels)
    """

    def __init__(
        self,
        dim_attributes: int,
        dims_one_hot_encodings: list,
        num_wl_iterations: int,
        step: int = 1,
    ):
        """
        Args:
            dim_attributes: The dimension of node attributes.
            dim_one_hot_encodings: The list of the dimensions of one hot encodings. The length of the lits gives the ffective number of node labels.
            num_wl_iterations: The number of continuous WL iterations.
            step: The setp of continuous WL iterations. step=1 -> no skips.
        """
        super().__init__()
        self.wl_encoder_continuous = WLEncoder(num_wl_iterations, step=step)
        self.dim_attributes = dim_attributes
        self.wl_encoder_categorical = WLEncoder_Categorical(
            num_wl_iterations, step=step
        )
        self.dims_one_hot_encodings = np.array(dims_one_hot_encodings)

    def forward(self, data):
        x_continuous, edge_index, batch = (
            data.x[:, : self.dim_attributes],
            data.edge_index,
            data.batch,
        )
        x_continuous = self.wl_encoder_continuous(x_continuous, edge_index)

        x_cont_and_cat = [x_continuous]
        for i in range(len(self.dims_one_hot_encodings)):
            start = self.dim_attributes + np.sum(self.dims_one_hot_encodings[:i])
            end = self.dim_attributes + np.sum(self.dims_one_hot_encodings[: i + 1])
            x_categorical = data.x[:, start:end]
            x_categorical = self.wl_encoder_categorical(x_categorical, edge_index)
            x_cont_and_cat.append(x_categorical)

        x_cont_and_cat = torch.concat(x_cont_and_cat, axis=-1)
        return x_cont_and_cat


class SWWL_Encoder_ContinuousAndCategorical(torch.nn.Module):
    """
    Encoder that computes the projected quantile embeddings of SWWL with the following steps:
    - transformation of the node attributes with the continuous WL iterations,
    - projection of the concatenated attributes onto the hypersphere with uniform random projections,
    - computation of the quantiles accross the nodes.
    Here, we take into account several one-hot encoded categorical labels for nodes.
    In practice, we simply separate continuous and categorical information, compute their embeddings, and then merge the projected quantile embeddings.

    Size of embeddings: 2*num_projections*num_quantiles.
    """

    def __init__(
        self,
        dim_attributes: int,
        dims_one_hot_encodings: list,
        num_wl_iterations: int,
        num_projections: int,
        num_quantiles: int,
        step: int = 1,
        generator: torch.Generator = None,
    ):
        """
        Args:
            dim_attributes: The dimension of node attributes.
            dim_one_hot_encodings: The list of the dimensions of one hot encodings. The length of the lits gives the ffective number of node labels.
            num_wl_iterations: The number of continuous WL iterations.
            num_projections: The number of projections for the MC estimation of the sliced Wasserstein distance.
            num_quantiles: The number of quantiles for the estimation of 1-d Wasserstein distances.
            step: The setp of continuous WL iterations. step=1 -> no skips.
            generator: The torch random number generator (for reproducibility).
        """
        super().__init__()
        self.wl_encoder_continuous = WLEncoder(num_wl_iterations, step=step)
        self.dim_attributes = dim_attributes
        size_after_wl_iterations_cont = dim_attributes * (num_wl_iterations + 1)
        self.sw_encoder_continuous = SWEncoder(
            size_after_wl_iterations_cont,
            num_projections,
            num_quantiles,
            generator=generator,
        )

        self.wl_encoder_categorical = WLEncoder_Categorical(
            num_wl_iterations, step=step
        )
        self.dims_one_hot_encodings = np.array(dims_one_hot_encodings)
        size_after_wl_iterations_cat = len(dims_one_hot_encodings) * (
            num_wl_iterations + 1
        )
        self.sw_encoder_categorical = SWEncoder(
            size_after_wl_iterations_cat,
            num_projections,
            num_quantiles,
            generator=generator,
        )

    def forward(self, data):
        x_continuous, edge_index, batch = (
            data.x[:, : self.dim_attributes],
            data.edge_index,
            data.batch,
        )
        x_continuous = self.wl_encoder_continuous(x_continuous, edge_index)
        x_continuous = self.sw_encoder_continuous(x_continuous, batch)

        x_cat = []
        for i in range(len(self.dims_one_hot_encodings)):
            start = self.dim_attributes + np.sum(self.dims_one_hot_encodings[:i])
            end = self.dim_attributes + np.sum(self.dims_one_hot_encodings[: i + 1])
            x_categorical = data.x[:, start:end]
            x_categorical = self.wl_encoder_categorical(x_categorical, edge_index)
            x_cat.append(x_categorical)
        x_categorical = torch.concat(x_cat, axis=-1).to(torch.float)
        x_categorical = self.sw_encoder_categorical(x_categorical, batch)

        x_cont_and_cat = torch.concat([x_continuous, x_categorical], axis=-1)
        return x_cont_and_cat
