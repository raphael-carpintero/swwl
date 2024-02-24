# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import torch
from torch_geometric.utils import unbatch


class SWEncoder(torch.nn.Module):
    """
    Encoder that computes the quantiles of the randomly projected nodes attributes in view of computing a Sliced Wasserstein kernel.
    The projection directions are fixed once for all.
    """

    def __init__(
        self,
        in_channels: int,
        num_projections: int,
        num_quantiles: int,
        generator: torch.Generator = None,
        p: int = 2,
    ):
        """
        Args:
            in_channels: The dimension of node attributes after the continuous WL embeddings.
            num_projections: The number of projections for the MC estimation of the sliced Wasserstein distance.
            num_quantiles: The number of quantiles for the estimation of 1-d Wasserstein distances.
            generator: The torch random number generator (for reproducibility).
            p: The p-sliced Wasserstein is computed (in practice, we use p=2).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_projections = num_projections
        self.num_quantiles = num_quantiles
        if generator is None:
            generator = torch.Generator()
        self.generator = generator
        self.p = p

        self.resample_projections()
        self.register_buffer(
            "cum_weights", torch.linspace(0.0, 1.0, self.num_quantiles)
        )

    def resample_projections(self):
        projections = torch.randn(
            (self.in_channels, self.num_projections), generator=self.generator
        )
        projections = projections / torch.sqrt(
            torch.sum(projections**2, 0, keepdims=True)
        )
        self.register_buffer("projections", projections)

    def set_projections(self, projections):
        self.register_buffer("projections", projections)

    def forward(self, x, batch):
        x = torch.matmul(x, self.projections)
        quantiles = []
        for xb in unbatch(x, batch):
            quantiles += [
                torch.quantile(
                    xb, self.cum_weights, interpolation="lower", axis=0
                ).flatten()
            ]
        quantiles = torch.stack(quantiles)
        return quantiles / (self.num_quantiles * self.num_projections) ** (1.0 / self.p)


class Anisotropic_SWEncoder(torch.nn.Module):
    """
    Encoder that computes the quantiles of the randomly projected nodes attributes(separately for each WL iteration) in view of computing a Sliced Wasserstein kernel.
    The projection directions are fixed once for all. All WL iterations share the same projections.
    """

    def __init__(
        self,
        dim_features: int,
        num_iterations: int,
        num_projections: int,
        num_quantiles: int,
        generator: torch.Generator = None,
        p: int = 2,
    ):
        """
        Args:
            dim_features: The original dimension of node attributes.
            num_iterations: The number of continuous WL iterations.
            num_projections: The number of projections for the MC estimation of the sliced Wasserstein distance.
            num_quantiles: The number of quantiles for the estimation of 1-d Wasserstein distances.
            generator: The torch random number generator (for reproducibility).
            p: The p-sliced Wasserstein is computed (in practice, we use p=2).
        """
        super().__init__()
        self.dim_features = dim_features
        self.num_iterations = num_iterations
        self.num_projections = num_projections
        self.num_quantiles = num_quantiles
        if generator is None:
            generator = torch.Generator()
        self.generator = generator
        self.p = p

        self.resample_projections()
        self.register_buffer(
            "cum_weights", torch.linspace(0.0, 1.0, self.num_quantiles)
        )

    def resample_projections(self):
        projections = torch.randn(
            (self.dim_features, self.num_projections), generator=self.generator
        )
        # same projections for all
        projections = projections / torch.sqrt(
            torch.sum(projections**2, 0, keepdims=True)
        )
        self.register_buffer("projections", projections)

    def set_projections(self, projections):
        self.register_buffer("projections", projections)

    def forward(self, x, batch):
        quantiles = []
        for i in range(self.num_iterations + 1):
            # WWL_0, ..., WWL_H
            x_i = torch.matmul(
                x[:, i * self.dim_features : (i + 1) * self.dim_features],
                self.projections,
            )
            quantiles_i = []
            for xb_i in unbatch(x_i, batch):
                quantiles_i += [
                    torch.quantile(
                        xb_i, self.cum_weights, interpolation="lower", axis=0
                    ).flatten()
                ]
            quantiles_i = torch.stack(quantiles_i)
            quantiles_i = quantiles_i / (self.num_quantiles * self.num_projections) ** (
                1.0 / self.p
            )
            quantiles += [quantiles_i]
        return torch.concat(quantiles, axis=-1)
