# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import torch
from torch_geometric.nn.conv import WLConv, WLConvContinuous


class WLEncoder(torch.nn.Module):
    """
    Encoder that transforms the node attributes using continuous Weisfeiler-Lehman iterations. This encoding step is deterministic.
    """

    def __init__(self, num_iterations: int, step: int = 1):
        """
        Args:
            num_iterations: The number of continuous WL iterations (message passing).
            step: The setp of continuous WL iterations. step=1 -> no skips.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.conv = WLConvContinuous()
        self.step = step

    def forward(self, x, edge_index):
        feats = [x]
        for _ in range(self.num_iterations):
            for _ in range(self.step):
                x = self.conv(x, edge_index)
            feats += [x]
        return torch.concat(feats, axis=-1)


class WLEncoder_Categorical(torch.nn.Module):
    """
    Encoder that transforms the categorical node labels using Weisfeiler-Lehman iterations. This encoding step is deterministic.
    """

    def __init__(self, num_iterations: int, step: int = 1):
        """
        Args:
            num_iterations: The number of WL iterations (message passing).
            step: The setp of WL iterations. step=1 -> no skips.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.conv = WLConv()
        self.step = step

    def forward(self, x, edge_index):
        x = torch.argmax(x, dim=1)
        feats = [x.reshape(-1, 1)]
        for _ in range(self.num_iterations):
            for _ in range(self.step):
                x = self.conv(x, edge_index)
            feats += [x.reshape(-1, 1)]
            # feats += [x]
        return torch.concat(feats, axis=-1)


class WLEncoderWeighted(torch.nn.Module):

    """
    Encoder that transforms the node attributes using continuous Weisfeiler-Lehman iterations when edge weights are not constant equal to. This encoding step is deterministic.
    """

    def __init__(self, num_iterations: int, step: int = 1):
        """
        Args:
            num_iterations: The number of continuous WL iterations (message passing).
            step: The setp of continuous WL iterations. step=1 -> no skips.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.conv = WLConvContinuous()

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.
        edge_weights = torch.sqrt(torch.sum(edge_attr**2, dim=1))
        row, col = edge_index
        feats = []
        for _ in range(self.num_iterations):
            x = self.conv(x, edge_index, edge_weights)
            dist = torch.norm(x[col] - x[row], p=2, dim=-1).view(-1, 1)
            dist = dist / dist.max()
            edge_weights = torch.sqrt(torch.sum(dist**2, dim=1))
            feats += [x]
        return torch.concat(feats, axis=-1)
