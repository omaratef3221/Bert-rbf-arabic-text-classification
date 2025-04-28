#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Alessio Russo [alessior@kth.se]. All rights reserved.
#
# This file is part of PytorchRBFLayer.
#
# PytorchRBFLayer is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with PytorchRBFLayer.
# If not, see <https://opensource.org/licenses/MIT>.
#
import torch
import torch.nn as nn

from typing import Callable

def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)

# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

def multiquadric_kernel(eps_r):
    return torch.sqrt(eps_r ** 2 + 1)

def linear_kernel(x):
    return x  # Linear kernel simply returns the input

def rbf_laplacian(x):
    return (-x.abs()).exp()

def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Alessio Russo [alessior@kth.se]. All rights reserved.
#
# This file is part of PytorchRBFLayer.
#
# PytorchRBFLayer is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with PytorchRBFLayer.
# If not, see <https://opensource.org/licenses/MIT>.
#
import torch
import torch.nn as nn
from typing import Callable

class RBFLayer(nn.Module):
    def __init__(self,
                 in_features_dim: int,
                 num_kernels: int,
                 out_features_dim: int,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 norm_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True):
        super(RBFLayer, self).__init__()

        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization

        # Centers: (num_kernels, in_features_dim)
        self.kernels_centers = nn.Parameter(torch.empty(num_kernels, in_features_dim))
        nn.init.xavier_uniform_(self.kernels_centers)

        # Shapes: (num_kernels,)
        self.log_shapes = nn.Parameter(torch.empty(num_kernels))
        nn.init.normal_(self.log_shapes, mean=0.0, std=0.1)

        # Weights for combining kernel outputs: (out_features_dim, num_kernels)
        self.weights = nn.Parameter(torch.empty(out_features_dim, num_kernels))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, in_features_dim) or (batch_size, 1, in_features_dim)
        Output: (batch_size, out_features_dim)
        """

        # If input is (B, 1, Fin) squeeze it
        if input.dim() == 3 and input.size(1) == 1:
            input = input.squeeze(1)

        batch_size = input.size(0)

        # Broadcasting: (batch_size, num_kernels, in_features_dim)
        diff = input.unsqueeze(1) - self.kernels_centers.unsqueeze(0)

        # Norm across last dim
        r = self.norm_function(diff)  # (batch_size, num_kernels)

        # Scale by shape parameters
        eps_r = self.log_shapes.exp().unsqueeze(0) * r  # (batch_size, num_kernels)

        # Apply radial function
        rbfs = self.radial_function(eps_r)  # (batch_size, num_kernels)

        if self.normalization:
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1, keepdim=True))

        # Linear combination
        out = torch.matmul(rbfs, self.weights.t())  # (batch_size, out_features_dim)

        return out

    @property
    def get_kernels_centers(self):
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        return self.weights.detach()

    @property
    def get_shapes(self):
        return self.log_shapes.detach().exp()