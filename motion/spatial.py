"""Explicit 3D displacement-field convention.

* volume layout ``[B,C,Z,Y,X]``, flow layout ``[B,3,Z,Y,X]`` with channels ``[dz,dy,dx]`` in voxels
* backward sampling: ``out(x) = source(x + flow(x))``, ``align_corners=True``, ``padding_mode='border'``
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_grid(shape: Sequence[int], device, dtype) -> torch.Tensor:
    vectors = [torch.arange(int(size), device=device, dtype=dtype) for size in shape]
    grid_z, grid_y, grid_x = torch.meshgrid(vectors, indexing="ij")
    return torch.stack((grid_z, grid_y, grid_x), dim=0).unsqueeze(0)


class SpatialTransformer(nn.Module):
    """Warp a 3D tensor with a dense voxel displacement field."""

    def __init__(self, shape_zyx: Sequence[int], mode: str = "bilinear",
                 padding_mode: str = "border", align_corners: bool = True):
        super().__init__()
        self.shape_zyx = tuple(int(v) for v in shape_zyx)
        self.mode, self.padding_mode, self.align_corners = mode, padding_mode, bool(align_corners)
        self.register_buffer("grid_zyx", _identity_grid(self.shape_zyx, "cpu", torch.float32), persistent=False)

    def forward(self, source: torch.Tensor, flow_zyx: torch.Tensor) -> torch.Tensor:
        grid = self.grid_zyx.to(device=flow_zyx.device, dtype=flow_zyx.dtype)
        z_size, y_size, x_size = self.shape_zyx
        flow = flow_zyx
        if min(self.shape_zyx) == 1:
            keep = flow.new_tensor([float(z_size > 1), float(y_size > 1), float(x_size > 1)]).view(1, 3, 1, 1, 1)
            flow = flow * keep
        loc = grid + flow
        z = 2.0 * loc[:, 0] / float(max(z_size - 1, 1)) - 1.0
        y = 2.0 * loc[:, 1] / float(max(y_size - 1, 1)) - 1.0
        x = 2.0 * loc[:, 2] / float(max(x_size - 1, 1)) - 1.0
        return F.grid_sample(source, torch.stack((x, y, z), dim=-1), mode=self.mode,
                             padding_mode=self.padding_mode, align_corners=self.align_corners)


def compose_displacements(first: torch.Tensor, second: torch.Tensor,
                          transformer: SpatialTransformer) -> torch.Tensor:
    """Compose backward-sampling displacements: ``first + warp(second, first)``."""
    return first + transformer(second, first)


def jacobian_determinant(flow_zyx: torch.Tensor) -> torch.Tensor:
    """``det(d(x + flow)/dx)`` on the forward-difference interior grid."""
    grid = _identity_grid(flow_zyx.shape[2:], flow_zyx.device, flow_zyx.dtype)
    phi = grid + flow_zyx
    base = phi[:, :, :-1, :-1, :-1]
    d_dz = phi[:, :, 1:, :-1, :-1] - base
    d_dy = phi[:, :, :-1, 1:, :-1] - base
    d_dx = phi[:, :, :-1, :-1, 1:] - base
    j00, j10, j20 = d_dz[:, 0], d_dz[:, 1], d_dz[:, 2]
    j01, j11, j21 = d_dy[:, 0], d_dy[:, 1], d_dy[:, 2]
    j02, j12, j22 = d_dx[:, 0], d_dx[:, 1], d_dx[:, 2]
    return (j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20))
