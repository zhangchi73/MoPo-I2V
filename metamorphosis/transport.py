"""Move a residual defined on the frame-t grid to the frame-r grid through the flow maps."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_sample_vox(vol, disp):
    B, _, Z, Y, X = vol.shape
    dev = vol.device
    gz, gy, gx = torch.meshgrid(torch.arange(Z, device=dev, dtype=vol.dtype), torch.arange(Y, device=dev, dtype=vol.dtype),
                                torch.arange(X, device=dev, dtype=vol.dtype), indexing="ij")
    nz = (gz[None] + disp[:, 0]) / max(Z - 1, 1) * 2 - 1
    ny = (gy[None] + disp[:, 1]) / max(Y - 1, 1) * 2 - 1
    nx = (gx[None] + disp[:, 2]) / max(X - 1, 1) * 2 - 1
    return F.grid_sample(vol, torch.stack([nx, ny, nz], -1), mode="bilinear", padding_mode="border", align_corners=True)


def transport(s_t, phi_t, phi_r, iters=3):
    """(s_t o U)(x) = s_t(x + phi_r(x) + psi_t(y)), psi_t = inverse of phi_t by fixed-point iteration."""
    psi = -grid_sample_vox(phi_t, phi_r)
    for _ in range(iters):
        psi = -grid_sample_vox(phi_t, phi_r + psi)
    return grid_sample_vox(s_t, phi_r + psi)
