"""Motion-Posterior Metamorphosis, transport step: warp the first volume along a flow map with a
3D cubic B-spline kernel (scipy ``map_coordinates(order=3)``, with prefiltering)."""
from __future__ import annotations

import numpy as np
import torch


def warp_bspline3d(volume: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """volume [B,1,Z,Y,X], flow [B,3,Z,Y,X] (dz,dy,dx) in voxels -> volume(x + flow(x))."""
    from scipy.ndimage import map_coordinates
    b, _, Z, Y, X = volume.shape
    gz, gy, gx = np.meshgrid(np.arange(Z, dtype=np.float64), np.arange(Y, dtype=np.float64),
                             np.arange(X, dtype=np.float64), indexing="ij")
    vol = volume[:, 0].double().cpu().numpy()
    fl = flow.double().cpu().numpy()
    outs = []
    for m in range(b):
        coords = np.stack([gz + fl[m, 0], gy + fl[m, 1], gx + fl[m, 2]])
        outs.append(map_coordinates(vol[m], coords, order=3, mode="nearest"))
    return torch.from_numpy(np.stack(outs)).to(volume.device).float().unsqueeze(1)


def zero_pad_slot0(phi: torch.Tensor) -> torch.Tensor:
    """[.., K, 3, Z,Y,X] (t=1..K) -> [.., K+1, 3, Z,Y,X] with slot 0 = zero field (t=0)."""
    z = torch.zeros_like(phi[..., :1, :, :, :, :])
    return torch.cat([z, phi], dim=-5)
