"""Window dataset and conditioning helpers for Motion Generation.

Field cache: one ``<case>.npz`` per case with
    phi_gen_norm  [n_win, K, 3, z, y, x]  fp16  teacher displacement fields phi_{s->s+t}, t=1..K, in the
                                             generation space (avg-pooled by POOL) divided by norm_const
    z0            [n_win, 4, z, y, x]     fp16  latent of the anchor frame I_s (frozen VAE)
    phase         [n_win, K]              float frame phase (s+t)/(T-1)
    anchor_phase  [n_win]                 float anchor phase s/(T-1)
    amp           [n_win]                 float motion amplitude of the window
    win_starts, case_id, norm_const
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

K = 7
SHAPE_PX = (32, 128, 128)
SPACING_ZYX = (3.12, 1.5, 1.5)   # voxel spacing in mm; MAISI expects spacing * 100
MODALITY = 3                      # MAISI modality class label


def unet_call(unet, x, t, dev, spacing=SPACING_ZYX, modality=MODALITY):
    n = x.shape[0]
    sp = torch.tensor([spacing], dtype=torch.float32) * 100.0
    return unet(x=x, timesteps=t, spacing_tensor=sp.expand(n, -1).to(dev),
                class_labels=torch.full((n,), modality, dtype=torch.long, device=dev))


def cond_emb(cond, phase, aph, amp, drop_mask=None):
    """Three linear-MLP increments; drop_mask=True zeroes the amplitude condition (CFG dropout)."""
    e = cond["phase"](phase) + cond["anchor_phase"](aph)
    a = cond["amp"](amp)
    if drop_mask is not None:
        a = a * (~drop_mask).float().unsqueeze(-1)
    return e + a


def cat_z0(x, z0):
    if z0.shape[-3:] != x.shape[-3:]:
        z0 = F.interpolate(z0, size=x.shape[-3:], mode="trilinear", align_corners=True)
    return torch.cat([x, z0], 1)


def to_delivery(phi_norm, norm_const, gain=1.0, shape_px=SHAPE_PX):
    """Generation-space field -> full-resolution voxel displacement (times the calibration gain)."""
    return F.interpolate(phi_norm * norm_const, size=shape_px, mode="trilinear", align_corners=True) * gain


def neg_jac(phi):
    """Fraction of voxels with non-positive Jacobian determinant."""
    dz = phi[:, :, 1:, :-1, :-1] - phi[:, :, :-1, :-1, :-1]
    dy = phi[:, :, :-1, 1:, :-1] - phi[:, :, :-1, :-1, :-1]
    dx = phi[:, :, :-1, :-1, 1:] - phi[:, :, :-1, :-1, :-1]
    j = torch.stack([dz, dy, dx], 2) + torch.eye(3, device=phi.device)[None, :, :, None, None, None]
    det = (j[:, 0, 0] * (j[:, 1, 1] * j[:, 2, 2] - j[:, 1, 2] * j[:, 2, 1])
           - j[:, 0, 1] * (j[:, 1, 0] * j[:, 2, 2] - j[:, 1, 2] * j[:, 2, 0])
           + j[:, 0, 2] * (j[:, 1, 0] * j[:, 2, 1] - j[:, 1, 1] * j[:, 2, 0]))
    return float((det <= 0).float().mean())


class Windows:
    """Whole-window dataset: K fields + shared z0 / anchor phase / amplitude per window."""

    def __init__(self, cache_dir, split):
        self.items, self.norm_const = [], None
        for fp in sorted((Path(cache_dir) / split).glob("*.npz")):
            c = np.load(fp, allow_pickle=True)
            nc = float(c["norm_const"])
            assert self.norm_const is None or abs(nc - self.norm_const) < 1e-6
            self.norm_const = nc
            case = str(c["case_id"])
            for wi in range(len(c["win_starts"])):
                self.items.append({"phi": c["phi_gen_norm"][wi], "z0": c["z0"][wi],
                                   "phase": np.asarray(c["phase"][wi]), "aphase": float(c["anchor_phase"][wi]),
                                   "amp": float(c["amp"][wi]), "case": case, "win": len(self.items)})

    def __len__(self):
        return len(self.items)

    def amp_stats(self):
        a = np.array([it["amp"] for it in self.items])
        return float(a.mean()), float(a.std())

    def collect(self, idx, dev, mu, sd):
        """Flatten to [B*K, ...]; z0 / anchor phase / amplitude repeated for the K frames."""
        g = lambda k: np.asarray([self.items[i][k] for i in idx])
        t_ = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
        b = len(idx)
        _p = g("phi")
        phi = t_(_p).reshape(b * K, 3, *_p.shape[-3:])
        z0 = t_(g("z0")).repeat_interleave(K, dim=0)
        ph = t_(g("phase")).reshape(b * K)
        aph = t_(g("aphase")).repeat_interleave(K)
        amp = ((t_(g("amp")) - mu) / sd).repeat_interleave(K)
        return phi, z0, ph, aph, amp


@torch.no_grad()
def sample_fields(unet, cond, sched, holder, z0, phase, aph, amp, dev, steps=30, seed=0, w=1.0):
    """Rectified-flow sampling of the K flow maps of a window (jointly), with classifier-free
    guidance on the amplitude condition (w = 1 disables guidance). Returns [B*K, 3, z, y, x]."""
    n = z0.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, *z0.shape[2:], generator=gen).to(dev)
    sched.set_timesteps(steps, input_img_size_numel=int(np.prod(z0.shape[2:])), device=dev)
    e_c = cond_emb(cond, phase, aph, amp, None)
    e_u = cond_emb(cond, phase, aph, amp, torch.ones(n, dtype=torch.bool, device=dev))
    for t in sched.timesteps:
        tb = t.expand(n).to(dev) if torch.is_tensor(t) else torch.full((n,), float(t), device=dev)
        holder["cond_emb"] = e_c
        v = unet_call(unet, cat_z0(x, z0), tb, dev)
        if w != 1.0:
            holder["cond_emb"] = e_u
            v_u = unet_call(unet, cat_z0(x, z0), tb, dev)
            v = v_u + w * (v - v_u)
        x, _ = sched.step(v, t, x)
    return x
