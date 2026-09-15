"""Composition consistency (cocycle) of flow maps.

For a window pair starting at s and r = s + delta of the same sequence, the flow maps must satisfy
    phi_{s->t} = phi_{r->t} + warp(phi_{s->r}, phi_{r->t})        (backward-sampling convention)
i.e. ``compose_displacements(first = phi_{r->t}, second = phi_{s->r})``. The residual of this identity
is the cocycle loss; it is evaluated in the full-resolution voxel space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .spatial import SpatialTransformer, compose_displacements

SHAPE_PX = (32, 128, 128)
POOL = 4
K = 7
DELTA_MAX = 6


def make_st(space, dev, shape_px=SHAPE_PX, pool=POOL):
    return SpatialTransformer(shape_px if space == "full" else tuple(s // pool for s in shape_px)).to(dev)


def to_geom(phi_norm, norm_const, space, shape_px=SHAPE_PX, pool=POOL):
    """Normalised generation-space field -> voxel displacement in the composition space."""
    if space == "full":
        return F.interpolate(phi_norm * norm_const, size=shape_px, mode="trilinear", align_corners=True)
    return phi_norm * (norm_const / pool)


def triplet_slots(delta):
    """Slot j holds phi_{s->s+j+1}. Returns (m, lhs slot, first slot [r->t], second slot [s->r])."""
    return [(m, m - 1, m - delta - 1, delta - 1) for m in range(delta + 1, K + 1)]


def cocycle_terms(A, B, delta, st):
    """A, B: [K,3,*] fields of windows s and s+delta. Returns (residuals, lhs, additive residuals)."""
    res, lhs_l, add_l = [], [], []
    for _m, i_l, i_f, i_s in triplet_slots(delta):
        lhs, fst, snd = A[i_l:i_l + 1], B[i_f:i_f + 1], A[i_s:i_s + 1]
        res.append(compose_displacements(fst, snd, st) - lhs)
        add_l.append(fst + snd - lhs)
        lhs_l.append(lhs)
    return res, lhs_l, add_l


def loss_cocycle(x0_A, x0_B, delta, st, norm_const, space, gt_energy):
    A, B = to_geom(x0_A, norm_const, space), to_geom(x0_B, norm_const, space)
    res, _, _ = cocycle_terms(A, B, delta, st)
    return torch.stack([r.pow(2).mean() for r in res]).mean() / gt_energy


def loss_amp(x0_hat, x0_gt, eps=1e-6):
    """Amplitude-moment matching: squared log-ratio of per-frame RMS."""
    m = lambda u: u.pow(2).mean(dim=(1, 2, 3, 4)).clamp_min(eps).sqrt()
    return (m(x0_hat).log() - m(x0_gt).log()).pow(2).mean()


def invert_field(u, st, iters=10):
    """Fixed-point inverse: v such that v + warp(u, v) = 0."""
    v = -u
    for _ in range(iters):
        v = -st(u, v)
    return v


def window_pairs(cache_dir, split, n_items):
    """Same-sequence sliding-window pairs -> [(idx_A, idx_B, delta)] indexed like ``Windows``."""
    starts, gidx, per_case = [], 0, []
    for fp in sorted((Path(cache_dir) / split).glob("*.npz")):
        c = np.load(fp, allow_pickle=True)
        ss = [int(s) for s in c["win_starts"]]
        per_case.append({s: gidx + i for i, s in enumerate(ss)})
        starts += ss
        gidx += len(ss)
    assert gidx == n_items, f"{gidx} windows in cache vs {n_items} in dataset"
    pairs = []
    for pos in per_case:
        for s, ia in sorted(pos.items()):
            for d in range(1, DELTA_MAX + 1):
                if s + d in pos:
                    pairs.append((ia, pos[s + d], d))
    return pairs, starts
