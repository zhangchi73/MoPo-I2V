"""Data access for the deterministic head G and the generative bridge F.

Reads the posterior cache written by ``build_posterior_cache.py`` (one ``.npz`` per window with keys
I0, gt, hint, h_k, phi_bar, sigma2_phi, eta_raw, p995). Intensities are in the normalised domain
x = clip(I, p995) / p995; the evaluation domain is x * p995 clipped to [0, 1].
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

K = 7   # generated frames per window (t = 1..K)
KEYS_G = ("I0", "gt", "hint", "phi_bar", "sigma2_phi", "eta_raw", "p995")
KEYS_F = ("I0", "gt", "hint", "phi_bar", "sigma2_phi", "p995", "h_k")


class Store:
    def __init__(self, cache_dir, split, keys=KEYS_G, mu_path=None, limit=0):
        files = sorted((Path(cache_dir) / split).glob("*.npz"))
        if limit:
            files = files[:limit]
        mu = np.load(mu_path) if mu_path else None
        self.items, t0 = [], time.time()
        for fp in files:
            d = np.load(fp)
            it = {k: d[k] for k in keys} | {"name": fp.stem}
            if mu is not None:
                it["mu"] = mu[fp.stem]
            self.items.append(it)
        print(f"[{split}] {len(self.items)} windows ({time.time()-t0:.0f}s)", flush=True)

    def __len__(self):
        return len(self.items)


def sigma2_closed(it, t, c, cz, floor):
    """Closed-form variance: sigma^2 = c * Var_k h^(k) + c_z * eta + sigma_0^2 (no learned parameters)."""
    return c * it["sigma2_phi"][t].astype(np.float32) + cz * it["eta_raw"][t].astype(np.float32) + floor


def cond_channels(I0, hint, phi, s2, t):
    """7-channel conditioning: I0, hint_t, phi_bar_t (3), log sigma, t/K."""
    return np.concatenate([I0[None], hint[None], phi, 0.5 * np.log(s2)[None], np.full_like(hint, t / K)[None]], 0)


# ---------------------------------------------------------------- G
def make_sample(it, t, a, aug_rng=None):
    I0 = it["I0"].astype(np.float32)
    gt = it["gt"][t].astype(np.float32)
    hint = it["hint"][t].astype(np.float32)
    phi = it["phi_bar"][t - 1].astype(np.float32)
    s2 = sigma2_closed(it, t, a.sig_c, a.sig_cz, a.sig_floor)
    if aug_rng is not None and a.aug:
        for ax, comp in ((1, 1), (2, 2)):
            if aug_rng.random() < 0.5:
                I0, gt, hint, s2 = (np.flip(v, ax).copy() for v in (I0, gt, hint, s2))
                phi = np.flip(phi, ax + 1).copy()
                phi[comp] = -phi[comp]
    return {"x": cond_channels(I0, hint, phi, s2, t), "r": (gt - hint)[None], "sig2": s2[None],
            "hint": hint[None], "gt": gt[None], "p995": float(it["p995"])}


def collate_g(samples, dev):
    out = {k: torch.from_numpy(np.stack([s[k] for s in samples])).to(dev) for k in ("x", "r", "sig2", "hint", "gt")}
    out["p995"] = torch.tensor([s["p995"] for s in samples], device=dev)
    return out


# ---------------------------------------------------------------- F
def frame_arrays(it, t, a, k, flip=None):
    """x0 = hint + mu, gt, sigma^2, cond (7ch), xi (posterior-sample noise), phi."""
    I0 = it["I0"].astype(np.float32)
    gt = it["gt"][t].astype(np.float32)
    hint = it["hint"][t].astype(np.float32)
    mu = it["mu"][t - 1].astype(np.float32)
    phi = it["phi_bar"][t - 1].astype(np.float32)
    s2 = a.sig_c * it["sigma2_phi"][t].astype(np.float32) + a.sig_floor
    xi = math.sqrt(a.sig_c) * (it["h_k"][k, t].astype(np.float32) - hint) if a.noise == "xi" else None
    if flip is not None:
        for ax, comp, do in ((1, 1, flip[0]), (2, 2, flip[1])):
            if do:
                I0, gt, hint, mu, s2 = (np.flip(v, ax).copy() for v in (I0, gt, hint, mu, s2))
                if xi is not None:
                    xi = np.flip(xi, ax).copy()
                phi = np.flip(phi, ax + 1).copy()
                phi[comp] = -phi[comp]
    return {"x0": (hint + mu)[None], "gt": gt[None], "sig2": s2[None], "cond": cond_channels(I0, hint, phi, s2, t),
            "hint": hint[None], "xi": None if xi is None else xi[None], "phi": phi, "p995": float(it["p995"])}


def make_pair(it, a, rng):
    t, r = rng.choice(np.arange(1, K + 1), 2, replace=False)
    k = int(rng.integers(0, it["h_k"].shape[0]))
    flip = (rng.random() < 0.5, rng.random() < 0.5) if a.aug else None
    return [frame_arrays(it, int(t), a, k, flip), frame_arrays(it, int(r), a, k, flip)]


def collate_f(frames, dev):
    out = {k: torch.from_numpy(np.stack([f[k] for f in frames])).to(dev) for k in ("x0", "gt", "sig2", "cond", "hint", "phi")}
    out["p995"] = torch.tensor([f["p995"] for f in frames], device=dev)
    if frames[0]["xi"] is not None:
        out["xi"] = torch.from_numpy(np.stack([f["xi"] for f in frames])).to(dev)
    return out


def draw_noise(b, a, gen=None):
    """xi mode: sqrt(c) (h^(k) - hint) + sigma_0 z  (anisotropic, on the deformation orbit);
    iso mode: sigma * z."""
    z = torch.randn(b["gt"].shape, device=b["gt"].device, generator=gen)
    if a.noise == "xi":
        return b["xi"] + math.sqrt(a.sig_floor) * z
    return b["sig2"].sqrt() * z
