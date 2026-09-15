"""Sample the motion posterior: N flow-map samples phi^(k)_{0->t}, t=1..K, per case from the
anchor-frame latent z0, with classifier-free guidance on the amplitude condition. The samples are
delivered in full-resolution voxel units and written as ``<out>/<split>/<case>.npz`` with
    phi        [N, n_win, K, 3, Z, Y, X]  fp16
    win_starts, p995 (intensity normaliser of the case, from the field cache)

    python sample_motion.py --ckpt runs/motion/last.pt --cache data/field_cache --split test \
        --base-ckpt ... --base-cfg ... --out data/posterior --nseed 5 --w 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from motion.arch import build_motion_unet
from motion.data import K, Windows, sample_fields, to_delivery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nseed", type=int, default=5)
    ap.add_argument("--w", type=float, default=4.0, help="classifier-free guidance weight on amplitude")
    ap.add_argument("--gain", type=float, default=1.0, help="delivery-space amplitude calibration gain")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    dev = torch.device("cuda")
    from monai.networks.schedulers import RFlowScheduler
    sched = RFlowScheduler(num_train_timesteps=1000, use_discrete_timesteps=False,
                           use_timestep_transform=True, sample_method="uniform")
    holder = {"T": K, "cond_emb": None}
    ck = torch.load(a.ckpt, map_location="cpu")
    unet, cond, _ = build_motion_unet(holder, dev, a.base_ckpt, a.base_cfg, lora_rank=ck["cfg"]["lora_rank"])
    pmap = {f"unet.{n}": p for n, p in unet.named_parameters()}
    pmap.update({f"cond.{n}": p for n, p in cond.named_parameters()})
    with torch.no_grad():
        for k, v in ck["ema"].items():
            pmap[k].copy_(v.to(dev))
    unet.eval()
    mu, sd = ck["amp_norm"]["mu"], ck["amp_norm"]["sd"]
    norm_const = ck["norm_const"]

    ds = Windows(a.cache, a.split)
    out = Path(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)
    by_case = {}
    for i, it in enumerate(ds.items):
        by_case.setdefault(it["case"], []).append(i)
    for case, idx in by_case.items():
        _, z0, ph, aph, ampn = ds.collect(idx, dev, mu, sd)
        phis = []
        for sdd in range(a.nseed):
            g = sample_fields(unet, cond, sched, holder, z0, ph, aph, ampn, dev, a.steps, seed=1000 * sdd + hash(case) % 1000, w=a.w)
            d = to_delivery(g, norm_const, a.gain)                        # [n_win*K, 3, Z, Y, X]
            phis.append(d.reshape(len(idx), K, 3, *d.shape[2:]).cpu().numpy().astype(np.float16))
        src = np.load(Path(a.cache) / a.split / f"{case}.npz", allow_pickle=True)
        np.savez_compressed(out / f"{case}.npz", phi=np.stack(phis), win_starts=src["win_starts"],
                            p995=np.float32(src["p995"]) if "p995" in src else np.float32(1.0))
        print(case, flush=True)


if __name__ == "__main__":
    main()
