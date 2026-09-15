"""Export the residual means mu_t (t = 1..K) predicted by a trained G for every window of a split
-> ``<run>/mu_<split>.npz`` (key = window name), consumed by train_f.py / sample_f.py.

    python export_mu.py --run runs/g --cache data/mpm_cache --split train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from metamorphosis import UNet3D
from metamorphosis.data import K, Store, collate_g, make_sample

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--cache", required=True)
ap.add_argument("--split", default="val")
a0 = ap.parse_args()
run = Path(a0.run)
ck = torch.load(run / a0.ckpt, map_location="cpu")
a = argparse.Namespace(**ck["args"])
a.aug = 0
dev = "cuda"
net = UNet3D(7, 1, a.base, a.levels).to(dev)
net.load_state_dict(ck["model"])
net.eval()
out = {}
with torch.no_grad():
    for it in Store(a0.cache, a0.split).items:
        out[it["name"]] = np.stack([net(collate_g([make_sample(it, t, a)], dev)["x"])[0, 0].cpu().numpy()
                                    for t in range(1, K + 1)]).astype(np.float16)
np.savez_compressed(run / f"mu_{a0.split}.npz", **out)
print(f"{len(out)} windows -> {run / f'mu_{a0.split}.npz'}")
