"""Image-to-video inference: for every window of a split, generate frames 1..K with the bridge F
(Euler steps, noise scale s) and write evaluator-format ``<out>/<name>_r<seed>/<case>.npz`` with
``prediction`` = clip(x * p995, 0, 1) [K+1, Z, Y, X] (frame 0 = the given volume) and ``frame_count``.

    python sample_f.py --run runs/f --cache data/mpm_cache --mu runs/g --split test --name mpm --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metamorphosis import FUNet3D
from metamorphosis.data import K, KEYS_F, Store, collate_f, frame_arrays
from train_f import sample_bridge

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--cache", required=True)
ap.add_argument("--mu", required=True, help="G run directory containing mu_<split>.npz")
ap.add_argument("--split", default="test")
ap.add_argument("--name", required=True)
ap.add_argument("--s", type=float, default=1.0, help="noise scale (posterior temperature)")
ap.add_argument("--seeds", default="0")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--out", default="cells")
a0 = ap.parse_args()
run = Path(a0.run)
ck = torch.load(run / a0.ckpt, map_location="cpu")
a = argparse.Namespace(**ck["args"])
a.aug = 0
dev = "cuda"
net = FUNet3D(8, a.base, a.levels).to(dev)
net.load_state_dict(ck["model"])
net.eval()
st = Store(a0.cache, a0.split, KEYS_F, Path(a0.mu) / f"mu_{a0.split}.npz")
seeds = [int(x) for x in a0.seeds.split(",")]
for sd in seeds:
    out = Path(a0.out) / f"{a0.name}_r{sd}"
    out.mkdir(parents=True, exist_ok=True)
    for it in st.items:
        p995 = float(it["p995"])
        nk = it["h_k"].shape[0]
        xs = [it["hint"][0].astype(np.float32)]
        for t in range(1, K + 1):
            k = (sd + t) % nk                        # one posterior sample per (seed, frame)
            b = collate_f([frame_arrays(it, t, a, k)], dev)
            g = torch.Generator(device=dev).manual_seed(100000 * sd + 1000 * t + hash(it["name"]) % 1000)
            xs.append(sample_bridge(net, b, a, a0.s, g, steps=a0.steps)[0, 0].cpu().numpy())
        x = np.stack(xs)
        np.savez_compressed(out / f"{it['name']}.npz", prediction=np.clip(x * p995, 0, 1).astype(np.float16),
                            frame_count=np.int64(K + 1), p995=np.float32(p995))
    (out / "manifest.json").write_text(json.dumps({"run": str(run), "ckpt": a0.ckpt, "step": ck["step"], "s": a0.s,
                                                   "euler_steps": a0.steps, "seed": sd}, indent=2) + "\n")
    print(f"[{a0.name}] seed {sd} -> {out}", flush=True)
