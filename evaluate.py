"""Evaluate generated sequences (evaluator-format npz) against the HDF5 ground truth.

Given frame 0, frames 1..T-1 are scored (case-equal means): PSNR, SSIM, 2.5D LPIPS, and the motion
magnitude ratio |v| = RMS(x_t - x_0) of the prediction over that of the ground truth.

    python evaluate.py --npz-dir cells/mpm_r0 --h5dir data/acdc_h5 --out results/mpm_r0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim


def psnr_ssim(recon, target):
    r = np.clip(recon, 0.0, 1.0).astype(np.float64)
    t = target.astype(np.float64)
    mse = float(((r - t) ** 2).mean())
    return {"psnr": 10.0 * np.log10(1.0 / max(mse, 1e-12)),
            "ssim": float(np.mean([_ssim(t[k], r[k], data_range=1.0) for k in range(len(t))]))}


@torch.no_grad()
def lpips_25d(recon, target, network):
    dev = next(network.parameters()).device
    r = torch.from_numpy(np.clip(recon, 0.0, 1.0)).float().to(dev)
    t = torch.from_numpy(np.clip(target, 0.0, 1.0)).float().to(dev)
    planes = []
    for axis in (1, 2, 3):
        n = r.shape[axis]
        stride = max(1, n // 8)
        vals = []
        for idx in range(0, n, stride):
            sel = torch.tensor([idx], device=dev)
            rs = r.index_select(axis, sel).squeeze(axis).unsqueeze(1).repeat(1, 3, 1, 1) * 2 - 1
            ts = t.index_select(axis, sel).squeeze(axis).unsqueeze(1).repeat(1, 3, 1, 1) * 2 - 1
            vals.append(float(network(rs, ts).mean()))
        planes.append(float(np.mean(vals)))
    return float(np.mean(planes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--h5dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-lpips", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lp = None
    if not a.no_lpips:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    per_case = []
    for fp in sorted(Path(a.npz_dir).glob("*.npz")):
        cid = fp.stem.split("_w")[0]
        z = np.load(fp)
        T = int(z["frame_count"])
        pr = z["prediction"][:T].astype(np.float32)
        with h5py.File(Path(a.h5dir) / f"{cid}.h5", "r") as hf:
            s = int(z["win_start"]) if "win_start" in z else 0
            gt = hf["image"][s:s + T].astype(np.float32)
        m = {"case": fp.stem, **psnr_ssim(pr[1:], gt[1:])}
        m["motion_ratio"] = float(np.sqrt(np.mean((pr[1:] - pr[:1]) ** 2)) / max(np.sqrt(np.mean((gt[1:] - gt[:1]) ** 2)), 1e-8))
        if lp is not None:
            m["lpips"] = lpips_25d(pr[1:], gt[1:], lp)
        per_case.append(m)
        print(fp.stem, {k: round(v, 4) for k, v in m.items() if k != "case"}, flush=True)
    keys = [k for k in per_case[0] if k != "case"]
    summary = {k: float(np.mean([c[k] for c in per_case])) for k in keys}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"per_case": per_case, "summary": summary}, indent=2))
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
