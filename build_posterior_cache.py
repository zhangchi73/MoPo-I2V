"""Motion-Posterior Metamorphosis, step 0: from N sampled flow maps per case build the transport
quantities consumed by the deterministic head G and the generative bridge F.

    h^(k)_t    = warp_bspline(I_0, phi^(k)_{0->t})           per-sample transports
    hint_t     = warp_bspline(I_0, phi_bar_{0->t})           transport along the posterior mean
    sigma2_phi = Var_k h^(k)_t                               geometric uncertainty pushed to intensity
    eta_raw    = slice-phase observability * |d_z hint|^2    (optional through-plane term)

Output ``<out>/<split>/<case>[_w<start>].npz`` in the normalised intensity domain x = clip(I, p995) / p995.

    python build_posterior_cache.py --posterior data/posterior --h5dir data/acdc_h5 --split test --out data/mpm_cache
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

from transport import warp_bspline3d, zero_pad_slot0

K = 8   # frames per window including the anchor


def gradz(x):
    g = np.empty_like(x)
    g[..., 1:-1, :, :] = 0.5 * (x[..., 2:, :, :] - x[..., :-2, :, :])
    g[..., 0, :, :] = x[..., 1, :, :] - x[..., 0, :, :]
    g[..., -1, :, :] = x[..., -1, :, :] - x[..., -2, :, :]
    return g


def build_one(args):
    fp, h5dir, split, out_dir, dz_vox, single = args
    c = np.load(fp, allow_pickle=True)
    case = fp.stem
    starts = [int(x) for x in c["win_starts"]]
    p995 = float(c["p995"])
    with h5py.File(Path(h5dir) / f"{case}.h5", "r") as f:
        img = f["image"][:].astype(np.float32)
    x_all = np.clip(img, 0, p995) / max(p995, 1e-6)
    done = []
    for wi, s in enumerate(starts):
        tag = case if single else f"{case}_w{s}"
        dst = out_dir / f"{tag}.npz"
        if dst.exists():
            done.append(tag)
            continue
        x0 = torch.from_numpy(x_all[s])
        gt = x_all[s:s + K]
        phi = c["phi"][:, wi].astype(np.float32)                       # [N, K-1, 3, Z, Y, X]
        rep = x0[None, None].expand(K, -1, -1, -1, -1).contiguous()
        hk = []
        for k in range(phi.shape[0]):
            hk.append(warp_bspline3d(rep, zero_pad_slot0(torch.from_numpy(phi[k])[None])[0])[:, 0].numpy())
        hk = np.stack(hk)
        phi_bar = phi.mean(0)
        phi8b = zero_pad_slot0(torch.from_numpy(phi_bar)[None])[0]
        hint = warp_bspline3d(rep, phi8b)[:, 0].numpy()
        sig2_phi = hk.var(0, ddof=1)
        frac = np.mod(phi8b[:, 0].numpy() / dz_vox, 1.0)
        eta_raw = (1.0 - np.abs(2.0 * frac - 1.0)) * gradz(hint) ** 2
        np.savez_compressed(dst, I0=x0.numpy().astype(np.float16), gt=gt.astype(np.float16),
                            hint=hint.astype(np.float16), h_k=hk.astype(np.float16),
                            phi_bar=phi_bar.astype(np.float16), sigma2_phi=sig2_phi.astype(np.float32),
                            eta_raw=eta_raw.astype(np.float32), p995=np.float32(p995),
                            win_start=np.int64(s), dz_vox=np.float32(dz_vox))
        done.append(tag)
    return case, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", required=True, help="output of sample_motion.py")
    ap.add_argument("--h5dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dz-vox", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--single-window", action="store_true",
                    help="one window per case (evaluation splits): file name = case id")
    a = ap.parse_args()
    files = sorted((Path(a.posterior) / a.split).glob("*.npz"))
    assert files
    out_dir = Path(a.out) / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    t0 = time.time()
    jobs = [(f, a.h5dir, a.split, out_dir, a.dz_vox, a.single_window) for f in files]
    with Pool(a.workers) as pool:
        for i, (case, done) in enumerate(pool.imap_unordered(build_one, jobs)):
            print(f"[{i+1}/{len(files)}] {case}: {len(done)} windows ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
