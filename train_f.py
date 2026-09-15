"""Motion-Posterior Metamorphosis, generative bridge F.

Bridge from x0 = hint_t + mu_t (transport + deterministic mean) to x1 = I_t:
    x_tau = (1 - tau) x0 + tau x1 + sqrt(tau (1 - tau)) xi,   tau ~ U(0, 1)
    xi    = sqrt(c) (h^(k)_t - hint_t) + sigma_0 z            noise on the deformation orbit (posterior sample k)
    x1_hat = x0 + F(x_tau, tau, cond, log sigma)

    L = mean(w (x1_hat - I_t)^2) + lam_p LPIPS_2.5D + lam_c L_cons + lam_a L_adv
    L_cons  transports the predicted residual from frame t to frame r through the flow maps and penalises the
            disagreement where sigma is small; L_adv is an RpGAN + R1 PatchGAN on the three orthogonal planes.

    python train_f.py --cache data/mpm_cache --mu runs/g --out runs/f
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from metamorphosis import FUNet3D, PatchD2D, add_hp, lpips_25d_train
from metamorphosis.data import K, KEYS_F, Store, collate_f, draw_noise, frame_arrays, make_pair
from metamorphosis.transport import transport


@torch.no_grad()
def sample_bridge(net, b, a, s=1.0, gen=None, steps=4):
    """Euler steps tau in {0, 1/steps, ...}; the noise realisation xi is fixed along the path; s scales it."""
    x0 = b["x0"]
    xi = draw_noise(b, a, gen) * s
    smap = 0.5 * torch.log(b["sig2"])
    x1 = x0.clone()
    for i in range(steps):
        tau = i / steps
        xt = (1 - tau) * x0 + tau * x1 + math.sqrt(tau * (1 - tau)) * xi
        tv = torch.full((x0.shape[0],), tau, device=x0.device)
        x1 = x0 + net(torch.cat([xt, b["cond"]], 1), tv, smap)
    return x1


@torch.no_grad()
def validate(net, store, a, dev, n_var=3):
    """PSNR (evaluation domain, case-equal) of one sample at s = 1, and the sample-spread ratio
    v = std over n_var samples / sigma on the top-20% sigma voxels (guards against collapse)."""
    net.eval()
    ps_x0, ps_f, vs = [], [], []
    for it in store.items:
        fr0, frf, vlist = [], [], []
        for t in range(1, K + 1):
            gens = [torch.Generator(device=dev).manual_seed(7919 + 131 * j + t) for j in range(n_var)]
            nk = it["h_k"].shape[0]
            frames = [frame_arrays(it, t, a, j % nk) for j in range(n_var)]
            b = collate_f(frames, dev)
            xs = torch.stack([sample_bridge(net, {kk: v[j:j + 1] for kk, v in b.items()}, a, 1.0, gens[j])[0, 0]
                              for j in range(n_var)])
            p995 = float(it["p995"])
            gt = (b["gt"][0, 0] * p995).clamp(0, 1)
            for pred, lst in ((b["x0"][0, 0] * p995, fr0), (xs[0] * p995, frf)):
                mse = ((pred.clamp(0, 1) - gt) ** 2).mean().item()
                lst.append(10 * math.log10(1.0 / max(mse, 1e-12)))
            sig = b["sig2"][0, 0].sqrt()
            m = sig >= torch.quantile(sig.flatten(), 0.8)
            vlist.append((xs.std(0, unbiased=True)[m] / sig[m]).mean().item())
        ps_x0.append(np.mean(fr0))
        ps_f.append(np.mean(frf))
        vs.append(np.mean(vlist))
    net.train()
    return float(np.mean(ps_f)), float(np.mean(ps_x0)), float(np.mean(vs))


def slices3(vol, phi, n, dev, idxs=None):
    outs, used = [], []
    x = torch.cat([vol, phi], 1)
    for i, axis in enumerate((2, 3, 4)):
        idx = idxs[i] if idxs is not None else torch.randint(0, x.shape[axis], (n,), device=dev)
        used.append(idx)
        outs.append(x.index_select(axis, idx).movedim(axis, 1).flatten(0, 1))
    return outs, used


def f_step(net, D, b, a, step, optD):
    B = b["x0"].shape[0]
    tau = torch.rand(B // 2, device=b["x0"].device).repeat_interleave(2)     # shared by the frame pair
    xi = draw_noise(b, a)
    tv = tau[:, None, None, None, None]
    xt = (1 - tv) * b["x0"] + tv * b["gt"] + torch.sqrt(tv * (1 - tv)) * xi
    smap = 0.5 * torch.log(b["sig2"])
    x1 = b["x0"] + net(torch.cat([xt, b["cond"]], 1), tau, smap)
    w = 1.0 / b["sig2"] ** a.w_pow
    w = w / w.mean(dim=(1, 2, 3, 4), keepdim=True)
    main = (w * (x1 - b["gt"]) ** 2).mean()
    terms, L = {"main": main}, main
    p995 = b["p995"].view(-1, 1, 1, 1, 1)
    if a.lam_p > 0:
        lp = lpips_25d_train(x1 * p995, b["gt"] * p995)
        L = L + a.lam_p * lp
        terms["lpips"] = lp
    if a.lam_c > 0:
        st, sr = x1[0::2] - b["hint"][0::2], x1[1::2] - b["hint"][1::2]
        moved = transport(st, b["phi"][0::2], b["phi"][1::2])
        sh = b["sig2"][1::2].sqrt()
        sh = sh / sh.amax(dim=(1, 2, 3, 4), keepdim=True)
        lc = ((1 - sh) * (moved - sr) ** 2).mean()
        L = L + a.lam_c * lc
        terms["cons"] = lc
    if a.lam_a > 0 and step >= a.adv_start:
        fake, idxs = slices3(x1 * p995, b["phi"], a.n_slices, x1.device)
        real, _ = slices3(b["gt"] * p995, b["phi"], a.n_slices, x1.device, idxs)
        fake = [add_hp(f) for f in fake]
        real = [add_hp(r) for r in real]
        ld = 0.0
        for fk, rl in zip(fake, real):
            rl = rl.detach().requires_grad_(True)
            dr, df = D(rl), D(fk.detach())
            gr = torch.autograd.grad(dr.sum(), rl, create_graph=True)[0]
            ld = ld + F.softplus(-(dr - df)).mean() + a.r1_gamma / 2 * gr.pow(2).flatten(1).sum(1).mean()
        optD.zero_grad(set_to_none=True)
        (ld / 3).backward()
        optD.step()
        terms["d"] = ld / 3
        la = sum(F.softplus(-(D(fk) - D(rl.detach()))).mean() for fk, rl in zip(fake, real)) / 3
        L = L + a.lam_a * la
        terms["adv"] = la
    return L, terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mu", required=True, help="G run directory containing mu_train.npz / mu_val.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", choices=["xi", "iso"], default="xi")
    ap.add_argument("--sig_c", type=float, default=6.29)
    ap.add_argument("--sig_floor", type=float, default=3.59e-3)
    ap.add_argument("--w_pow", type=float, default=1.0)
    ap.add_argument("--lam_p", type=float, default=0.1)
    ap.add_argument("--lam_c", type=float, default=0.1)
    ap.add_argument("--lam_a", type=float, default=0.01)
    ap.add_argument("--adv_start", type=int, default=5000)
    ap.add_argument("--r1_gamma", type=float, default=0.01)
    ap.add_argument("--n_slices", type=int, default=2)
    ap.add_argument("--d_lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=2, help="batch = pairs x 2 frames")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v_min", type=float, default=0.5)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=2))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed + 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    tr = Store(a.cache, "train", KEYS_F, Path(a.mu) / "mu_train.npz")
    va = Store(a.cache, "val", KEYS_F, Path(a.mu) / "mu_val.npz")
    net, D = FUNet3D(8, a.base, a.levels).to(dev), PatchD2D().to(dev)
    print(f"F params {sum(p.numel() for p in net.parameters())/1e6:.2f}M, D {sum(p.numel() for p in D.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    optD = torch.optim.Adam(D.parameters(), lr=a.d_lr, betas=(0.0, 0.99))
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    step, best = 0, -1e9
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        D.load_state_dict(ck["D"])
        optD.load_state_dict(ck["optD"])
        step, best = ck["step"], ck.get("best", -1e9)

    def save(name, extra):
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "D": D.state_dict(), "optD": optD.state_dict(),
                    "step": step, "best": best, "args": vars(a), **extra}, run / name)

    t0, acc = time.time(), {}
    while step < a.steps:
        frames = []
        for i in np.random.randint(0, len(tr.items), a.pairs):
            frames += make_pair(tr.items[i], a, rng)
        b = collate_f(frames, dev)
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        L, terms = f_step(net, D, b, a, step, optD)
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if not math.isfinite(float(L)):
            print(f"[NaN] step {step}", flush=True)
            save("nan.pt", {})
            sys.exit(2)
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        acc["n"] = acc.get("n", 0) + 1
        if step % a.log_every == 0:
            n = acc.pop("n")
            print(f"step {step} " + " ".join(f"{k} {v/n:.4g}" for k, v in acc.items()) + f" {(time.time()-t0)/60:.1f}min", flush=True)
            acc = {}
        if step % a.val_every == 0 or step == a.steps:
            pf, p0, v = validate(net, va, a, dev)
            ok = v >= a.v_min
            if pf > best and ok:
                best = pf
                save("best.pt", {"val_psnr": pf, "val_v": v})
            save("last.pt", {"val_psnr": pf, "val_v": v})
            print(f"[val] step {step} PSNR F {pf:.4f} (x0 {p0:.4f}) v {v:.3f}{'' if ok else ' (< v_min, not best)'} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
