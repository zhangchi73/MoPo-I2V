"""Motion-Posterior Metamorphosis, deterministic head G.

    mu_t = G(I0, hint_t, phi_bar_t, log sigma_t, t/K)
    L = mean( w * (r_t - mu_t)^2 ) + lam_p * LPIPS_2.5D(hint_t + mu_t, I_t),   w = 1 / sigma^2 (per-sample normalised)
    sigma^2 = c * Var_k h^(k)_t + sigma_0^2  (closed form, from the motion posterior; no learned variance)

    python train_g.py --cache data/mpm_cache --out runs/g
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from metamorphosis import UNet3D, lpips_25d_train
from metamorphosis.data import K, Store, collate_g, make_sample


def g_loss(net, b, a):
    mu = net(b["x"])
    w = 1.0 / b["sig2"] ** a.w_pow
    w = w / w.mean(dim=(1, 2, 3, 4), keepdim=True)
    main = (w * (b["r"] - mu) ** 2).mean()
    terms = {"main": main}
    L = main
    if a.lam_p > 0:
        p995 = b["p995"].view(-1, 1, 1, 1, 1)
        lp = lpips_25d_train((b["hint"] + mu) * p995, b["gt"] * p995)
        L = L + a.lam_p * lp
        terms["lpips"] = lp
    return L, terms


@torch.no_grad()
def validate(net, store, a, dev):
    """Case-equal PSNR (evaluation domain) of hint + mu vs hint over frames 1..K."""
    net.eval()
    ps_hint, ps_g = [], []
    for it in store.items:
        fh, fg = [], []
        for t in range(1, K + 1):
            b = collate_g([make_sample(it, t, a)], dev)
            mu = net(b["x"])
            p995 = b["p995"].view(-1, 1, 1, 1, 1)
            gt = (b["gt"] * p995).clamp(0, 1)
            for pred, lst in ((b["hint"] * p995, fh), ((b["hint"] + mu) * p995, fg)):
                mse = ((pred.clamp(0, 1) - gt) ** 2).mean().item()
                lst.append(10 * math.log10(1.0 / max(mse, 1e-12)))
        ps_hint.append(np.mean(fh))
        ps_g.append(np.mean(fg))
    net.train()
    return float(np.mean(ps_g)), float(np.mean(ps_hint))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sig_c", type=float, default=6.29)
    ap.add_argument("--sig_cz", type=float, default=0.0)
    ap.add_argument("--sig_floor", type=float, default=3.59e-3)
    ap.add_argument("--w_pow", type=float, default=1.0)
    ap.add_argument("--lam_p", type=float, default=0.1)
    ap.add_argument("--base", type=int, default=40)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=2))
    tr, va = Store(a.cache, "train"), Store(a.cache, "val")
    net = UNet3D(7, 1, a.base, a.levels).to(dev)
    print(f"G params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    step, best = 0, -1e9
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", -1e9)
    print(f"val hint PSNR = {validate(net, va, a, dev)[1]:.4f}", flush=True)
    aug_rng = np.random.default_rng(a.seed + 1)
    t0, acc = time.time(), {}
    while step < a.steps:
        idx = np.random.randint(0, len(tr.items), a.bs)
        ts = np.random.randint(1, K + 1, a.bs)
        b = collate_g([make_sample(tr.items[i], int(t), a, aug_rng) for i, t in zip(idx, ts)], dev)
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        L, terms = g_loss(net, b, a)
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        acc["n"] = acc.get("n", 0) + 1
        if step % a.log_every == 0:
            n = acc.pop("n")
            print(f"step {step} " + " ".join(f"{k} {v/n:.4g}" for k, v in acc.items()) + f" {(time.time()-t0)/60:.1f}min", flush=True)
            acc = {}
        if step % a.val_every == 0 or step == a.steps:
            p_g, p_h = validate(net, va, a, dev)
            print(f"[val] step {step} PSNR G {p_g:.4f} (hint {p_h:.4f}) best {max(best, p_g):.4f}", flush=True)
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": max(best, p_g),
                  "args": vars(a), "val_psnr": p_g}
            torch.save(ck, run / "last.pt")
            if p_g > best:
                best = p_g
                torch.save(ck, run / "best.pt")


if __name__ == "__main__":
    main()
