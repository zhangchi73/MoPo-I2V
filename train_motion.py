"""Motion Generation: train the non-autonomous flow-map diffusion with the composition-consistency
(cocycle) constraint.

    L = L_rflow + lam_coc * L_coc + lam_amp * L_amp

* L_rflow  rectified-flow velocity regression on the K flow maps of a window (joint over frames);
* L_coc    cocycle residual between the x0-estimates of a window pair (s, s+delta) of the same
           sequence, evaluated in full-resolution voxel space; skipped at high noise (tau > tau_max);
* L_amp    amplitude-moment matching (optional).

Example:
    python train_motion.py --cache data/field_cache --out runs/motion \
        --base-ckpt NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt \
        --base-cfg  NV-Generate-MR/configs/config_network_rflow.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motion.arch import build_motion_unet
from motion.cocycle import cocycle_terms, loss_amp, make_st, to_geom, window_pairs
from motion.data import K, Windows, cat_z0, cond_emb, neg_jac, sample_fields, to_delivery, unet_call

PAIR_FRAMES = 2 * K


class PairSampler:
    def __init__(self, pairs, rng):
        self.pairs, self.rng, self.q = pairs, rng, []

    def take(self, n):
        out = []
        while len(out) < n:
            if not self.q:
                self.q = self.rng.permutation(len(self.pairs)).tolist()
            out.append(self.pairs[self.q.pop()])
        return out


@torch.no_grad()
def validate(unet, cond, sched, holder, va, dev, mu, sd, pairs, st, space, norm_const, steps=30, wpb=4, seed=4242):
    """Composition-consistency ratio ||residual|| / ||phi|| and negative-Jacobian fraction on sampled fields."""
    unet.eval()
    nf, nj = [], []
    for i0 in range(0, len(va), wpb):
        idx = list(range(i0, min(i0 + wpb, len(va))))
        _, z0, ph, aph, ampn = va.collect(idx, dev, mu, sd)
        g = sample_fields(unet, cond, sched, holder, z0, ph, aph, ampn, dev, steps, seed=seed + i0)
        nj.append(neg_jac(to_delivery(g, norm_const)))
        for kk in range(len(idx)):
            nf.append(g[kk * K:(kk + 1) * K].cpu())
    unet.train()
    num, den = [], []
    for ia, ib, d in pairs:
        A, B = to_geom(nf[ia].to(dev), norm_const, space), to_geom(nf[ib].to(dev), norm_const, space)
        res, lhs, _ = cocycle_terms(A, B, d, st)
        num += [float(r.norm(dim=1).mean()) for r in res]
        den += [float(l.norm(dim=1).mean()) for l in lhs]
    return {"comp_ratio": float(np.mean(num)) / float(np.mean(den)), "njac": float(np.mean(nj))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="field cache directory with train/ and val/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-cfg", required=True)
    ap.add_argument("--init", default=None, help="checkpoint to start from (e.g. a run without L_coc)")
    ap.add_argument("--lam-coc", type=float, default=1.7)
    ap.add_argument("--lam-amp", type=float, default=0.0)
    ap.add_argument("--cocycle-space", choices=("full", "gen"), default="full")
    ap.add_argument("--tau-max", type=float, default=0.7)
    ap.add_argument("--warmup-ep", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pairs-per-ep", type=int, default=196)
    ap.add_argument("--batch", type=int, default=1, help="window pairs per step (2K frames each)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp-dropout", type=float, default=0.10)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda")
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)

    from monai.networks.schedulers import RFlowScheduler
    sched = RFlowScheduler(num_train_timesteps=1000, use_discrete_timesteps=False,
                           use_timestep_transform=True, sample_method="uniform")
    holder = {"T": K, "cond_emb": None}
    unet, cond, trainable = build_motion_unet(holder, dev, a.base_ckpt, a.base_cfg, lora_rank=a.lora_rank)
    pmap = {f"unet.{n}": p for n, p in unet.named_parameters()}
    pmap.update({f"cond.{n}": p for n, p in cond.named_parameters()})

    tr, va = Windows(a.cache, "train"), Windows(a.cache, "val")
    norm_const = tr.norm_const
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        with torch.no_grad():
            for k in ck["ema"]:
                if k in pmap:
                    pmap[k].copy_(ck["ema"][k].to(dev))
        mu, sd = ck["amp_norm"]["mu"], ck["amp_norm"]["sd"]
    else:
        mu, sd = tr.amp_stats()
    pairs_tr, _ = window_pairs(a.cache, "train", len(tr))
    pairs_va, _ = window_pairs(a.cache, "val", len(va))
    st = make_st(a.cocycle_space, dev)
    smp = PairSampler(pairs_tr, rng)
    n_step = a.pairs_per_ep // a.batch
    print(f"train {len(tr)} windows / {len(pairs_tr)} pairs | val {len(va)} windows / {len(pairs_va)} pairs | "
          f"{n_step} steps/epoch | trainable {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-5)
    ema_params = {k: p for k, p in pmap.items() if p.requires_grad}
    ema = {k: p.detach().clone() for k, p in ema_params.items()}
    drop_gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    hist = []

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        lam_c = a.lam_coc * min(1.0, ep / max(a.warmup_ep, 1))
        agg = {"rflow": 0.0, "coc": 0.0, "amp": 0.0, "n": 0}
        for _ in range(n_step):
            pr = smp.take(a.batch)
            idx = [i for ia, ib, _ in pr for i in (ia, ib)]
            deltas = [d for _, _, d in pr]
            phi, z0, ph, aph, ampn = tr.collect(idx, dev, mu, sd)
            nb = len(pr)
            t_pair = sched.sample_timesteps(phi[::PAIR_FRAMES])       # one timestep per window pair
            t = t_pair.repeat_interleave(PAIR_FRAMES)
            eps = torch.randn_like(phi)
            xt = sched.add_noise(phi, eps, t)
            dm = (torch.rand(nb, device=dev, generator=drop_gen) < a.amp_dropout).repeat_interleave(PAIR_FRAMES)
            holder["cond_emb"] = cond_emb(cond, ph, aph, ampn, dm)
            v = unet_call(unet, cat_z0(xt, z0), t, dev)
            l_rf = F.mse_loss(v, phi - eps)
            loss = l_rf
            tau = (t / sched.num_train_timesteps).view(-1, 1, 1, 1, 1)
            x0h = xt + v * tau                                          # rectified-flow x0 estimate
            l_coc = phi.new_zeros(())
            if a.lam_coc > 0:
                gt_e = (phi * norm_const).pow(2).mean().detach()
                geo = to_geom(x0h, norm_const, a.cocycle_space)
                nc, acc = 0, phi.new_zeros(())
                for p_ in range(nb):
                    if float(t_pair[p_]) / sched.num_train_timesteps > a.tau_max:
                        continue
                    o = p_ * PAIR_FRAMES
                    res, _, _ = cocycle_terms(geo[o:o + K], geo[o + K:o + PAIR_FRAMES], deltas[p_], st)
                    acc = acc + torch.stack([r.pow(2).mean() for r in res]).mean()
                    nc += 1
                if nc:
                    l_coc = acc / nc / gt_e
                    loss = loss + lam_c * l_coc
            l_amp = phi.new_zeros(())
            if a.lam_amp > 0:
                keep = (t_pair / sched.num_train_timesteps <= a.tau_max).repeat_interleave(PAIR_FRAMES)
                if bool(keep.any()):
                    l_amp = loss_amp(x0h[keep], phi[keep])
                    loss = loss + a.lam_amp * l_amp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            with torch.no_grad():
                for k, p_ in ema_params.items():
                    ema[k].mul_(a.ema).add_(p_.detach(), alpha=1 - a.ema)
            agg["rflow"] += float(l_rf)
            agg["coc"] += float(l_coc)
            agg["amp"] += float(l_amp)
            agg["n"] += 1
        n = max(agg["n"], 1)
        line = f"ep {ep:3d}  L_rflow {agg['rflow']/n:.5f}  L_coc {agg['coc']/n:.5f}  L_amp {agg['amp']/n:.5f}  {time.time()-t0:.0f}s"

        if ep % a.val_every == 0 or ep == a.epochs:
            bak = {k: p_.detach().clone() for k, p_ in ema_params.items()}
            with torch.no_grad():
                for k, p_ in ema_params.items():
                    p_.copy_(ema[k])
            r = validate(unet, cond, sched, holder, va, dev, mu, sd, pairs_va, st, a.cocycle_space, norm_const, a.steps)
            with torch.no_grad():
                for k, p_ in ema_params.items():
                    p_.copy_(bak[k])
            hist.append({"ep": ep, "loss": agg["rflow"] / n, **r})
            payload = {"ema": ema, "ep": ep, "val": r, "cfg": vars(a), "amp_norm": {"mu": mu, "sd": sd},
                       "norm_const": norm_const}
            torch.save(payload, run / "last.pt")
            torch.save(payload, run / f"ckpt_ep{ep}.pt")
            (run / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
            line += f"  | comp_ratio {r['comp_ratio']:.4f}  njac {r['njac']:.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
