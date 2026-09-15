"""Generative bridge F: x0-parameterised flow-matching network with tau FiLM (channel-wise) and
sigma-map FiLM (spatial) on every block, plus a 2D PatchGAN discriminator shared by the three
orthogonal planes."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_g import gn


class FBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = gn(cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        self.tf = nn.Linear(tdim, 2 * cout)
        self.sf = nn.Conv3d(1, 2 * cout, 1)
        for m in (self.tf, self.sf):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, temb, smap):
        h = self.n1(self.c1(x))
        ts, tb = self.tf(temb)[:, :, None, None, None].chunk(2, 1)
        ss, sb = self.sf(smap).chunk(2, 1)
        h = F.silu(h * (1 + ts + ss) + tb + sb)
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class FUNet3D(nn.Module):
    """Input concat(x_tau, cond 7ch) = 8 ch -> residual 1 ch (zero-initialised)."""

    def __init__(self, cin=8, base=48, levels=4, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        ch = [base * 2 ** i for i in range(levels)]
        self.ks = [(2, 2, 2) if i <= 2 else (1, 2, 2) for i in range(1, levels)]
        self.inp = FBlock(cin, ch[0], tdim)
        self.down, self.enc = nn.ModuleList(), nn.ModuleList()
        for i in range(1, levels):
            self.down.append(nn.Conv3d(ch[i - 1], ch[i - 1], self.ks[i - 1], stride=self.ks[i - 1]))
            self.enc.append(FBlock(ch[i - 1], ch[i], tdim))
        self.mid = FBlock(ch[-1], ch[-1], tdim)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(ch[i], ch[i - 1], self.ks[i - 1], stride=self.ks[i - 1]))
            self.dec.append(FBlock(2 * ch[i - 1], ch[i - 1], tdim))
        self.out = nn.Conv3d(ch[0], 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def t_embed(self, tau):
        half = self.tdim // 2
        f = torch.exp(-math.log(1e4) * torch.arange(half, device=tau.device) / half)
        a = tau[:, None] * 1000.0 * f[None]
        return self.temb(torch.cat([a.sin(), a.cos()], 1))

    def forward(self, x, tau, smap):
        te = self.t_embed(tau)
        smaps = [smap]
        for k in self.ks:
            smaps.append(F.avg_pool3d(smaps[-1], k))
        hs = [self.inp(x, te, smaps[0])]
        for i, (d, e) in enumerate(zip(self.down, self.enc)):
            hs.append(e(d(hs[-1]), te, smaps[i + 1]))
        h = self.mid(hs[-1], te, smaps[-1])
        for j, (u, dcd, s) in enumerate(zip(self.up, self.dec, reversed(hs[:-1]))):
            h = dcd(torch.cat([u(h), s], 1), te, smaps[len(self.ks) - 1 - j])
        return self.out(h)


class PatchD2D(nn.Module):
    """2D PatchGAN on (slice, Laplacian high-pass, phi_bar slice (3)) = 5 channels."""

    def __init__(self, cin=5, base=64):
        super().__init__()
        L, c = [], cin
        for i, co in enumerate((base, base * 2, base * 4, base * 8)):
            L += [nn.Conv2d(c, co, 4, stride=2 if i < 3 else 1, padding=1)]
            if i > 0:
                L += [nn.GroupNorm(8, co)]
            L += [nn.LeakyReLU(0.2)]
            c = co
        L += [nn.Conv2d(c, 1, 4, padding=1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


_LAP = None


def add_hp(x):
    """x [N,C,H,W] (channel 0 = image) -> insert a Laplacian high-pass (x4) as channel 1."""
    global _LAP
    if _LAP is None or _LAP.device != x.device:
        _LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device).view(1, 1, 3, 3)
    return torch.cat([x[:, :1], F.conv2d(x[:, :1], _LAP, padding=1) * 4, x[:, 1:]], 1)
