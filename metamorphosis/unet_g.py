"""Deterministic head G: 3D U-Net predicting the residual mean mu_t = E[I_t - hint_t]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = gn(cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class UNet3D(nn.Module):
    """4 levels (base, 2b, 4b, 8b); stride-2 convs; Z is only downsampled on the first two levels."""

    def __init__(self, cin, cout, base=40, levels=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(levels)]
        self.inp = Block(cin, ch[0])
        self.down, self.enc = nn.ModuleList(), nn.ModuleList()
        for i in range(1, levels):
            k = (2, 2, 2) if i <= 2 else (1, 2, 2)
            self.down.append(nn.Conv3d(ch[i - 1], ch[i - 1], k, stride=k))
            self.enc.append(Block(ch[i - 1], ch[i]))
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            k = (2, 2, 2) if i <= 2 else (1, 2, 2)
            self.up.append(nn.ConvTranspose3d(ch[i], ch[i - 1], k, stride=k))
            self.dec.append(Block(2 * ch[i - 1], ch[i - 1]))
        self.out = nn.Conv3d(ch[0], cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        hs = [self.inp(x)]
        for d, e in zip(self.down, self.enc):
            hs.append(e(d(hs[-1])))
        h = hs[-1]
        for u, dcd, s in zip(self.up, self.dec, reversed(hs[:-1])):
            h = dcd(torch.cat([u(h), s], 1))
        return self.out(h)
