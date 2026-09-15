"""Non-autonomous flow-map diffusion network for Motion Generation.

Backbone: the 3D latent-diffusion U-Net of NV-Generate-MR (MAISI, MONAI), used as a frozen
spatial prior. On top of it:

* 22 temporal self-attention layers (one after every ResBlock / attention block on all levels),
  zero-initialised output projection so training starts from the identity;
* LoRA (rank 8) on every spatial 3x3x3 convolution;
* conv_in re-wired to 7 channels = noisy displacement field (3) | anchor-frame latent z0 (4),
  conv_out re-wired to 3 channels (dz, dy, dx);
* three scalar conditions (frame phase, anchor phase, motion amplitude), each through a
  zero-initialised linear MLP added to the time embedding (per-layer FiLM through the ResBlocks).

The network predicts the flow map phi_{0->t} for K frames jointly (holder["T"] = K), which makes
the map non-autonomous: every frame gets its own displacement rather than the integral of one
stationary velocity field.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn

MAX_T = 32
N_TEMPORAL_EXPECTED = 22


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32)[:, None]
    i = torch.arange(dim, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10000.0, (2 * (i // 2)) / dim)
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(ang[:, 0::2])
    pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe


class TemporalAttention(nn.Module):
    """Temporal self-attention + FFN over the K frames of a window; identity at initialisation."""

    def __init__(self, channels: int, holder: dict, num_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_mult * channels), nn.GELU(),
                                 nn.Linear(ffn_mult * channels, channels))
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.register_buffer("pe", sinusoidal_pe(MAX_T, channels), persistent=False)
        self.holder = holder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self.holder["T"]
        if T == 1:
            return x
        bt, c = x.shape[:2]
        b = bt // T
        sp = tuple(x.shape[2:])
        s = sp[0] * sp[1] * sp[2]
        seq = x.reshape(b, T, c, s).permute(0, 3, 1, 2).reshape(b * s, T, c)
        seq_n = self.norm1(seq) + self.pe[:T].to(seq.dtype)
        a, _ = self.attn(seq_n, seq_n, seq_n, need_weights=False)
        h = seq + a
        h = h + self.ffn(self.norm2(h))
        h = seq + self.proj_out(h)
        return h.reshape(b, s, T, c).permute(0, 2, 3, 1).reshape(bt, c, *sp)


class TemporalWrapper(nn.Module):
    def __init__(self, base, temporal):
        super().__init__()
        self.base, self.temporal = base, temporal

    def forward(self, x, *a, **k):
        return self.temporal(self.base(x, *a, **k))


class LoRAConv3d(nn.Module):
    def __init__(self, base: nn.Conv3d, rank: int = 8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Conv3d(base.in_channels, rank, base.kernel_size, stride=base.stride,
                           padding=base.padding, bias=False)
        self.B = nn.Conv3d(rank, base.out_channels, 1, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))


class CondMLP(nn.Module):
    """Scalar condition in [0, 1] -> additive time-embedding increment (zero-initialised)."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v):
        return self.net(v.reshape(-1, 1).float())


def inject_lora(unet: nn.Module, rank: int = 8, skip: tuple = ()) -> int:
    skip_ids = {id(m) for m in skip}
    targets = []
    for mod in unet.modules():
        for name, child in mod.named_children():
            if isinstance(child, nn.Conv3d) and max(child.kernel_size) > 1 and id(child) not in skip_ids:
                targets.append((mod, name, child))
    for mod, name, child in targets:
        setattr(mod, name, LoRAConv3d(child, rank))
    return len(targets)


def build_base_unet(base_ckpt: str, base_cfg: str, dev):
    """MAISI DiffusionModelUNet (MONAI >= 1.4) initialised from the NV-Generate-MR rectified-flow weights."""
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi

    cfg = json.load(open(base_cfg))["diffusion_unet_def"]
    kwargs = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k, ref in [("spatial_dims", 3), ("in_channels", 4), ("out_channels", 4)]:
        if isinstance(kwargs.get(k), str):
            kwargs[k] = ref
    for k in ("include_top_region_index_input", "include_bottom_region_index_input"):
        if isinstance(kwargs.get(k), str):
            kwargs[k] = False
    model = DiffusionModelUNetMaisi(**kwargs)
    sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    for key in ("unet_state_dict", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    missing, _ = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0, missing[:5]
    return model.to(dev).eval()


def build_motion_unet(holder: dict, dev, base_ckpt: str, base_cfg: str, lora_rank: int = 8):
    """-> (unet, cond_mods, trainable_params). Spatial layers frozen + LoRA; temporal layers trainable."""
    unet = build_base_unet(base_ckpt, base_cfg, dev)
    for p in unet.parameters():
        p.requires_grad_(False)

    n_tmp = 0

    def wrap(lst, ch_of):
        nonlocal n_tmp
        for i in range(len(lst)):
            lst[i] = TemporalWrapper(lst[i], TemporalAttention(ch_of(lst[i]), holder))
            n_tmp += 1

    attn_ch = lambda m: m.norm.num_channels
    res_ch = lambda m: m.out_channels
    for blk in unet.down_blocks:
        wrap(blk.attentions, attn_ch) if hasattr(blk, "attentions") else wrap(blk.resnets, res_ch)
    mid_c = unet.middle_block.attention.norm.num_channels
    unet.middle_block.attention = TemporalWrapper(unet.middle_block.attention, TemporalAttention(mid_c, holder))
    unet.middle_block.resnet_2 = TemporalWrapper(
        unet.middle_block.resnet_2, TemporalAttention(unet.middle_block.resnet_2.out_channels, holder))
    n_tmp += 2
    for blk in unet.up_blocks:
        wrap(blk.attentions, attn_ch) if hasattr(blk, "attentions") else wrap(blk.resnets, res_ch)
    assert n_tmp == N_TEMPORAL_EXPECTED, f"expect {N_TEMPORAL_EXPECTED} temporal layers, got {n_tmp}"

    # conv_in 4 -> 7 (noisy field 3 | z0 latent 4); conv_out 4 -> 3
    old = unet.conv_in.conv
    new = nn.Conv3d(7, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding).to(dev)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight[:, :3])
        new.bias.copy_(old.bias)
    unet.conv_in.conv = new
    oc = unet.out[-1].conv if hasattr(unet.out[-1], "conv") else unet.out[-1]
    new_o = nn.Conv3d(oc.in_channels, 3, kernel_size=oc.kernel_size, stride=oc.stride, padding=oc.padding).to(dev)
    with torch.no_grad():
        new_o.weight.copy_(oc.weight[:3])
        new_o.bias.copy_(oc.bias[:3])
    if hasattr(unet.out[-1], "conv"):
        unet.out[-1].conv = new_o
    else:
        unet.out[-1] = new_o

    if lora_rank > 0:
        inject_lora(unet, lora_rank, skip=(new, new_o))

    cond_dim = unet.time_embed[-1].out_features if hasattr(unet, "time_embed") else mid_c
    cond = nn.ModuleDict({"phase": CondMLP(cond_dim), "anchor_phase": CondMLP(cond_dim),
                          "amp": CondMLP(cond_dim)}).to(dev)

    def _cond_hook(_m, _i, out):
        inc = holder.get("cond_emb")
        return out if inc is None else out + inc

    unet.time_embed.register_forward_hook(_cond_hook)
    unet = unet.to(dev)
    trainable = [p for p in unet.parameters() if p.requires_grad] + list(cond.parameters())
    return unet, cond, trainable
