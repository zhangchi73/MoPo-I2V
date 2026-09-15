from __future__ import annotations

import torch

_LP = [None]


def lpips_25d_train(pred, gt, n_slices=2):
    """2.5D LPIPS (AlexNet) on random slices of the three orthogonal planes; inputs in [0, 1]."""
    if _LP[0] is None:
        import lpips as _l
        n = _l.LPIPS(net="alex", verbose=False).to(pred.device).eval()
        for q in n.parameters():
            q.requires_grad_(False)
        _LP[0] = n
    net = _LP[0]
    vals = []
    for axis in (2, 3, 4):
        k = min(n_slices, pred.shape[axis])
        idx = torch.randint(0, pred.shape[axis], (k,), device=pred.device)
        p = pred.index_select(axis, idx).movedim(axis, 1).flatten(0, 1).clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        g = gt.index_select(axis, idx).movedim(axis, 1).flatten(0, 1).clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        vals.append(net(p, g).mean())
    return sum(vals) / 3.0
