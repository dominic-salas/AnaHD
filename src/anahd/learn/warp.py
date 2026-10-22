"""Backward warp: sample `img` at x + flow(x), so a neighbour lands on the centre frame."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def backward_warp(img: Tensor, flow: Tensor) -> Tensor:
    """img (N, C, H, W), flow (N, 2, H, W) in pixels (x, y)."""
    _, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=flow.dtype),
        torch.arange(w, device=img.device, dtype=flow.dtype),
        indexing="ij",
    )
    gx = (xs + flow[:, 0]) * (2.0 / max(w - 1, 1)) - 1.0
    gy = (ys + flow[:, 1]) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        img, grid.to(img.dtype), mode="bilinear", padding_mode="border", align_corners=True
    )
