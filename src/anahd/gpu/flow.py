"""Dense optical flow from one frame to its neighbours with RAFT-small (torchvision weights).

Each frame is encoded once (features and context); the flow from a centre frame to all
of its neighbours is then refined in one batch. Both steps are replayed from CUDA graphs,
since at this size the GPU is otherwise left waiting on Python.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

T = TypeVar("T")


@dataclass(frozen=True)
class Encoded:
    features: Tensor  # (1, C, h/8, w/8), half precision
    context: Tensor


class _Corr:
    """RAFT's correlation pyramid and lookup, without host-side tensors (graph-safe)."""

    def __init__(self, levels: int, radius: int, device: torch.device) -> None:
        self.levels = levels
        d = torch.linspace(-radius, radius, 2 * radius + 1, device=device)
        side = 2 * radius + 1
        self.delta = torch.stack(torch.meshgrid(d, d, indexing="ij"), -1).view(1, side, side, 2)
        self.pyramid: list[Tensor] = []

    def build(self, f1: Tensor, f2: Tensor) -> None:
        b, c, h, w = f1.shape
        corr = torch.matmul(f1.view(b, c, h * w).transpose(1, 2), f2.view(b, c, h * w))
        corr = (corr * (1.0 / math.sqrt(c))).view(b * h * w, 1, h, w)
        self.pyramid = [corr]
        for _ in range(self.levels - 1):
            corr = F.avg_pool2d(corr, 2, 2)
            self.pyramid.append(corr)

    def lookup(self, coords: Tensor) -> Tensor:
        b, _, h, w = coords.shape
        c = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
        out = []
        for vol in self.pyramid:
            g = c + self.delta
            vh, vw = vol.shape[-2:]
            g = torch.stack([2 * g[..., 0] / (vw - 1) - 1, 2 * g[..., 1] / (vh - 1) - 1], -1)
            sampled = F.grid_sample(vol, g.to(vol.dtype), mode="bilinear", align_corners=True)
            out.append(sampled.view(b, h, w, -1))
            c = c / 2
        return torch.cat(out, -1).permute(0, 3, 1, 2).contiguous()


class Flow:
    """Flow from a centre frame to ``slots`` neighbours, at the size of the flow images."""

    def __init__(self, size: tuple[int, int], slots: int, iters: int, device: torch.device) -> None:
        w, h = size
        self.size = size
        self.pad_w, self.pad_h = (-w) % 8, (-h) % 8
        self.slots, self.iters, self.device = slots, iters, device
        model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        self.model = model
        self.corr = _Corr(model.corr_block.num_levels, model.corr_block.radius, device)
        ph, pw = (h + self.pad_h) // 8, (w + self.pad_w) // 8
        ys, xs = torch.meshgrid(
            torch.arange(ph, device=device), torch.arange(pw, device=device), indexing="ij"
        )
        self.coords0 = torch.stack([xs, ys], 0).float()[None].repeat(slots, 1, 1, 1)

        self._img = torch.zeros(1, 3, h + self.pad_h, w + self.pad_w, device=device)
        with torch.inference_mode():
            probe = self._encode()
        self._centre_f = torch.zeros_like(probe.features)
        self._centre_c = torch.zeros_like(probe.context)
        self._nb = torch.zeros(
            (slots, *probe.features.shape[1:]), dtype=probe.features.dtype, device=device
        )
        self._enc_graph, self._enc_out = self._capture(self._encode)
        self._flow_graph, self._flow_out = self._capture(self._flows)

    def _capture(self, fn: Callable[[], T]) -> tuple[torch.cuda.CUDAGraph, T]:
        with torch.inference_mode():
            side = torch.cuda.Stream(self.device)  # type: ignore[no-untyped-call, unused-ignore]
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
        return graph, out

    def _encode(self) -> Encoded:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            return Encoded(
                self.model.feature_encoder(self._img), self.model.context_encoder(self._img)
            )

    def _flows(self) -> Tensor:
        m = self.model
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            b = self.slots
            self.corr.build(self._centre_f.expand(b, -1, -1, -1), self._nb)
            hs = m.update_block.hidden_state_size
            hidden, ctx = torch.split(self._centre_c, [hs, self._centre_c.shape[1] - hs], 1)
            hidden = torch.tanh(hidden).expand(b, -1, -1, -1)
            ctx = F.relu(ctx).expand(b, -1, -1, -1)
            c0 = self.coords0
            c1 = c0.clone()
            for _ in range(self.iters):
                corr = self.corr.lookup(c1)
                hidden, delta = m.update_block(hidden, ctx, corr, (c1 - c0).to(hidden.dtype))
                c1 = c1 + delta.float()
            mask = None if m.mask_predictor is None else m.mask_predictor(hidden)
            flow = _upsample(c1 - c0, mask)
        w, h = self.size
        return flow[:, :, :h, :w]

    def encode(self, luma: Tensor) -> Encoded:
        """``luma``: (H, W) float in 0-255 at the flow size."""
        x = luma / 127.5 - 1.0
        if self.pad_w or self.pad_h:
            x = F.pad(x[None, None], (0, self.pad_w, 0, self.pad_h), mode="replicate")[0, 0]
        self._img.copy_(x.expand(3, -1, -1)[None])
        self._enc_graph.replay()
        out: Encoded = self._enc_out
        return Encoded(out.features.clone(), out.context.clone())

    def flows(self, centre: Encoded, neighbours: list[Encoded]) -> Tensor:
        """(slots, 2, h, w) flow from ``centre`` to each neighbour (x, y in pixels): the
        centre pixel p shows what the neighbour shows at p + flow. Missing slots repeat
        the last neighbour; the caller gives them no weight."""
        if not neighbours:
            raise ValueError("at least one neighbour is needed")
        self._centre_f.copy_(centre.features)
        self._centre_c.copy_(centre.context)
        for i in range(self.slots):
            self._nb[i].copy_(neighbours[min(i, len(neighbours) - 1)].features[0])
        self._flow_graph.replay()
        out: Tensor = self._flow_out
        return out.clone()


def _upsample(flow: Tensor, mask: Tensor | None) -> Tensor:
    b, c, h, w = flow.shape
    if mask is None:
        return 8 * F.interpolate(flow, size=(h * 8, w * 8), mode="bilinear", align_corners=True)
    weights = torch.softmax(mask.view(b, 1, 9, 8, 8, h, w).float(), 2)
    up = F.unfold(8 * flow, 3, padding=1).view(b, c, 9, 1, 1, h, w)
    return (weights * up).sum(2).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * 8, w * 8)
