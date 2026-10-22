"""Run a trained restorer inside the pipeline: aligned 5-frame stack in, cleaned frame out."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .net import build


class LearnedRestorer:
    def __init__(
        self, ckpt: Path, device: torch.device, strength: float = 1.0, graphs: bool = True
    ) -> None:
        st = torch.load(ckpt, map_location=device, weights_only=True)
        net = build(str(st["size"])).to(device)
        net.load_state_dict(st["net"])
        net = net.half().eval()
        self.net = net.to(memory_format=torch.channels_last)  # type: ignore[call-overload, unused-ignore]
        self.strength = strength
        self.scale = int(net.cfg.scale)  # output size over input size
        # The network is many small kernels at this size: replay it from a CUDA graph.
        self.graphs = graphs and device.type == "cuda"
        self._graph: torch.cuda.CUDAGraph | None = None
        self._in: Tensor | None = None
        self._out: Tensor | None = None

    def __call__(self, stack: Tensor, osd: Tensor | None = None) -> Tensor:
        with torch.inference_mode():
            return self._run(stack, osd)

    def _forward(self, x: Tensor) -> Tensor:
        if not self.graphs:
            out: Tensor = self.net(x)
            return out
        if self._graph is None or self._in is None or self._in.shape != x.shape:
            self._in = x.clone()
            dev = x.device
            side = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call, unused-ignore]
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.net(self._in)
            torch.cuda.current_stream(dev).wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._out = self.net(self._in)
        self._in.copy_(x)
        self._graph.replay()
        if self._out is None:
            raise RuntimeError("graph captured without an output")
        return self._out

    def _run(self, stack: Tensor, osd: Tensor | None) -> Tensor:
        """stack: (5, 3, H, W) YUV in [0, 1], neighbours already warped onto frame 2.

        osd: (H, W) bool at the output size, True = OSD; those pixels keep the centre frame
        (upscaled like the network's output when it upscales).
        Any size: the network needs multiples of 8, so the edges are repeated up to that.
        """
        h, w = stack.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        x = stack.half()
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        s = self.scale
        out = self._forward(x.unsqueeze(0))[0, :, : h * s, : w * s].float()
        centre = stack[2:3].float()
        if s != 1:
            centre = F.interpolate(centre, scale_factor=s, mode="bicubic", align_corners=False)
        centre = centre[0]
        if self.strength != 1.0:
            out = centre + self.strength * (out - centre)
        if osd is not None:
            out = torch.where(osd.unsqueeze(0), centre, out)
        return out.clamp(0.0, 1.0)
