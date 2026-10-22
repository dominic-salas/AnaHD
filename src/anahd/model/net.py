"""Small multi-frame restorer: 5 flow-aligned YUV frames in, cleaned centre frame out.

The pipeline already aligns neighbours to the centre frame with RAFT, so the network only
has to decide, per pixel, what is picture and what is analog damage. It works at half
resolution internally (pixel unshuffle, lossless) to stay inside the per-frame budget, and
predicts a residual on top of the centre frame, so an untrained or weak model is a no-op.
With ``scale`` 2 it also upscales: the residual goes on a bicubic 2x of the centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class NetConfig:
    frames: int = 5
    widths: tuple[int, int, int] = (48, 96, 160)
    blocks: tuple[int, int, int] = (2, 2, 4)
    scale: int = 1  # output size over input size


class ResBlock(nn.Module):  # type: ignore[misc, unused-ignore]
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = x + self.c2(self.act(self.c1(x)))
        return out


class Restorer(nn.Module):  # type: ignore[misc, unused-ignore]
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        w1, w2, w3 = cfg.widths
        b1, b2, b3 = cfg.blocks
        cin = cfg.frames * 3 * 4
        self.unshuffle = nn.PixelUnshuffle(2)
        self.head = nn.Sequential(nn.Conv2d(cin, w1, 3, padding=1), nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(*[ResBlock(w1) for _ in range(b1)])
        self.down1 = nn.Sequential(nn.Conv2d(w1, w2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(*[ResBlock(w2) for _ in range(b2)])
        self.down2 = nn.Sequential(nn.Conv2d(w2, w3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(*[ResBlock(w3) for _ in range(b3)])
        self.up2 = nn.Sequential(nn.Conv2d(w3, w2 * 4, 1), nn.PixelShuffle(2))
        self.dec2 = nn.Sequential(*[ResBlock(w2) for _ in range(b2)])
        self.up1 = nn.Sequential(nn.Conv2d(w2, w1 * 4, 1), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(*[ResBlock(w1) for _ in range(b1)])
        f = 2 * cfg.scale  # d1 is at half the input size
        self.tail = nn.Conv2d(w1, 3 * f * f, 3, padding=1)
        self.shuffle = nn.PixelShuffle(f)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)  # type: ignore[arg-type, unused-ignore]

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, 3, H, W) with the centre frame at T // 2; H and W divisible by 8."""
        b, t, c, h, w = x.shape
        centre = x[:, t // 2]
        z = self.unshuffle(x.reshape(b, t * c, h, w) - 0.5)
        e1 = self.enc1(self.head(z))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(self.up2(m) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        if self.cfg.scale != 1:
            centre = F.interpolate(
                centre, scale_factor=self.cfg.scale, mode="bicubic", align_corners=False
            )
        out: Tensor = centre + self.shuffle(self.tail(d1))
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build(name: str) -> Restorer:
    """Named sizes, so the speed check and training agree on what 'small' means."""
    sizes = {
        "s": NetConfig(widths=(32, 64, 128), blocks=(1, 2, 3)),
        "m": NetConfig(),
        "l": NetConfig(widths=(64, 128, 192), blocks=(2, 3, 6)),
        "m2": NetConfig(scale=2),
        "l2": NetConfig(widths=(64, 128, 192), blocks=(2, 3, 6), scale=2),
    }
    return Restorer(sizes[name])


if __name__ == "__main__":
    for n in ("s", "m", "l", "m2", "l2"):
        print(n, count_params(build(n)))
