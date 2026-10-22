"""Time the restorer at the real size (5 x 960x720 in, FP16) before training anything.

python -m anahd.learn.bench --sizes s m l --seconds 5
"""

from __future__ import annotations

import argparse
import time

import torch

from anahd.model.net import build, count_params


def bench(size: str, h: int, w: int, seconds: float, compile_: bool) -> float:
    dev = torch.device("cuda")
    net = build(size).to(dev).half().eval().to(memory_format=torch.channels_last)  # type: ignore[call-overload, unused-ignore]
    if compile_:
        net = torch.compile(net, mode="max-autotune")  # type: ignore[assignment, unused-ignore]
    x = torch.rand(1, 5, 3, h, w, device=dev, dtype=torch.half)
    for _ in range(10):
        net(x)
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        net(x)
        n += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["s", "m", "l"])
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    for s in a.sizes:
        with torch.inference_mode():
            ms = bench(s, a.h, a.w, a.seconds, a.compile)
        mp = count_params(build(s)) / 1e6
        print(f"size {s}: {mp:.2f} M params, {ms:.2f} ms/frame at {a.w}x{a.h}")


if __name__ == "__main__":
    main()
