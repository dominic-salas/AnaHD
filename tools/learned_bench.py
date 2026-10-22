"""Time the GPU cleaning step with the learned restorer (lab venv, GPU):

    PYTHONPATH=src python tools/learned_bench.py clip.mp4 CKPT [CKPT ...] [--frames 240] [--ai]

Frames are decoded and prepared first, so only cleaning (fusion or the network, repair,
upscale) is timed; encoding runs on the CPU in parallel in the real pipeline. Prints ms/frame
for --repair alone and with each checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anahd.gpu.pipeline import Cleaner, GpuConfig, Prepared
from anahd.media import probe, read_raw
from anahd.pipeline import Options, _windows, measure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("ckpts", type=Path, nargs="*")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--ai", action="store_true")
    a = ap.parse_args()
    opts = Options()
    info = probe(a.clip)
    meas = measure(a.clip, info, opts)
    dev = torch.device("cuda")
    for ck in [None, *a.ckpts]:
        gcfg = GpuConfig(ai=a.ai, repair=True, learned=ck)
        c = Cleaner(
            meas.crop, meas.osd, meas.curve, meas.size, info, opts.clean, opts.look, gcfg, dev
        )
        with torch.inference_mode():
            frames: list[Prepared] = []
            for raw in read_raw(a.clip, info, meas.crop):
                frames.append(c.prepare(raw))
                if len(frames) == a.frames:
                    break
            windows = list(_windows(iter(frames), opts.clean.radius))
            for centre, nbs in windows[:10]:  # warm up (and capture the graphs)
                c.clean(centre, nbs)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for centre, nbs in windows[10:]:
                c.clean(centre, nbs)
            end.record()
            torch.cuda.synchronize()
        ms = start.elapsed_time(end) / (len(windows) - 10)
        name = ck.name if ck else "repair only"
        print(f"{'--ai ' if a.ai else ''}{name}: {ms:.2f} ms/frame ({1000 / ms:.1f} fps)")
        del frames, windows, c
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
