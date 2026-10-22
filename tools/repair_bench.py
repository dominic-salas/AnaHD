"""Time the GPU cleaning step with and without breakup repair (lab venv, GPU):

    PYTHONPATH=src python tools/repair_bench.py clip.mp4 [frames]

Frames are decoded and prepared first, so only cleaning (fusion, repair, upscale) is timed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from anahd.gpu.pipeline import Cleaner, GpuConfig, Prepared
from anahd.media import probe, read_raw
from anahd.pipeline import Options, _windows, measure


def main() -> int:
    src = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    opts = Options()
    info = probe(src)
    meas = measure(src, info, opts)
    dev = torch.device("cuda")
    for ai in (False, True):
        times = {}
        for repair in (False, True):
            gcfg = GpuConfig(ai=ai, repair=repair)
            c = Cleaner(
                meas.crop, meas.osd, meas.curve, meas.size, info, opts.clean, opts.look, gcfg, dev
            )
            with torch.inference_mode():
                frames: list[Prepared] = []
                for raw in read_raw(src, info, meas.crop):
                    frames.append(c.prepare(raw))
                    if len(frames) == count:
                        break
                windows = list(_windows(iter(frames), opts.clean.radius))
                for centre, nbs in windows[:10]:  # warm up
                    c.clean(centre, nbs)
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for centre, nbs in windows[10:]:
                    c.clean(centre, nbs)
                end.record()
                torch.cuda.synchronize()
            times[repair] = start.elapsed_time(end) / (len(windows) - 10)
            del frames, windows, c
            torch.cuda.empty_cache()
        print(
            f"{'--ai' if ai else 'lanczos'}: {times[False]:.2f} ms/frame, with repair "
            f"{times[True]:.2f} ms/frame (+{times[True] - times[False]:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
