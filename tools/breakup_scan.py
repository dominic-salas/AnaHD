"""Scan recordings for breakup bands (GPU). Run from the repo with the lab venv:

    PYTHONPATH=src python tools/breakup_scan.py clip.mp4 ... [--save signals_dir]

Per clip it prints the frames with flagged rows, total flagged rows, whole-bad frames and
the worst frames, using the same detector as ``anahd --gpu --repair``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from anahd.gpu.breakup import BreakupConfig, Detector, RowSignals, flag_rows
from anahd.gpu.pipeline import Cleaner, GpuConfig, background
from anahd.media import probe, read_raw
from anahd.pipeline import Options, _windows, measure


def scan(src: Path) -> dict[str, np.ndarray]:
    opts = Options()
    info = probe(src)
    meas = measure(src, info, opts)
    dev = torch.device("cuda")
    cleaner = Cleaner(
        meas.crop, meas.osd, meas.curve, meas.size, info, opts.clean, opts.look, GpuConfig(), dev
    )
    det = Detector(cleaner.keep)
    rows: dict[str, list[np.ndarray]] = {"disagree": [], "snow": [], "chroma": []}
    with torch.inference_mode():
        raws = background(read_raw(src, info, meas.crop))
        prepared = (cleaner.prepare(raw) for raw in raws)
        for centre, nbs in _windows(prepared, opts.clean.radius):
            a = cleaner.align(centre, nbs)
            s = det.signals(
                centre.y, centre.u, centre.v, a.y, a.u, a.v, a.valid, a.offsets, a.dist, chroma=True
            )
            for key in KEYS:
                rows[key].append(np.asarray(getattr(s, key).cpu().numpy()))
    return {k: np.stack(v) for k, v in rows.items()}


def summary(name: str, sig: dict[str, np.ndarray], cfg: BreakupConfig) -> str:
    flags = np.stack(
        [
            flag_rows(RowSignals(*(torch.from_numpy(sig[k][i]) for k in KEYS)), cfg).numpy()
            for i in range(len(sig["disagree"]))
        ]
    )
    count = flags.sum(1)
    whole = int((count == flags.shape[1]).sum())
    worst = [int(i) for i in np.argsort(-count, kind="stable")[:10] if count[i]]
    return (
        f"{name}: {len(count)} frames, {int((count > 0).sum())} with flagged rows, "
        f"{int(count.sum())} rows, {whole} whole-bad; worst {worst}"
    )


KEYS = ("disagree", "snow", "chroma")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--save", type=Path, help="write each clip's row signals here (.npz)")
    args = parser.parse_args()
    cfg = BreakupConfig()
    for clip in args.clips:
        began = time.monotonic()
        if clip.suffix == ".npz":  # signals saved earlier: no GPU needed
            with np.load(clip) as z:
                sig = {k: z[k] for k in KEYS}
        else:
            sig = scan(clip)
        took = time.monotonic() - began
        if args.save and clip.suffix != ".npz":
            args.save.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save / f"{clip.stem}.npz", **sig)
        print(summary(clip.stem, sig, cfg) + f" ({took:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
