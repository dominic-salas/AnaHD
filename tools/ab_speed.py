"""A/B a GPU setting on a real clip: end-to-end speed and whether the output changes.

    uv run python tools/ab_speed.py clip.mp4 [--runs 3] [--seconds 30] [--set pinned_upload=0]

Runs the whole GPU pipeline (decode, clean, the default look, x264), as the app does, with
``GpuConfig()`` (A) and with the ``--set`` overrides (B), alternating A and B ``--runs``
times so drift (heat, other load) hits both alike. Prints each run's wall time, the median
per side and the change, and whether the two outputs are the same frame for frame (an
MD5 of each decoded frame, as ``ffmpeg -f framemd5``).
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path

from anahd.gpu.pipeline import GpuConfig, enhance
from anahd.model import weights
from anahd.pipeline import Options


def framemd5(path: Path) -> str:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    frames = [line.split(",")[-1].strip() for line in out.splitlines() if not line.startswith("#")]
    return f"{len(frames)} frames, {hashlib.sha256(''.join(frames).encode()).hexdigest()[:16]}"


def parse(sets: list[str]) -> dict[str, object]:
    types = {f.name: f.type for f in fields(GpuConfig)}
    out: dict[str, object] = {}
    for item in sets:
        key, _, value = item.partition("=")
        kind = types[key]
        out[key] = (value not in ("0", "false")) if "bool" in str(kind) else int(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    base = GpuConfig(repair=True, learned=weights.resolve(None))
    configs = {"A": base, "B": replace(base, **parse(a.set))}  # type: ignore[arg-type]
    work = Path(tempfile.mkdtemp())
    times: dict[str, list[float]] = {"A": [], "B": []}
    outs: dict[str, Path] = {}
    for run in range(a.runs):
        for side, gcfg in configs.items():
            out = work / f"{side}{run}.mp4"
            began = time.monotonic()
            enhance(a.clip, out, Options(duration=a.seconds), gcfg=gcfg, overwrite=True)
            times[side].append(time.monotonic() - began)
            print(f"run {run + 1} {side}: {times[side][-1]:.1f} s", flush=True)
            outs.setdefault(side, out)
    ma, mb = statistics.median(times["A"]), statistics.median(times["B"])
    print(f"median A {ma:.1f} s, B {mb:.1f} s: B is {(ma - mb) / ma:+.1%} faster")
    ha, hb = framemd5(outs["A"]), framemd5(outs["B"])
    print(f"A {ha}\nB {hb}\n{'SAME OUTPUT' if ha == hb else 'OUTPUT DIFFERS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
