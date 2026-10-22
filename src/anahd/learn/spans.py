"""Pick the source spans the training set is cut from.

Every DVR recording (1280x720, full-range 4:2:0, 60 fps, at least 20 s) under the given
folders is found, duplicates (the same recording under two names) are kept once, the sealed
test recordings are left out entirely, and the time ranges the dev excerpts came from are left
out with a margin. Spans are then spread evenly over what is left, more over day footage.

    python -m anahd.learn.spans OUT.json DIR [DIR ...] --windows 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Recordings by content (sha1 of the first 4 MB), so renamed copies are caught too.
SEALED = {"20261009-180033", "20261008-210503", "20261009-182156"}
# Dev excerpt sources (A0 README) and their start in seconds; [start-2, start+22] is left out.
DEV = {
    "hdz_0061": 60.0,
    "20261008-143238": 15.0,
    "20261009-181057": 165.0,
    "low light": 50.0,
    "nighttime, incredibly low light": 196.0,
    "indoors cruising": 215.0,
    "urban building dive": 70.0,
    "20261010-210449": 25.0,
}
DEV_MARGIN = (2.0, 22.0)
MIN_DURATION = 20.0
SPAN = 6.0  # seconds per span
STRIDE = 12  # frames between windows
EDGE = 3.0  # seconds kept clear of a recording's start and end


@dataclass(frozen=True)
class Span:
    source: str
    start_s: float
    dur_s: float
    stride: int
    day: bool


@dataclass(frozen=True)
class Recording:
    path: Path
    duration: float
    key: str
    luma: float


def _probe(path: Path) -> tuple[int, int, str, str, float] | None:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return None
    d = json.loads(out.stdout)
    if not d.get("streams"):
        return None
    s = d["streams"][0]
    return (
        int(s["width"]),
        int(s["height"]),
        str(s["pix_fmt"]),
        str(s["r_frame_rate"]),
        float(d["format"]["duration"]),
    )


def _key(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.sha1(f.read(4_000_000), usedforsecurity=False).hexdigest()


def _mean_luma(path: Path, duration: float) -> float:
    """Mean luma (0..1) of a few small frames spread over the recording."""
    vals = []
    for t in np.linspace(duration * 0.15, duration * 0.85, 5):
        out = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                f"{t:.2f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=64:36,format=gray",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=False,
        )
        if len(out.stdout) == 64 * 36:
            vals.append(np.frombuffer(out.stdout, np.uint8).mean() / 255.0)
    return float(np.mean(vals)) if vals else 0.0


def find(dirs: list[Path]) -> tuple[list[Recording], dict[str, str]]:
    """DVR recordings under ``dirs``, one per content key, and the content key of every
    file stem seen (for matching the sealed and dev lists by any of their names)."""
    seen: dict[str, Recording] = {}
    stems: dict[str, set[str]] = {}
    for d in dirs:
        for p in sorted([*d.glob("*.mp4"), *d.glob("*.ts")]):
            if "_montage" in p.stem:
                continue
            pr = _probe(p)
            if pr is None:
                continue
            w, h, fmt, rate, dur = pr
            if (w, h) != (1280, 720) or fmt != "yuvj420p" or rate != "60/1":
                continue
            if dur < MIN_DURATION:
                continue
            k = _key(p)
            stems.setdefault(k, set()).add(p.stem)
            if k not in seen:
                seen[k] = Recording(p, dur, k, 0.0)
    # Same recording in another container (.ts vs .mp4): same stem and duration.
    by_stem: dict[tuple[str, int], str] = {}
    for k, r in list(seen.items()):
        sk = (r.path.stem, round(r.duration))
        if sk in by_stem:
            stems[by_stem[sk]] |= stems.pop(k)
            del seen[k]
        else:
            by_stem[sk] = k
    name_keys = {s: k for k, v in stems.items() for s in v}
    return list(seen.values()), name_keys


def plan(dirs: list[Path], windows: int, day_weight: float = 2.0) -> list[Span]:
    recs, name_keys = find(dirs)
    sealed = {name_keys[n] for n in SEALED if n in name_keys}
    missing = SEALED - name_keys.keys()
    if missing:
        raise SystemExit(f"sealed recordings not found, cannot exclude them safely: {missing}")
    dev: dict[str, float] = {}
    for name, st in DEV.items():
        if name not in name_keys:
            raise SystemExit(f"dev recording not found, cannot exclude its range: {name}")
        dev[name_keys[name]] = st
    recs = [r for r in recs if r.key not in sealed]
    recs = [Recording(r.path, r.duration, r.key, _mean_luma(r.path, r.duration)) for r in recs]

    # Usable intervals per recording.
    usable: list[tuple[Recording, list[tuple[float, float]]]] = []
    for r in recs:
        ivs = [(EDGE, r.duration - EDGE)]
        if r.key in dev:
            lo, hi = dev[r.key] - DEV_MARGIN[0], dev[r.key] + DEV_MARGIN[1]
            ivs = [(a, min(b, lo)) for a, b in ivs] + [(max(a, hi), b) for a, b in ivs]
        ivs = [(a, b) for a, b in ivs if b - a >= SPAN]
        if ivs:
            usable.append((r, ivs))

    per_span = SPAN * 60 / STRIDE
    total_spans = math.ceil(windows * 1.5 / per_span)  # room for rejected windows
    day = {r.key: r.luma > 0.25 for r, _ in usable}
    weight = {
        r.key: sum(b - a for a, b in ivs) * (day_weight if day[r.key] else 1.0) for r, ivs in usable
    }
    wsum = sum(weight.values())
    spans: list[Span] = []
    for r, ivs in usable:
        n = max(1, round(total_spans * weight[r.key] / wsum))
        length = sum(b - a - SPAN for a, b in ivs)
        for j in range(n):
            # Evenly over the usable time (minus one span length per interval).
            pos = (j + 0.5) / n * length
            for a, b in ivs:
                room = b - a - SPAN
                if pos <= room:
                    spans.append(Span(str(r.path), round(a + pos, 2), SPAN, STRIDE, day[r.key]))
                    break
                pos -= room
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--windows", type=int, default=3000)
    a = ap.parse_args()
    spans = plan(a.dirs, a.windows)
    a.out.write_text(json.dumps([asdict(s) for s in spans], indent=1))
    days = sum(s.day for s in spans)
    srcs = len({s.source for s in spans})
    print(f"{len(spans)} spans from {srcs} recordings ({days} day)")


if __name__ == "__main__":
    main()
