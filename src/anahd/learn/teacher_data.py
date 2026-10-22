"""Training data for learning a slow teacher's look (SeedVR2) from real DVR input.

For each span of a recording the GPU cleaner (with repair) runs as in the pipeline, and per
span this writes into ``out/span_NNN/``:

  feed.mkv     the cleaner's picture before the look and the upscale (``Cleaner.restore``),
               lossless RGB (FFV1), for the teacher to upscale to 1440x1080
  stacks.npy   (K, 5, 3, H, W) uint8: what the learned restorer sees at K window centres
               (the repaired centre and its neighbours at -2..+2 warped onto it, YUV 4:4:4)
  flows.npy    (K, 2, 2, H/2, W/2) float16: flow from each centre to frames -1 and +1, in
               full-size pixels (to steady the teacher's frames into targets)
  osd.npy      (H, W) bool, True = OSD
  meta.json    source, start, frame indices of the centres, colour matrix

The teacher runs separately (``teacher.sh``); training pairs each stack with the teacher's
frames at the same indices. Inputs are the real recordings, so no synthetic damage is needed.

    python -m anahd.learn.teacher_data SPANS.json OUT --spans 120 --seconds 1.5
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.gpu.pipeline import Cleaner, GpuConfig
from anahd.media import probe, read_raw
from anahd.pipeline import Measured, Options, _windows, measure

EVERY = 4  # frames between window centres


def _rgb(cleaner: Cleaner, y: Tensor, u: Tensor, v: Tensor) -> np.ndarray:
    """Full-range planes (0..255, chroma at half size) -> (H, W, 3) uint8 RGB."""
    h, w = y.shape

    def up(c: Tensor) -> Tensor:
        return F.interpolate(c[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]

    ycc = torch.stack([y / 255.0, (up(u) - 128.0) / 255.0, (up(v) - 128.0) / 255.0])
    rgb = torch.einsum("ij,jhw->ihw", cleaner.rgb_from, ycc).clamp(0.0, 1.0)
    return np.ascontiguousarray((rgb * 255.0).round().byte().permute(1, 2, 0).cpu().numpy())


class Feed:
    """Lossless RGB video writer (FFV1 in Matroska), 60 fps."""

    def __init__(self, path: Path, w: int, h: int) -> None:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        cmd += ["-s", f"{w}x{h}", "-r", "60", "-i", "-", "-c:v", "ffv1", "-level", "3"]
        cmd += ["-pix_fmt", "bgr0", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg has no input pipe")
        self.proc.stdin.write(rgb.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed writing the feed")


def build_span(
    src: Path,
    start: float,
    seconds: float,
    out: Path,
    dev: torch.device,
    cache: dict[Path, Measured],
) -> dict[str, object]:
    info = probe(src)
    if src not in cache:
        cache[src] = measure(src, info, Options())
    meas = cache[src]
    opts = Options()
    cleaner = Cleaner(
        meas.crop,
        meas.osd,
        meas.curve,
        meas.size,
        info,
        opts.clean,
        opts.look,
        GpuConfig(repair=True),
        dev,
    )
    det = cleaner.detector
    if det is None:
        raise RuntimeError("repair is off")
    h, w = cleaner.h, cleaner.w
    out.mkdir(parents=True, exist_ok=False)
    feed = Feed(out / "feed.mkv", w, h)
    stacks: list[np.ndarray] = []
    flows: list[np.ndarray] = []
    centres: list[int] = []
    skipped = 0
    last = cleaner.stats
    raws = read_raw(src, info, meas.crop, start=start, duration=seconds)
    for i, (centre, nbs) in enumerate(
        _windows((cleaner.prepare(r) for r in raws), opts.clean.radius)
    ):
        a = cleaner.align(centre, nbs) if nbs else None
        if a is not None:
            a = cleaner.repair(det, centre, a)
            y, u, v = cleaner.blend(centre.y, centre.u, centre.v, a)
        else:
            y, u, v = centre.y, centre.u, centre.v
        y = cleaner.destripe(y)
        u, v = cleaner.derainbow(y, u, v)
        feed.write(_rgb(cleaner, y, u, v))
        st = cleaner.stats
        broken = st.whole != last.whole or st.left != last.left  # whole-frame outage
        last = st
        if i % EVERY != 2 or a is None or -1 not in a.offsets or 1 not in a.offsets:
            continue
        if broken:
            skipped += 1
            continue
        stack = cleaner.learned_stack(centre, a)
        stacks.append((stack * 255.0).round().clamp(0, 255).byte().cpu().numpy())
        fl = a.flow[[a.offsets.index(-1), a.offsets.index(1)]]
        flows.append(F.avg_pool2d(fl, 2).half().cpu().numpy())
        centres.append(i)
    feed.close()
    np.save(out / "stacks.npy", np.stack(stacks) if stacks else np.zeros((0, 5, 3, h, w), np.uint8))
    np.save(
        out / "flows.npy",
        np.stack(flows) if flows else np.zeros((0, 2, 2, h // 2, w // 2), np.float16),
    )
    np.save(out / "osd.npy", meas.osd)
    meta = {
        "source": str(src),
        "start_s": start,
        "seconds": seconds,
        "centres": centres,
        "skipped_broken": skipped,
        "size": [w, h],
        "bt709": bool(info.bt709),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spans", type=Path, help="JSON list from anahd.learn.spans")
    ap.add_argument("out", type=Path)
    ap.add_argument("--spans", type=int, default=120, dest="count")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--first", type=int, default=0, help="resume: skip spans already built")
    a = ap.parse_args()
    spans = json.loads(a.spans.read_text())
    rng = np.random.default_rng(5)
    pick = rng.permutation(len(spans))[: a.count]
    dev = torch.device("cuda")
    cache: dict[Path, Measured] = {}
    with torch.inference_mode():
        for n, k in enumerate(pick):
            if n < a.first:
                continue
            sp = spans[int(k)]
            dst = a.out / f"span_{n:03d}"
            if (dst / "meta.json").exists():
                continue
            meta = build_span(Path(sp["source"]), float(sp["start_s"]), a.seconds, dst, dev, cache)
            windows = meta["centres"]
            count = len(windows) if isinstance(windows, list) else 0
            print(
                f"{n}: {Path(sp['source']).name} @{sp['start_s']:.0f}s, {count} windows", flush=True
            )


if __name__ == "__main__":
    main()
