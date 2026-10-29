"""The gate for a change that alters the output for speed: is it no worse than today?
(lab venv, GPU; see docs/designs/live-view-and-speed.md, "Build spec", the gate)

    uv run python tools/gate.py OUT_DIR --set flow_iters=4 [--real clip.mp4 ...]

Renders each synthetic scene (known truth) with today's ``GpuConfig`` (off) and with the
``--set`` overrides (on), at the shipped defaults (Learned, repair, the look), and compares:

- drift with breakup bands: PSNR vs truth, in the band rows of damaged frames, and flicker
  (how much the error changes from frame to frame);
- the 180 deg/s roll and the 720 deg/s flip: PSNR, and texture detail kept (higher better);
- a static scene: pulsing (frame-to-frame change where nothing moves), mean and the worst
  1% of 32x32 tiles;
- whole-frame outages: PSNR of the outage frames and of their neighbours;
- the OSD: a box around it (luma of a 6 px ring outside it against a band beyond).

Thresholds: PSNR on >= off - 0.05 dB; lower-is-better metrics on <= off x 1.02 (pulsing's
worst tiles x 1.05); detail on >= off x 0.98; OSD box |ring - band| on - off <= 1.0 level.
Then each ``--real`` clip is rendered both ways for the eye check: the per-frame mean
|on - off|, the 20 frames that differ most saved side by side (off | on | x8 difference)
and temporal strips (one row over 120 frames), plus the same pulsing measure where the
off render is near-still. Writes OUT_DIR/gate.json and prints PASS or FAIL per check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import synthetic as syn
from anahd.gpu.pipeline import GpuConfig, enhance
from anahd.model import weights
from anahd.pipeline import Options

Img = NDArray[np.uint8]
TILE = 32
OUTAGES = (40, 41, 90)  # whole frames of snow (two in a row, then one)


def read(path: Path, size: tuple[int, int] = (syn.OUT_W, syn.OUT_H)) -> Iterator[Img]:
    """Frames one at a time (a clip at 1440x1080 held whole is gigabytes), at ``size``."""
    return syn.read_bgr(path, size, f"scale={size[0]}:{size[1]}")


def fit(frame: Img, truth: Img) -> NDArray[np.float64]:
    x = np.c_[frame.reshape(-1, 3).astype(np.float64), np.ones(frame.shape[0] * frame.shape[1])]
    m, *_ = np.linalg.lstsq(x, truth.reshape(-1, 3).astype(np.float64), rcond=None)
    return np.asarray(x @ m).reshape(truth.shape)


def psnr(a: NDArray[np.float64], b: Img) -> float:
    mse = float(np.mean((a - b.astype(np.float64)) ** 2))
    return float(10 * np.log10(255**2 / max(mse, 1e-9)))


def detail(img: NDArray[np.float64] | Img) -> float:
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def write_custom(path: Path, frames: int, make: Callable[[int, np.random.Generator], Img]) -> None:
    lay = syn.FOUR_THREE
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    cmd += ["-s", f"{lay.dvr_w}x{lay.dvr_h}", "-r", str(syn.FPS), "-i", "-"]
    cmd += ["-c:v", "libx264", "-crf", "10", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    proc = subprocess.Popen([*cmd, str(path)], stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("no pipe to ffmpeg")
    rng = np.random.default_rng(5)
    for i in range(frames):
        proc.stdin.write(make(i, rng).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def outage_frame(i: int, rng: np.random.Generator) -> Img:
    if i in OUTAGES:
        lay = syn.FOUR_THREE
        frame = np.zeros((lay.dvr_h, lay.dvr_w, 3), np.uint8)
        snow = rng.uniform(0, 255, (lay.pic_h, lay.pic_w, 3)).astype(np.uint8)
        frame[:, lay.pic_x : lay.pic_x + lay.pic_w] = snow
        return frame
    return syn.degrade(syn.clean_frame(i), syn.FOUR_THREE, rng, band=False)


def render(src: Path, out: Path, gcfg: GpuConfig, seconds: float | None = None) -> Path:
    if not out.exists():
        enhance(src, out, Options(duration=seconds), gcfg=gcfg, overwrite=True)
    return out


class Still:
    """Running sum of |frame - previous frame| (grey), for pulsing where nothing moves."""

    def __init__(self) -> None:
        self.prev: NDArray[np.float32] | None = None
        self.total: NDArray[np.float32] | None = None
        self.n = 0

    def add(self, frame: Img) -> None:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.prev is not None:
            d = np.abs(g - self.prev)
            self.total = d if self.total is None else self.total + d
            self.n += 1
        self.prev = g

    def mean_map(self) -> NDArray[np.float32]:
        if self.total is None:
            raise ValueError("fewer than two frames")
        return self.total / self.n


def pulsing(d: NDArray[np.float32], keep: NDArray[np.bool_] | None = None) -> tuple[float, float]:
    """Mean change over ``keep``, and the 99th percentile over 32x32 tiles inside it."""
    h, w = d.shape
    th, tw = h // TILE, w // TILE
    tiles = d[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE).mean(axis=(1, 3))
    if keep is not None:
        share = keep[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE).mean(axis=(1, 3))
        k = share == 1.0
        if not k.any():  # the still parts are scattered: tiles that are mostly still
            k = share >= 0.5
        vals = tiles[k]
    else:
        vals = tiles.ravel()
    return float(vals.mean()), float(np.percentile(vals, 99))


def osd_box(frame: Img) -> float:
    """|mean luma of a 6 px ring just outside the OSD text box - a 6 px band beyond it|."""
    x0, y1 = syn.OSD_ORG
    (tw, th), _ = cv2.getTextSize(syn.OSD_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 9)
    a, b, c, d = x0 - 4, y1 - th - 8, x0 + tw + 4, y1 + 12
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring = np.r_[g[b - 6 : b, a:c].ravel(), g[d : d + 6, a:c].ravel()]
    band = np.r_[g[b - 12 : b - 6, a:c].ravel(), g[d + 6 : d + 12, a:c].ravel()]
    return abs(float(ring.mean()) - float(band.mean()))


def synthetic(work: Path, cfgs: dict[str, GpuConfig]) -> dict[str, dict[str, float]]:
    m: dict[str, dict[str, float]] = {k: {} for k in cfgs}
    bands = work / "bands.mp4"
    if not bands.exists():
        syn.write(bands, 150)
    roll = work / "roll.mp4"
    if not roll.exists():
        syn.write_with(roll, syn.FLIP.stop + 5, syn.roll_truth, syn.FOUR_THREE)
    still = work / "still.mp4"
    if not still.exists():
        syn.write_with(still, 120, lambda i: syn.clean_frame(0), syn.FOUR_THREE)
    outage = work / "outage.mp4"
    if not outage.exists():
        write_custom(outage, 120, outage_frame)
    osd_free = np.ones((syn.OUT_H, syn.OUT_W), bool)
    osd_free[900:, :] = False  # leave the OSD out of "nothing moves"
    r0, r1 = (round(r * syn.OUT_H / syn.PIC_H) for r in syn.BAND_ROWS)
    near = {i for k in OUTAGES for i in range(k - 3, k + 4)} - set(OUTAGES)
    for side, gcfg in cfgs.items():
        res = m[side]
        whole, band, flick, box = [], [], [], []
        prev_err: NDArray[np.float64] | None = None
        for i, f in enumerate(read(render(bands, work / f"bands_{side}.mp4", gcfg))):
            t = syn.clean_frame(i)
            ft = fit(f, t)
            whole.append(psnr(ft, t))
            if syn.damaged(i):
                band.append(psnr(ft[r0:r1], t[r0:r1]))
            err = ft - t.astype(np.float64)
            if prev_err is not None:
                flick.append(float(np.abs(err - prev_err).mean()))
            prev_err = err
            box.append(osd_box(f))
        res["bands.psnr"], res["bands.band_psnr"] = float(np.mean(whole)), float(np.mean(band))
        res["bands.flicker"], res["osd.box"] = float(np.mean(flick)), float(np.mean(box))

        spans = {"roll": set(syn.ROLL), "flip": set(syn.FLIP)}
        got: dict[str, list[float]] = {f"{k}.{q}": [] for k in spans for q in ("psnr", "detail")}
        for i, f in enumerate(read(render(roll, work / f"roll_{side}.mp4", gcfg))):
            for name, span in spans.items():
                if i in span:
                    t = syn.roll_truth(i)
                    ft = fit(f, t)
                    got[f"{name}.psnr"].append(psnr(ft, t))
                    got[f"{name}.detail"].append(detail(ft))
        res.update({k: float(np.mean(v)) for k, v in got.items()})

        acc = Still()
        for i, f in enumerate(read(render(still, work / f"still_{side}.mp4", gcfg))):
            if i >= 10:
                acc.add(f)
        res["still.pulsing"], res["still.pulsing_worst"] = pulsing(acc.mean_map(), osd_free)

        frames, neigh = [], []
        for i, f in enumerate(read(render(outage, work / f"outage_{side}.mp4", gcfg))):
            if i in OUTAGES or i in near:
                t = syn.clean_frame(i)
                (frames if i in OUTAGES else neigh).append(psnr(fit(f, t), t))
        res["outage.frames_psnr"], res["outage.neighbours_psnr"] = (
            float(np.mean(frames)),
            float(np.mean(neigh)),
        )
        print(f"{side}: {json.dumps({k: round(v, 3) for k, v in res.items()})}", flush=True)
    return m


HIGHER = ("psnr", "detail")


def judge(m: dict[str, dict[str, float]]) -> list[tuple[str, float, float, bool, str]]:
    rows = []
    for key in m["off"]:
        off, on = m["off"][key], m["on"][key]
        if key.endswith("psnr"):
            ok, rule = on >= off - 0.05, "on >= off - 0.05 dB"
        elif key.endswith("detail"):
            ok, rule = on >= off * 0.98, "on >= off x 0.98"
        elif key == "osd.box":
            ok, rule = on - off <= 1.0, "on - off <= 1.0"
        elif key.endswith("pulsing_worst"):
            ok, rule = on <= off * 1.05, "on <= off x 1.05"
        else:
            ok, rule = on <= off * 1.02, "on <= off x 1.02"
        rows.append((key, off, on, ok, rule))
    return rows


def real(work: Path, clip: Path, cfgs: dict[str, GpuConfig], seconds: float) -> dict[str, float]:
    """Both renders side by side, one frame pair at a time; the frames that differ most are
    decoded again for the eye check (keeping them all would not fit in memory)."""
    size = (960, 720)
    paths = {
        s: render(clip, work / f"real_{clip.stem}_{s}.mp4", g, seconds) for s, g in cfgs.items()
    }
    diffs: list[float] = []
    acc = {"off": Still(), "on": Still()}
    rows: dict[str, list[Img]] = {"off": [], "on": []}
    for off, on in zip(read(paths["off"], size), read(paths["on"], size), strict=False):
        diffs.append(float(np.abs(off.astype(np.int16) - on).mean()))
        for side, f in (("off", off), ("on", on)):
            acc[side].add(f)
            if len(rows[side]) < 120:
                rows[side].append(f[f.shape[0] // 2].copy())
    still_map = acc["off"].mean_map()
    keep = still_map <= np.percentile(still_map, 20)  # the stillest fifth of the picture (off)
    po, pw = pulsing(acc["off"].mean_map(), keep)
    qo, qw = pulsing(acc["on"].mean_map(), keep)
    folder = work / f"eye_{clip.stem}"
    folder.mkdir(exist_ok=True)
    top = set(sorted(range(len(diffs)), key=lambda i: -diffs[i])[:20])
    pairs = zip(read(paths["off"], size), read(paths["on"], size), strict=False)
    for i, (off, on) in enumerate(pairs):
        if i in top:
            d = np.clip(np.abs(off.astype(np.int16) - on) * 8, 0, 255).astype(np.uint8)
            cv2.imwrite(str(folder / f"top_{i:05d}.png"), np.hstack([off, on, d]))
    for side, strip in rows.items():
        img = np.stack(strip)
        cv2.imwrite(
            str(folder / f"strip_{side}.png"),
            cv2.resize(img, (img.shape[1], 360), interpolation=cv2.INTER_NEAREST),
        )
    res = {
        "mean_abs_diff": float(np.mean(diffs)),
        "max_frame_diff": float(np.max(diffs)),
        "still_pulsing_off": po,
        "still_pulsing_on": qo,
        "still_worst_off": pw,
        "still_worst_on": qw,
    }
    print(f"{clip.name}: {json.dumps({k: round(v, 3) for k, v in res.items()})}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--real", type=Path, action="append", default=[])
    ap.add_argument("--seconds", type=float, default=12.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    types = {f.name: str(f.type) for f in fields(GpuConfig)}
    over: dict[str, object] = {}
    for item in a.set:
        k, _, v = item.partition("=")
        over[k] = (v not in ("0", "false")) if "bool" in types[k] else int(v)
    base = GpuConfig(repair=True, learned=weights.resolve(None))
    cfgs = {"off": base, "on": replace(base, **over)}  # type: ignore[arg-type]
    m = synthetic(a.out, cfgs)
    rows = judge(m)
    for key, off, on, ok, rule in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {key:28s} off {off:9.3f}  on {on:9.3f}  ({rule})")
    realres = {c.name: real(a.out, c, cfgs, a.seconds) for c in a.real}
    for name, r in realres.items():
        ok = r["still_pulsing_on"] <= r["still_pulsing_off"] * 1.02 and (
            r["still_worst_on"] <= r["still_worst_off"] * 1.05
        )
        rows.append(
            (
                f"real {name} pulsing",
                r["still_pulsing_off"],
                r["still_pulsing_on"],
                ok,
                "on <= off x 1.02, worst x 1.05",
            )
        )
        print(
            f"{'PASS' if ok else 'FAIL'}  real {name} still-area pulsing "
            f"{r['still_pulsing_off']:.3f} -> {r['still_pulsing_on']:.3f}"
        )
    verdict = all(r[3] for r in rows)
    (a.out / "gate.json").write_text(
        json.dumps(
            {
                "set": over,
                "synthetic": m,
                "real": realres,
                "pass": verdict,
                "checks": [
                    {"metric": r[0], "off": r[1], "on": r[2], "pass": r[3], "rule": r[4]}
                    for r in rows
                ],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("GATE", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
