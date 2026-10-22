"""Before/after stills for the README: the same frame as a plain Lanczos upscale of the
recording's picture area (what an editor would do) and as AnaHD's output.

python tools/readme_stills.py SRC RENDER OUT_DIR --frames 1200 4800 9000 [--crop x y w h]

Writes per frame ``<n>_full.png`` (both whole frames, side by side, reduced) and
``<n>_detail.png`` (a 1:1 crop of each, side by side). PNGs are reduced to 256 colours with
FFmpeg's palette filters to stay small.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from anahd.media import probe
from anahd.pipeline import Options, measure


def grab(path: Path, n: int, fps: float, vf: str, out: Path) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{n / fps:.4f}", "-i", str(path)]
    cmd += ["-frames:v", "1", "-vf", vf, str(out)]
    subprocess.run(cmd, check=True)


def pair(left: Path, right: Path, each: str, out: Path) -> None:
    """``each`` applied to both, side by side, then a per-image palette (small PNG)."""
    graph = (
        f"[0]{each}[l];[1]{each}[r];[l][r]hstack,split[a][b];"
        "[a]palettegen=max_colors=256:stats_mode=single[p];[b][p]paletteuse=dither=sierra2_4a"
    )
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(left), "-i", str(right)]
    subprocess.run([*cmd, "-filter_complex", graph, "-frames:v", "1", str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("render", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--detail", type=int, nargs=4, default=(360, 270, 720, 540), metavar="XYWH")
    a = ap.parse_args()
    info = probe(a.src)
    num, den = (int(x) for x in info.fps.split("/"))
    fps = num / den
    meas = measure(a.src, info, Options())
    w, h = meas.size
    plain = f"{meas.crop.filter},scale={w}:{h}:flags=lanczos,setsar=1"
    a.out.mkdir(parents=True, exist_ok=True)
    x, y, cw, ch = a.detail
    for n in a.frames:
        before, after = a.out / f"_{n}_before.png", a.out / f"_{n}_after.png"
        grab(a.src, n, fps, plain, before)
        grab(a.render, n, fps, "setsar=1", after)
        pair(before, after, f"scale={w // 2}:-2:flags=area", a.out / f"{n}_full.png")
        pair(before, after, f"crop={cw}:{ch}:{x}:{y}", a.out / f"{n}_detail.png")
        before.unlink()
        after.unlink()


if __name__ == "__main__":
    main()
