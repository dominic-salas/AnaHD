"""Side-by-sides of the learned restorer for review (lab venv, GPU for the renders):

    PYTHONPATH=src python tools/make_a5.py CKPT LEFT_DIR OUT [--clips a b ...] [--slow a b c]

Left = the previous review (LEFT_DIR/<clip>_repair.mp4: --gpu --ai --repair), right = the same
clip through --gpu --repair --learned CKPT (rendered into OUT/work unless already there).
Writes OUT/<clip>_sbs.mp4 (60 fps), OUT/<clip>_slowmo.mp4 (0.25x, 3 s around the clip's
worst breakup, for the --slow clips) and OUT/stills.png (2x crops, left | right) to judge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

A0 = Path("D:/FPVEdit/res/anahd_a0/dev/source")
FONT = "C\\:/Windows/Fonts/arial.ttf"
W, H = 1440, 1080


def render(ckpt: Path, clip: str, dst: Path, strength: float) -> None:
    if dst.exists():
        return
    cmd = [sys.executable, "-m", "anahd.cli", str(A0 / f"{clip}.mp4"), "--gpu", "--repair"]
    cmd += ["--learned", str(ckpt), "--learned-strength", str(strength), "-o", str(dst)]
    subprocess.run(cmd, check=True)


def label(text: str, x: str, y: str = "16") -> str:
    return (
        f"drawtext=fontfile='{FONT}':text='{text}':x={x}:y={y}:fontsize=34:fontcolor=white:"
        "box=1:boxcolor=black@0.6:boxborderw=8"
    )


def sbs(a: Path, b: Path, dst: Path, extra: str = "") -> None:
    graph = (
        f"[0][1]hstack,{label('current (--gpu --ai --repair)', '20')},"
        f"{label('learned restorer', 'w/2+20')}{extra}"
    )
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex", graph]
    cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd += ["-r", "60", "-an", str(dst)]
    subprocess.run(cmd, check=True)


def frame(path: Path, n: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n})"]
    cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def stills(pairs: list[tuple[str, Path, Path, int]], dst: Path) -> None:
    """Per pick: a 480x360 crop from the middle of each side, shown at 2x, left | right."""
    import cv2

    rows = []
    for clip, a, b, n in pairs:
        fa, fb = frame(a, n), frame(b, n)
        y0, x0 = (H - 360) // 2, (W - 480) // 2
        ca, cb = fa[y0 : y0 + 360, x0 : x0 + 480], fb[y0 : y0 + 360, x0 : x0 + 480]
        row = np.hstack([ca, np.full((360, 6, 3), 255, np.uint8), cb])
        row = cv2.resize(row, (row.shape[1] * 2, 720), interpolation=cv2.INTER_NEAREST)
        cv2.putText(row, f"{clip} {n}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        rows.append(row)
    cv2.imwrite(str(dst), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("left", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clips", nargs="+", default=[p.stem for p in sorted(A0.glob("*.mp4"))])
    ap.add_argument("--slow", nargs="*", default=["day_sunny", "night_dark", "urban"])
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--stills-only", action="store_true")
    a = ap.parse_args()
    work = a.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    picks = json.loads((a.left.parent / "picks.json").read_text())
    pairs = []
    for clip in a.clips:
        left, right = a.left / f"{clip}_repair.mp4", work / f"{clip}_learned.mp4"
        render(a.ckpt, clip, right, a.strength)
        pairs += [(clip, left, right, n) for n in (300, picks[clip]["biggest_change_frames"][0])]
        if a.stills_only:
            continue
        sbs(left, right, a.out / f"{clip}_sbs.mp4")
        if clip in a.slow:
            start = picks[clip]["burst_seconds"][0]
            trim = f",trim=start={start:.3f}:duration=3,setpts=4*(PTS-STARTPTS)"
            sbs(
                left,
                right,
                a.out / f"{clip}_slowmo.mp4",
                trim + "," + label("0.25x slow motion", "(w-tw)/2", "h-th-24"),
            )
    stills(pairs, a.out / "stills.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
