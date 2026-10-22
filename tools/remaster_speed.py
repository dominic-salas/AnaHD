"""Measure the render speed of each remaster effect, for the app's speed estimate.

    uv run python tools/remaster_speed.py > speed.json

Renders a 10 s synthetic 1280x720 DVR recording (tests/synthetic.py) with each effect on
its own, all of them together, and the look off, for the Learned and Classic cleaning.
Per case: the steady speed (frames 120 on, from the progress reports) and the wall time;
``anahd.app.speed`` is built from these numbers.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import synthetic as syn
from anahd import cli
from anahd.remaster import NO_LOOK, RemasterParams

FRAMES = 600
EFFECTS = {
    "motion_blur": {"motion_blur": 39.0},
    "stabilisation": {"stabilisation": 50.0},
    "superview": {"superview": "stretch"},
    "sharpen": {"sharpen": 50.0},
}


def run(src: Path, params: RemasterParams, learned: bool) -> dict[str, float]:
    out = src.with_name(f"AnaHD_{src.stem}.mp4")
    out.unlink(missing_ok=True)
    argv = [str(src), *cli.remaster_args(params)] + ([] if learned else ["--no-learned"])
    args = cli._parser().parse_args(argv)
    reports: list[tuple[float, int]] = []
    began = time.monotonic()
    cli.render(src, out, args, lambda d, _t, _r: reports.append((time.monotonic(), d)))
    wall = time.monotonic() - began
    steady = [(t, n) for t, n in reports if n >= 120]
    (t0, n0), (t1, n1) = steady[0], steady[-1]
    out.unlink(missing_ok=True)
    return {"steady_fps": (n1 - n0) / (t1 - t0), "wall_s": wall, "frames": FRAMES}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["learned", "classic", "both"], default="both")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case (the fastest is kept)")
    a = ap.parse_args()
    engines = {"learned": (True,), "classic": (False,), "both": (True, False)}[a.engine]
    results: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        src = syn.write(Path(tmp) / "speed.mp4", FRAMES).path
        for learned in engines:
            engine = "learned" if learned else "classic"
            cases = {
                "default": RemasterParams(),
                "no_look": NO_LOOK,
                **{name: RemasterParams.from_dict(ch) for name, ch in EFFECTS.items()},
                "all_effects": RemasterParams.from_dict(
                    {k: v for ch in EFFECTS.values() for k, v in ch.items()}
                ),
            }
            for name, params in cases.items():
                runs = [run(src, params, learned) for _ in range(a.repeat)]
                results[f"{engine}/{name}"] = max(runs, key=lambda r: r["steady_fps"])
                print(f"{engine}/{name}: {results[f'{engine}/{name}']}", file=sys.stderr)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
