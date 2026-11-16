"""The digital (HDZero) path: no alignment, repair or analog cleaning, only the upscale and
the look; and the route a recording takes to it. Needs a CUDA GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import synthetic as syn  # noqa: E402
from anahd.gpu.pipeline import GpuConfig, enhance  # noqa: E402
from anahd.media import probe  # noqa: E402
from anahd.pipeline import Options  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _frames(path: Path) -> list[str]:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if not line.startswith("#")]


def test_a_sharp_720p60_clip_takes_the_digital_path_by_itself(tmp_path: Path) -> None:
    """Routed from the recording (digital=None), forced digital, and forced analog: the first
    two are the same bytes, the analog one is cleaned (breakup repair on) and differs."""
    clip = syn.write_with(
        tmp_path / "hdzero.mp4", 60, syn.without_osd(syn.WIDE, speed=10), syn.WIDE, lines=None
    )
    assert probe(clip.path).fps == "60/1"
    made = {}
    for name, digital in (("auto", None), ("digital", True), ("analog", False)):
        out = tmp_path / f"{name}.mp4"
        result = enhance(
            clip.path,
            out,
            Options(),
            gcfg=GpuConfig(repair=True, digital=digital),
            overwrite=True,
        )
        made[name] = (_frames(out), result)
    assert len(made["auto"][0]) == len(made["analog"][0]) == 60
    assert made["auto"][0] == made["digital"][0]
    assert made["auto"][0] != made["analog"][0]
    assert made["digital"][1].notes == ""  # nothing was repaired
    assert made["digital"][1].size == made["analog"][1].size  # the same 1440-line output
