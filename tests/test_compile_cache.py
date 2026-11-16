"""The compile of a new picture size is paid once: a second process finds it in the cache.
Needs a CUDA GPU and Triton; the first run compiles for about a minute and a half."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import synthetic as syn  # noqa: E402
from anahd.gpu.engine import compile_available  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PREPARING = "Preparing the GPU for this picture size"


def _clean(src: Path, out: Path, cache: Path) -> tuple[float, str]:
    """Run the CLI; seconds from its start to the first frame being cleaned, and its stderr."""
    env = dict(
        os.environ,
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        ANAHD_COMPILE_MIN_SECONDS="1",  # this clip is short; the compile is wanted anyway
    )
    began = time.monotonic()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from anahd.cli import main; raise SystemExit(main())",
            str(src),
            "-o",
            str(out),
            "--force",
        ],
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stderr is not None
    seen: list[str] = []
    ready = 0.0
    for line in proc.stderr:
        seen.append(line)
        if not ready and "Cleaning..." in line:
            ready = time.monotonic() - began
    assert proc.wait() == 0, "".join(seen)[-2000:]
    return ready, "".join(seen)


def test_second_start_does_not_compile_again(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    if not compile_available():
        pytest.skip("no Triton")
    src = syn.write(tmp_path / "in.mp4", 70).path
    first, said_first = _clean(src, tmp_path / "a.mp4", tmp_path)
    second, said_second = _clean(src, tmp_path / "b.mp4", tmp_path)
    record_property("report", f"start-up {first:.0f} s first, {second:.0f} s second")
    assert PREPARING in said_first  # disclosed the first time
    assert PREPARING not in said_second  # and never again: it was found in the cache
    assert second < 45 and second < first / 2
