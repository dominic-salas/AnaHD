"""The 4 px line filter on the restorer's 2x output, the fallbacks of the accelerators (a
clean never fails because TensorRT or the compiler cannot run), and when they are used."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anahd.gpu import engine, ops  # noqa: E402
from anahd.gpu.pipeline import (  # noqa: E402
    FAST_COLD_SECONDS,
    FAST_MIN_SECONDS,
    GpuConfig,
    _with_speedups,
)
from anahd.media import VideoInfo  # noqa: E402
from anahd.pipeline import Options  # noqa: E402


def _gain(k: np.ndarray, cycles_per_px: float) -> float:
    n = np.arange(len(k)) - len(k) // 2
    return float(abs((k * np.exp(-2j * np.pi * cycles_per_px * n)).sum()))


def test_notch_removes_the_4px_period_and_keeps_what_a_picture_can_hold() -> None:
    k = ops.notch_kernel()
    assert _gain(k, 0.25) < 1e-3
    for f in (0.0, 0.05, 0.1, 0.15, 0.167):  # up to an analog picture's limit at 1440 lines
        assert _gain(k, f) == pytest.approx(1.0, abs=0.03)


def test_notch_filter_cleans_lines_and_leaves_an_edge() -> None:
    rows = torch.arange(160, dtype=torch.float32)
    lines = 6.0 * torch.sin(2 * torch.pi * rows / 4)[:, None].expand(160, 64)
    step = torch.where(rows < 80, 40.0, 200.0)[:, None].expand(160, 64)
    out = ops.vertical_notch(step + lines)
    assert float((out - step)[20:60].abs().max()) < 0.2  # the lines are gone
    # next to a 160-level edge the filter rings by under 3 % of it, and not at all away from it
    assert abs(float(out[90, 0]) - 200.0) < 5.0 and abs(float(out[70, 0]) - 40.0) < 5.0
    assert abs(float(out[120, 0]) - 200.0) < 0.5


def test_tensorrt_failure_means_the_pytorch_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object) -> None:
        raise RuntimeError("no TensorRT here")

    monkeypatch.setattr(engine, "Engine", boom)
    assert engine.build_engine(torch.nn.Identity(), torch.zeros(1), "k") is None


def test_compile_failure_runs_the_plain_function(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(fn: object, **_kw: object) -> object:
        def run(*_a: object) -> None:
            raise RuntimeError("Compiler: cl is not found")

        return run

    monkeypatch.setattr(torch, "compile", broken)
    f = engine.Compiled(lambda x: x + 1)
    assert f(1) == 2 and f.failed
    assert f(2) == 3  # and stays on the plain path


def test_accelerators_are_for_clips_that_earn_back_their_start_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("anahd.gpu.pipeline.compile_available", lambda: True)
    info = VideoInfo(1280, 720, "60/1", False, False, 0.0)
    cfg = GpuConfig(learned=Path("m.pt"))

    def decide(seconds: float, config: GpuConfig = cfg) -> GpuConfig:
        return _with_speedups(config, replace(info, duration=seconds), Options())

    assert decide(FAST_MIN_SECONDS - 1).fast is False
    assert decide(FAST_MIN_SECONDS + 1).fast is True and not decide(FAST_MIN_SECONDS + 1).fast_cold
    assert decide(FAST_COLD_SECONDS + 1).fast_cold
    assert decide(300, replace(cfg, fast=False)).fast is False  # an explicit choice stays
    assert decide(300, GpuConfig()).fast is None  # the classic path is left alone
