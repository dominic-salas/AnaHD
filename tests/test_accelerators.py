"""The 4 px line filter on the restorer's 2x output, the fallbacks of the accelerators (a
clean never fails because TensorRT or the compiler cannot run), and when they are used."""

from __future__ import annotations

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


def _decide(seconds: float, config: GpuConfig | None = None) -> GpuConfig:
    info = VideoInfo(1280, 720, "60/1", False, False, seconds)
    return _with_speedups(config or GpuConfig(learned=Path("m.pt")), info, Options())


def test_accelerators_are_for_clips_that_earn_back_their_start_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("ANAHD_COMPILE", raising=False)
    assert _decide(FAST_MIN_SECONDS - 1).fast is False
    assert _decide(FAST_MIN_SECONDS + 1).fast is True
    assert not _decide(FAST_COLD_SECONDS + 1).fast_cold  # compiling is off by default
    assert _decide(300, GpuConfig(learned=Path("m.pt"), fast=False)).fast is False
    assert _decide(300, GpuConfig()).fast is None  # the classic path is left alone


def test_compiling_runs_only_when_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pytest.importorskip("triton")
    monkeypatch.delenv("ANAHD_COMPILE", raising=False)
    monkeypatch.delenv("ANAHD_COMPILE_MIN_SECONDS", raising=False)
    assert not engine.compile_available()
    monkeypatch.setenv("ANAHD_COMPILE", "1")
    assert engine.compile_available()
    assert not _decide(FAST_COLD_SECONDS - 1).fast_cold
    assert _decide(FAST_COLD_SECONDS + 1).fast_cold


def test_the_tensorrt_notice_is_for_a_build_and_not_for_a_cached_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from anahd.gpu import trt as finder

    heard: list[str] = []
    engine.set_build_notice(heard.append)

    class Logger:
        ERROR = 0

        def __init__(self, *_a: object) -> None: ...

    fake = type("FakeTensorRT", (), {"Logger": Logger})  # has no Builder: a build fails
    monkeypatch.setattr(finder, "module", lambda: fake)
    monkeypatch.setattr(finder, "engine_cache_dir", lambda _gpu=None: tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d=None: "gpu")
    try:
        with pytest.raises(Exception):  # noqa: B017 - the fake TensorRT cannot build
            engine.Engine(torch.nn.Identity(), torch.zeros(1), "k")
        assert heard == [engine.BUILD_NOTICE]
        (tmp_path / "j.engine").write_bytes(b"x")
        heard.clear()
        with pytest.raises(Exception):  # noqa: B017 - nor load one
            engine.Engine(torch.nn.Identity(), torch.zeros(1), "j")
        assert heard == []
    finally:
        engine.set_build_notice(None)


def test_the_marker_is_for_this_gpu_and_software_and_not_written_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "compile_stamp", lambda: "aaaa")
    assert not engine.compile_is_warm("raft_1x1")

    class Broken:
        failed = True

        def __call__(self) -> None: ...

    engine.mark_compiled("raft_1x1", Broken())
    assert not engine.compile_is_warm("raft_1x1")  # a failed compile leaves no marker

    class Fine:
        failed = False

        def __call__(self) -> None: ...

    engine.mark_compiled("raft_1x1", Fine())
    assert engine.compile_is_warm("raft_1x1")
    monkeypatch.setattr(engine, "compile_stamp", lambda: "bbbb")  # new driver, torch, AnaHD...
    assert not engine.compile_is_warm("raft_1x1")


def test_the_stamp_is_the_same_every_time() -> None:
    assert engine.compile_stamp() == engine.compile_stamp()
    assert len(engine.compile_stamp()) == 12
