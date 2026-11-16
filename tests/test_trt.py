"""TensorRT discovery and engine cache: no GPU and no TensorRT needed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from anahd.gpu import trt


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("ANAHD_TENSORRT", str(tmp_path / "trt"))
    monkeypatch.setattr(trt, "_driver", lambda: "570.1")
    trt._probe.cache_clear()
    trt._warned.clear()
    yield
    trt._probe.cache_clear()


def test_not_available_without_tensorrt_gives_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # makes the import fail
    status = trt.status()
    assert not status.ok
    assert not trt.available()
    assert "not installed" in status.reason
    assert trt.module() is None


def test_broken_folder_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    (tmp_path / "trt" / "tensorrt_bindings").mkdir(parents=True)
    status = trt.status()
    assert not status.ok
    assert "did not load" in status.reason
    assert str(tmp_path / "trt") not in sys.path


def test_found_when_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("tensorrt")
    fake.__version__ = "10.15.1.29"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    assert trt.available()
    assert trt.status().version == "10.15.1.29"


def test_cache_key_names_gpu_driver_and_version() -> None:
    key = trt.cache_key("NVIDIA GeForce RTX 4070 Ti", "570.1", "10.15.1.29")
    assert key == "NVIDIA_GeForce_RTX_4070_Ti_driver570.1_trt10.15.1.29"
    assert trt.cache_key("A", "1", "2") != trt.cache_key("A", "2", "2")
    assert trt.cache_key("A", "1", "2") != trt.cache_key("B", "1", "2")
    assert trt.cache_key("A", "1", "2") != trt.cache_key("A", "1", "3")


def test_engine_cache_dir_is_under_local_app_data(tmp_path: Path) -> None:
    folder = trt.engine_cache_dir("RTX 4070 Ti")
    assert folder.parent == tmp_path / "local" / "AnaHD" / "engines"
    assert folder.name.startswith("RTX_4070_Ti_driver570.1_trt")


def test_get_engine_is_none_without_tensorrt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    built: list[int] = []

    def build(_t: ModuleType) -> bytes:
        built.append(1)
        return b"x"

    assert trt.get_engine("m", build, gpu="g") is None
    assert trt.get_engine("m", build, gpu="g") is None
    assert not built
    assert len([r for r in caplog.records if r.name == "anahd.trt"]) == 1  # logged once


def _fake_trt(monkeypatch: pytest.MonkeyPatch, engine: object | None) -> ModuleType:
    class Runtime:
        def __init__(self, _logger: object) -> None: ...

        def deserialize_cuda_engine(self, _data: bytes) -> object | None:
            return engine

    class Logger:
        ERROR = 0

        def __init__(self, _level: int) -> None: ...

    fake = ModuleType("tensorrt")
    fake.__version__ = "1.0"  # type: ignore[attr-defined]
    fake.Runtime = Runtime  # type: ignore[attr-defined]
    fake.Logger = Logger  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    return fake


def test_get_engine_builds_once_then_loads_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object()
    _fake_trt(monkeypatch, engine)
    builds: list[int] = []

    def build(_t: ModuleType) -> bytes:
        builds.append(1)
        return b"engine-bytes"

    assert trt.get_engine("model_960x720", build, gpu="G") is engine
    assert trt.get_engine("model_960x720", build, gpu="G") is engine
    assert builds == [1]
    assert (trt.engine_cache_dir("G") / "model_960x720.engine").read_bytes() == b"engine-bytes"


@pytest.mark.parametrize("failure", ["build raises", "build returns nothing", "bad cache"])
def test_get_engine_returns_none_on_any_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _fake_trt(monkeypatch, None if failure == "bad cache" else object())

    def build(_t: ModuleType) -> bytes | None:
        if failure == "build raises":
            raise RuntimeError("no kernels")
        return None if failure == "build returns nothing" else b"junk"

    assert trt.get_engine("m", build, gpu="G") is None
    if failure == "bad cache":  # the unreadable engine is removed so the next run rebuilds
        assert not (trt.engine_cache_dir("G") / "m.engine").exists()
