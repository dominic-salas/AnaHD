"""The shipped command: defaults, output safety, model lookup, and no training code imported."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from anahd import cli
from anahd.errors import AnaHDError
from anahd.model import weights


def test_ship_config_is_the_default() -> None:
    args = cli._parser().parse_args(["flight.ts"])
    assert args.learned and args.repair and not args.cpu and not args.ai
    assert args.model is None and args.learned_strength == 1.0
    off = cli._parser().parse_args(["flight.ts", "--no-learned", "--no-repair"])
    assert not off.learned and not off.repair
    assert cli._parser().parse_args(["a.ts", "--gpu"]).learned  # old flag still accepted


def test_output_goes_next_to_the_recording() -> None:
    src = Path("D:/clips/flight.ts")
    assert cli.default_output(src) == Path("D:/clips/AnaHD_flight.mp4")
    assert cli.partial_output(cli.default_output(src)).name == "AnaHD_flight.partial.mp4"
    assert cli.default_output(src, Path("E:/out")) == Path("E:/out/AnaHD_flight.mp4")


def test_never_overwrites_without_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "flight.ts"
    src.write_bytes(b"recording")
    out = cli.default_output(src)
    out.write_bytes(b"earlier render")
    assert cli.main([str(src)]) == 1
    assert "pass --force" in capsys.readouterr().err
    assert out.read_bytes() == b"earlier render"
    assert src.read_bytes() == b"recording"


def test_never_writes_over_the_recording(tmp_path: Path) -> None:
    src = tmp_path / "flight.mp4"
    src.write_bytes(b"recording")
    assert cli.main([str(src), "-o", str(src), "--force"]) == 1
    assert src.read_bytes() == b"recording"


def test_one_output_name_for_one_input() -> None:
    with pytest.raises(SystemExit):
        cli.main(["a.ts", "b.ts", "-o", "x.mp4"])


def test_model_lookup_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv(weights.ENV, raising=False)
    cached = tmp_path / "AnaHD" / "models" / weights.MODEL_NAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"weights")
    monkeypatch.setattr(weights, "MODEL_SHA256", weights.sha256(cached))
    assert weights.resolve() == cached
    env = tmp_path / "env.pt"
    env.write_bytes(b"x")
    monkeypatch.setenv(weights.ENV, str(env))
    assert weights.resolve() == env
    given = tmp_path / "given.pt"
    given.write_bytes(b"y")
    assert weights.resolve(given) == given  # --model wins over the environment
    with pytest.raises(AnaHDError):
        weights.resolve(tmp_path / "missing.pt")


def test_damaged_cache_is_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv(weights.ENV, raising=False)
    cached = weights.models_dir() / weights.MODEL_NAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"truncated")
    fetched: list[Path] = []

    def fetch(dest: Path) -> Path:
        fetched.append(dest)
        return dest

    monkeypatch.setattr(weights, "_fetch", fetch)
    assert weights.resolve() == cached
    assert fetched == [cached] and not cached.exists()
    with pytest.raises(AnaHDError, match="--model"):
        weights.resolve(download=False)


def test_product_does_not_import_training_code() -> None:
    mods = ["anahd.cli", "anahd.model.weights"]
    try:
        import torch  # noqa: F401

        mods += ["anahd.gpu.pipeline", "anahd.model.infer"]
    except ImportError:
        pass
    code = (
        f"import sys; import {', '.join(mods)}; "
        "bad = sorted(m for m in sys.modules if m.startswith('anahd.learn')); "
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr


def _tiny_clip(path: Path) -> Path:
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=60"]
    subprocess.run([*cmd, "-frames:v", "12", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _flaky_enhance(failures: list[str]) -> object:
    """A stand-in for the pipeline that fails with each message in turn, then succeeds by
    copying the source (same frame count, so the real checks downstream pass)."""
    from anahd.errors import MediaError
    from anahd.media import Crop
    from anahd.pipeline import Result

    calls: list[int] = []

    def enhance(src: Path, tmp: Path, *_: object, **__: object) -> Result:
        calls.append(1)
        if len(calls) <= len(failures):
            tmp.write_bytes(b"half a video")
            raise MediaError(failures[len(calls) - 1])
        shutil.copyfile(src, tmp)
        return Result(12, Crop(0, 0, 320, 240), (320, 240), 0.0)

    enhance.calls = calls  # type: ignore[attr-defined]
    return enhance


OOM_X264 = (
    "ffmpeg failed while encoding: [vost#0:0/libx264 @ 0000020a4215fe80] Terminating "
    "thread with return code -542398533 (Generic error in an external library)"
)
OOM_DEMUX = (
    "ffmpeg stopped while encoding: [in#0/rawvideo @ 0000023f1c4b3500] Error during "
    "demuxing: Cannot allocate memory"
)


@pytest.mark.parametrize("message", [OOM_X264, OOM_DEMUX])
def test_out_of_memory_in_ffmpeg_is_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    """Both messages were seen on this PC with Windows' commit charge nearly used up."""
    src = _tiny_clip(tmp_path / "flight.mp4")
    fake = _flaky_enhance([message])
    monkeypatch.setattr(cli, "enhance", fake)
    assert cli.main([str(src), "--cpu"]) == 0
    assert len(fake.calls) == 2  # type: ignore[attr-defined]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AnaHD_flight.mp4", "flight.mp4"]


def test_out_of_memory_twice_fails_clearly_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _tiny_clip(tmp_path / "flight.mp4")
    fake = _flaky_enhance([OOM_DEMUX, OOM_X264])
    monkeypatch.setattr(cli, "enhance", fake)
    assert cli.main([str(src), "--cpu"]) == 1
    assert len(fake.calls) == 2  # type: ignore[attr-defined]  # once more, not forever
    assert "ran out of memory twice" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["flight.mp4"]


def test_other_ffmpeg_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _tiny_clip(tmp_path / "flight.mp4")
    fake = _flaky_enhance(["ffmpeg failed while encoding: Invalid data found"])
    monkeypatch.setattr(cli, "enhance", fake)
    assert cli.main([str(src), "--cpu"]) == 1
    assert len(fake.calls) == 1  # type: ignore[attr-defined]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["flight.mp4"]


def test_version_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    from anahd import __version__

    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"AnaHD {__version__}"
