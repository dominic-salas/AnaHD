"""The recording's sound in the output, in sync with the picture.

A synthetic recording flashes white on known frames and clicks at exactly those moments,
with its sound starting 83 ms after its first frame as HDZero goggle DVRs record it. The
render path runs for real (seeking, cutting, joining, parts, muxing) with a stand-in for
the cleaning that passes the frames through unchanged, and the flashes and clicks of the
output are found independently and compared.
"""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from anahd import cli
from anahd.cuts import keep_spans
from anahd.media import AudioSpan, Encoder, add_audio, probe, read_raw, sound_clock
from anahd.pipeline import Result

FPS = 60
FRAMES = 600  # 10 s
W, H = 64, 48
RATE = 48000
SOUND_LATE = 0.083  # the sound's first sample comes this long after the first frame
FLASHES = (45, 130, 200, 333, 410, 577)  # frames that flash, and click at their moment


def _recording(tmp: Path, suffix: str) -> Path:
    frames = np.zeros((FRAMES, H * W * 3 // 2), np.uint8)
    frames[:, : H * W] = 16
    frames[:, H * W :] = 128
    for f in FLASHES:
        frames[f, : H * W] = 235
    sound = np.zeros(int(RATE * (FRAMES / FPS - SOUND_LATE)), np.int16)
    for f in FLASHES:
        at = round((f / FPS - SOUND_LATE) * RATE)
        sound[at : at + 240] = 20000  # a 5 ms click, starting on the flash
    wav = tmp / "sound.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(sound.tobytes())
    src = tmp / f"flight{suffix}"
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p"]
    cmd += ["-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-"]
    cmd += ["-itsoffset", str(SOUND_LATE), "-i", str(wav), "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-g", "30", "-c:a", "aac", "-b:a", "128k", "-y", str(src)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return src


def _flashes(video: Path) -> list[float]:
    """Seconds into ``video`` of each frame that flashes."""
    info = probe(video)
    luma = W * H
    times = []
    for i, raw in enumerate(read_raw(video, info)):
        if np.asarray(raw[:luma]).mean() > 128:
            times.append(i / FPS)
    return times


def _clicks(video: Path) -> list[float]:
    """Seconds into ``video`` (on the picture's clock) where each click starts."""
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    sound = np.abs(np.frombuffer(pcm, np.int16).astype(np.int32))
    loud = np.flatnonzero(sound > 8000)
    if loud.size == 0:
        return []
    starts = loud[np.concatenate([[True], np.diff(loud) > RATE // 100])]
    # Where the sound starts on the picture's clock (both streams of the output start at 0
    # unless the muxer delays one; ask rather than assume).
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    start = {k: float(v) for k, v in (line.split(",") for line in out.split())}
    return [start["audio"] - start["video"] + s / RATE for s in starts]


def _assert_in_sync(video: Path, expected: int) -> None:
    flashes, clicks = _flashes(video), _clicks(video)
    assert len(flashes) == expected
    assert len(clicks) == expected
    for f, c in zip(flashes, clicks, strict=True):
        # AAC smears a click's onset by a fraction of a millisecond; a frame is 16.7 ms.
        assert abs(c - f) < 0.002, (f, c)


def _passthrough(
    src: Path, tmp: Path, opts: object, args: object, progress: object, measured: object = None
) -> Result:
    """Stands in for the cleaning: the same reading of the recording (seek, frame count),
    written unchanged."""
    info = probe(src)
    encoder = Encoder(tmp, info.width, info.height, info, overwrite=True)
    for raw in read_raw(src, info, start=opts.start, duration=opts.duration):  # type: ignore[attr-defined]
        encoder.write_raw(raw)
    encoder.close()
    return Result(encoder.frames, None, (info.width, info.height), 0.0, 1.0, 0.0)  # type: ignore[arg-type]


@pytest.fixture
def fake_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_clean_with_one_retry", _passthrough)
    monkeypatch.setattr(cli, "measure", lambda *a, **k: None)
    monkeypatch.setattr(cli, "verify", lambda *a, **k: None)


def _args(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = {"cut": None, "parts": False, "start": None, "duration": None}
    base["force"] = True
    base.update(kw)
    return argparse.Namespace(steady=False, **base)


@pytest.fixture(autouse=True)
def _no_remaster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "remaster_params", lambda args: None)


@pytest.mark.parametrize("suffix", [".mp4", ".ts"])
def test_whole_recording_keeps_its_sound_in_sync(
    tmp_path: Path, fake_clean: None, suffix: str
) -> None:
    src = _recording(tmp_path, suffix)
    out = tmp_path / "AnaHD_flight.mp4"
    cli.render(src, out, _args())
    _assert_in_sync(out, len(FLASHES))
    # the sound lasts as long as the picture, from its first frame
    lengths = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=duration", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert abs(float(lengths[0]) - float(lengths[1])) < 0.03


@pytest.mark.parametrize("suffix", [".mp4", ".ts"])
def test_cuts_joined_into_one_file_stay_in_sync(
    tmp_path: Path, fake_clean: None, suffix: str
) -> None:
    src = _recording(tmp_path, suffix)
    out = tmp_path / "AnaHD_flight.mp4"
    cuts = [(0.0, 1.0), (2.5, 3.1), (6.0, 7.5)]  # cut away the flashes at 200 and 410
    cli.render(src, out, _args(cut=cuts))
    kept = keep_spans(cuts, FRAMES, FPS)
    expected = sum(any(s.start <= f < s.end for s in kept) for f in FLASHES)
    assert expected == 4
    _assert_in_sync(out, expected)


@pytest.mark.parametrize("suffix", [".mp4", ".ts"])
def test_each_part_has_its_own_sound_in_sync(tmp_path: Path, fake_clean: None, suffix: str) -> None:
    src = _recording(tmp_path, suffix)
    out = tmp_path / "AnaHD_flight.mp4"
    cli.render(src, out, _args(cut=[(4.0, 5.0)], parts=True))
    one, two = tmp_path / "AnaHD_flight_part1.mp4", tmp_path / "AnaHD_flight_part2.mp4"
    _assert_in_sync(one, 3)  # flashes at 45, 130, 200
    _assert_in_sync(two, 3)  # 333, 410, 577


def test_a_start_inside_the_late_sound_is_padded_with_silence(tmp_path: Path) -> None:
    """A span from the very first frame: the 83 ms before the sound starts are silent, so
    nothing after slides early."""
    src = _recording(tmp_path, ".mp4")
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-an", "-c:v", "copy", "-y", str(video)],
        check=True,
    )
    out = tmp_path / "with_sound.mp4"
    clock = sound_clock(src)
    assert clock is not None
    add_audio(video, src, [AudioSpan(clock.video, FRAMES / FPS)], out)
    _assert_in_sync(out, len(FLASHES))


def test_no_sound_stays_no_sound(tmp_path: Path, fake_clean: None) -> None:
    src = _recording(tmp_path, ".mp4")
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-an", "-c:v", "copy", "-y", str(silent)],
        check=True,
    )
    out = tmp_path / "AnaHD_silent.mp4"
    cli.render(silent, out, _args())
    streams = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert streams == ["video"]
