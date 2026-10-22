"""FFmpeg in and out: probe a recording, stream its frames as YUV planes, encode the result.

Frames travel as raw 8-bit 4:2:0 planes (Y at full size, U and V at half size), the
format goggle DVRs record in, so no colour conversion happens on the way in. The DVR's
range (full or limited) is kept and only converted once, to limited-range BT.709, when
the result is encoded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from anahd.errors import MediaError

# FFmpeg runs without a console window of its own (the desktop app has none to share).
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW

Plane = NDArray[np.uint8]

# Same encoder settings as the A0 baselines, so blind comparisons differ only in the picture.
X264_PRESET = "slow"
X264_CRF = "12"


@dataclass(frozen=True)
class Frame:
    y: Plane
    u: Plane
    v: Plane


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: str  # exact rational, e.g. "60/1"
    full_range: bool
    bt709: bool  # source matrix; DVRs of analog video use BT.601
    duration: float
    sar: float = 1.0  # pixel width over height (SD DVRs store 4:3 in 720x480 or 720x576)


@dataclass(frozen=True)
class Crop:
    """The picture area. ``width`` x ``height`` is the size frames are delivered at; when
    ``src_width`` is set, the area is ``src_width`` x ``src_height`` in the file and is
    scaled down to the delivered size (inputs taller than MAX_PICTURE_H)."""

    x: int
    y: int
    width: int
    height: int
    src_width: int = 0
    src_height: int = 0

    @property
    def filter(self) -> str:
        if self.src_width:
            return (
                f"crop={self.src_width}:{self.src_height}:{self.x}:{self.y},"
                f"scale={self.width}:{self.height}:flags=area"
            )
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"

    def scaled_to(self, max_height: int) -> Crop:
        """This area delivered at most ``max_height`` lines tall (even sizes, aspect kept)."""
        if self.height <= max_height:
            return self
        h = max_height - max_height % 2
        w = round(self.width * h / self.height / 2) * 2
        return Crop(self.x, self.y, w, h, self.width, self.height)


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise MediaError(f"{tool} not found on PATH; install FFmpeg 6 or newer")
    return path


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [
            require("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,pix_fmt,color_range,color_space,sample_aspect_ratio:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        creationflags=NO_WINDOW,
    )
    if out.returncode != 0:
        raise MediaError(f"ffprobe could not read {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"{path} has no video stream")
    stream = streams[0]
    # Any format FFmpeg decodes is fine: frames are always read as 8-bit 4:2:0 (older
    # analog DVRs record MJPEG in 4:2:2, for one). yuvj* is full range.
    pix_fmt = str(stream.get("pix_fmt", ""))
    full = pix_fmt.startswith("yuvj") or stream.get("color_range") == "pc"
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=str(stream["r_frame_rate"]),
        full_range=full,
        bt709=stream.get("color_space") == "bt709",
        duration=float(data.get("format", {}).get("duration", 0.0)),
        sar=_ratio(str(stream.get("sample_aspect_ratio", "1:1"))),
    )


def _ratio(text: str) -> float:
    """ffprobe's "8:9" as 0.888...; "0:1", "N/A" or nonsense as 1 (square pixels)."""
    num, _, den = text.partition(":")
    try:
        value = int(num) / int(den)
    except (ValueError, ZeroDivisionError):
        return 1.0
    return value if 0.25 <= value <= 4 else 1.0


def _input_args(path: Path, start: float | None, duration: float | None) -> list[str]:
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    return args


def read_frames(
    path: Path,
    info: VideoInfo,
    crop: Crop | None = None,
    *,
    start: float | None = None,
    duration: float | None = None,
    every: float | None = None,
    keyframes: bool = False,
) -> Iterator[Frame]:
    """Decode frames as Y/U/V planes, optionally cropped, sampled every ``every`` seconds,
    or only the keyframes (fast: the other frames are not decoded at all)."""
    width, height = (crop.width, crop.height) if crop else (info.width, info.height)
    filters = [crop.filter] if crop else []
    if every is not None:
        filters.append(f"fps=1/{every}")
    pix_fmt = "yuvj420p" if info.full_range else "yuv420p"
    cmd = [require("ffmpeg"), "-nostdin", "-v", "error"]
    if keyframes:
        cmd += ["-skip_frame", "nokey"]
    cmd += [*_input_args(path, start, duration), "-map", "0:v:0", "-an"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-fps_mode", "passthrough", "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]
    yield from _stream(cmd, width, height)


def read_raw(
    path: Path,
    info: VideoInfo,
    crop: Crop | None = None,
    *,
    start: float | None = None,
    duration: float | None = None,
) -> Iterator[Plane]:
    """Decode frames as one flat buffer each: Y, then U, then V (as the DVR stores them)."""
    width, height = (crop.width, crop.height) if crop else (info.width, info.height)
    pix_fmt = "yuvj420p" if info.full_range else "yuv420p"
    cmd = [require("ffmpeg"), "-nostdin", "-v", "error", *_input_args(path, start, duration)]
    cmd += ["-map", "0:v:0", "-an"]
    if crop:
        cmd += ["-vf", crop.filter]
    cmd += ["-fps_mode", "passthrough", "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]
    yield from _stream_raw(cmd, width * height + 2 * ((width // 2) * (height // 2)))


def _stream(cmd: list[str], width: int, height: int) -> Iterator[Frame]:
    luma = width * height
    chroma = (width // 2) * (height // 2)
    for buf in _stream_raw(cmd, luma + 2 * chroma):
        yield Frame(
            y=buf[:luma].reshape(height, width),
            u=buf[luma : luma + chroma].reshape(height // 2, width // 2),
            v=buf[luma + chroma :].reshape(height // 2, width // 2),
        )


def _stream_raw(cmd: list[str], size: int) -> Iterator[Plane]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW
    )
    if proc.stdout is None:  # pragma: no cover - Popen with PIPE always sets it
        raise MediaError("could not read from ffmpeg")
    try:
        while True:
            raw = _read_exact(proc.stdout, size)
            if raw is None:
                break
            yield np.frombuffer(raw, dtype=np.uint8)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        raise MediaError(f"ffmpeg failed while decoding: {err.strip()}")


def _read_exact(stream: IO[bytes], size: int) -> bytearray | None:
    """Exactly ``size`` bytes (a writable buffer), or None at the end of the stream."""
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        chunk = stream.read(size - got)
        if not chunk:
            return None
        view[got : got + len(chunk)] = chunk
        got += len(chunk)
    return buf


# NVENC at its slowest, highest-quality settings; constant quality like x264's CRF.
NVENC_ARGS = (
    "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "16",
    "-b:v", "0", "-multipass", "fullres", "-spatial-aq", "1", "-temporal-aq", "1",
    "-bf", "3", "-b_ref_mode", "middle", "-rc-lookahead", "32",
)  # fmt: skip


_BT709_TV = (
    "-color_range",
    "tv",
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
)


class Encoder:
    """Pipe 4:2:0 frames into FFmpeg; write limited-range BT.709 H.264.

    By default frames arrive full range in the source's matrix (the CPU pipeline) and FFmpeg
    converts them; with ``ready`` they are already limited-range BT.709 (the GPU pipeline).
    ``nvenc`` encodes on the GPU instead of with x264.
    """

    def __init__(
        self,
        out: Path,
        width: int,
        height: int,
        info: VideoInfo,
        *,
        overwrite: bool = False,
        ready: bool = False,
        nvenc: bool = False,
    ) -> None:
        matrix = "bt709" if info.bt709 else "bt601"
        pix_in = "yuv420p" if ready else "yuvj420p"
        vf = (
            "setsar=1"
            if ready
            else f"scale=in_range=pc:out_range=tv:in_color_matrix={matrix}"
            ":out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,setsar=1,format=yuv420p"
        )
        codec = (
            list(NVENC_ARGS)
            if nvenc
            else ["-c:v", "libx264", "-preset", X264_PRESET, "-crf", X264_CRF]
        )
        cmd = [
            require("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_in,
            "-s",
            f"{width}x{height}",
            "-framerate",
            info.fps,
            # Say what the frames are. Untagged, FFmpeg 9 takes raw YUV for BT.601 and
            # converts it to the BT.709 the file declares: frames AnaHD already made BT.709
            # came out ~3.5 levels off in luma and ~2 in colour (found 2026-10-21 by the
            # preview-equals-render test). The CPU path's frames are converted by the
            # scale filter below, which names its input matrix itself.
            *(_BT709_TV if ready else ()),
            "-i",
            "-",
            "-vf",
            vf,
            *codec,
            "-profile:v",
            "high",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-movflags",
            "+faststart",
            "-an",
            "-y" if overwrite else "-n",
            str(out),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW
        )
        self.frames = 0

    def write(self, frame: Frame) -> None:
        stdin = self._stdin()
        try:
            stdin.write(frame.y.tobytes())
            stdin.write(frame.u.tobytes())
            stdin.write(frame.v.tobytes())
        except BrokenPipeError as exc:
            raise MediaError(f"ffmpeg stopped while encoding: {self._error()}") from exc
        self.frames += 1

    def write_raw(self, data: bytes | memoryview | Plane) -> None:
        """One frame as a flat Y, U, V buffer."""
        try:
            self._stdin().write(data)
        except BrokenPipeError as exc:
            raise MediaError(f"ffmpeg stopped while encoding: {self._error()}") from exc
        self.frames += 1

    def close(self) -> None:
        self._stdin().close()
        err = self._error()
        if self._proc.wait() != 0:
            raise MediaError(f"ffmpeg failed while encoding: {err}")

    def _stdin(self) -> IO[bytes]:
        if self._proc.stdin is None:  # pragma: no cover - Popen with PIPE always sets it
            raise MediaError("could not write to ffmpeg")
        return self._proc.stdin

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()

    def _error(self) -> str:
        return (
            self._proc.stderr.read().decode(errors="replace").strip() if self._proc.stderr else ""
        )


def count_frames(path: Path) -> int:
    """Video frames in ``path``, counted from its packets (one per frame; nothing is decoded)."""
    out = subprocess.run(
        [
            require("ffprobe"),
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        creationflags=NO_WINDOW,
    )
    if out.returncode != 0 or not out.stdout.strip().isdigit():
        raise MediaError(f"could not count frames of {path}: {out.stderr.strip()}")
    return int(out.stdout.strip())
