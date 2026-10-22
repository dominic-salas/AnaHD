"""Single frames as PNG bytes, grabbed with FFmpeg in a background thread: clip
thumbnails, and the before/after pair of the preview."""

from __future__ import annotations

import queue
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from anahd.errors import AnaHDError
from anahd.media import NO_WINDOW, Crop, probe, require
from anahd.origin import judge
from anahd.picture import output_size
from anahd.pipeline import Options, measure

THUMB_HEIGHT = 126


def grab(path: Path, seconds: float, vf: str) -> bytes | None:
    cmd = [require("ffmpeg"), "-v", "error", "-ss", f"{max(seconds, 0):.3f}", "-i", str(path)]
    cmd += ["-frames:v", "1", "-vf", vf, "-f", "image2pipe", "-c:v", "png", "-"]
    run = subprocess.run(cmd, capture_output=True, check=False, creationflags=NO_WINDOW)
    return run.stdout if run.returncode == 0 and run.stdout else None


@dataclass(frozen=True)
class Probed:
    path: Path
    duration: float
    details: str  # length, size and rate, upper case
    fps: float = 60.0
    size: tuple[int, int] = (0, 0)  # width, height of the recording as shown (pixel shape)


@dataclass(frozen=True)
class Still:
    path: Path
    seconds: float
    png: bytes | None


@dataclass(frozen=True)
class Pair:
    path: Path
    seconds: float  # in the output
    before: bytes | None  # the picture area, plain Lanczos to the output size
    after: bytes | None  # the cleaned output, None until there is one


class FrameReader(QThread):
    """Probes clips (with a thumbnail), checks whether they look digital, and grabs
    preview stills and before/after pairs, one job at a time. Probes come first, then
    what the user is looking at (only the latest still and pair), then the checks, which
    measure the whole clip and take seconds each."""

    probed = Signal(object, bytes)  # Probed, thumbnail PNG (may be empty)
    checked = Signal(object, object)  # path, Origin
    failed = Signal(object, str)  # path, why
    pair = Signal(object)  # Pair
    still = Signal(object)  # Still

    PRIORITY = ("probe", "still", "pair", "check")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: queue.Queue[tuple[str, tuple[object, ...]] | None] = queue.Queue()
        self._crops: dict[Path, Crop] = {}

    def set_crop(self, path: Path, crop: Crop) -> None:
        self._crops[path] = crop  # from the worker, so no second measurement

    def probe(self, path: Path) -> None:
        """Length, size and a thumbnail; then (later) whether it looks digital."""
        self._jobs.put(("probe", (path,)))
        self._jobs.put(("check", (path,)))

    def preview(
        self, path: Path, output: Path | None, out_seconds: float, src_seconds: float
    ) -> None:
        """Before (the recording at ``src_seconds``) and after (the output at
        ``out_seconds``): the same moment, which differ once parts are cut out."""
        self._jobs.put(("pair", (path, output, out_seconds, src_seconds)))

    def frame_at(self, path: Path, seconds: float) -> None:
        """A preview-size still of the recording (scrubbing on the clips page)."""
        self._jobs.put(("still", (path, seconds)))

    def stop(self) -> None:
        self._jobs.put(None)
        self.wait()

    def run(self) -> None:
        waiting: dict[str, list[tuple[object, ...]]] = {k: [] for k in self.PRIORITY}
        while True:
            # Take everything queued; a newer still or pair replaces an older one.
            block = not any(waiting.values())
            while True:
                try:
                    job = self._jobs.get(block=block)
                except queue.Empty:
                    break
                block = False
                if job is None:
                    return
                kind, args = job
                if kind in ("still", "pair"):
                    waiting[kind] = [args]
                else:
                    waiting[kind].append(args)
            kind = next(k for k in self.PRIORITY if waiting[k])
            args = waiting[kind].pop(0)
            try:
                if kind == "probe":
                    self._probe(*args)  # type: ignore[arg-type]
                elif kind == "check":
                    self._check(*args)  # type: ignore[arg-type]
                elif kind == "still":
                    path, seconds = args
                    png = grab(path, seconds, "scale=-2:540")  # type: ignore[arg-type]
                    self.still.emit(Still(path, seconds, png))  # type: ignore[arg-type]
                else:
                    self._pair(*args)  # type: ignore[arg-type]
            except (AnaHDError, OSError) as exc:
                self.failed.emit(args[0], str(exc))

    def _check(self, path: Path) -> None:
        info = probe(path)
        meas = measure(path, info, Options())
        self._crops.setdefault(path, meas.crop)
        self.checked.emit(path, judge(info, meas))

    def _probe(self, path: Path) -> None:
        info = probe(path)
        rate = info.fps.split("/")
        fps = int(rate[0]) / max(int(rate[1]) if len(rate) > 1 else 1, 1)
        minutes, secs = divmod(round(info.duration), 60)
        details = f"{minutes}:{secs:02d}   {info.width}x{info.height}   {fps:.0f} fps"
        # The preview's shape is the one shown: stored width times the pixel shape (and
        # 4:3 for SD DVR sizes), as the cleaned output will have it.
        shown = output_size(
            Crop(0, 0, info.width, info.height), info.height, info.sar, (info.width, info.height)
        )
        thumb_w = round(THUMB_HEIGHT * shown[0] / shown[1] / 2) * 2
        thumb = grab(path, 0.3 * info.duration, f"scale={thumb_w}:{THUMB_HEIGHT}") or b""
        self.probed.emit(Probed(path, info.duration, details, fps, shown), thumb)

    def _pair(
        self, path: Path, output: Path | None, out_seconds: float, src_seconds: float
    ) -> None:
        info = probe(path)
        crop = self._crops.get(path)
        if crop is None:
            crop = measure(path, info, Options()).crop
            self._crops[path] = crop
        w, h = output_size(crop, sar=info.sar, frame=(info.width, info.height))
        before = grab(path, src_seconds, f"{crop.filter},scale={w}:{h}:flags=lanczos,setsar=1")
        after = grab(output, out_seconds, "setsar=1") if output and output.is_file() else None
        self.pair.emit(Pair(path, out_seconds, before, after))
