"""Clean one clip in a separate process, so the window stays responsive and a stop is a
clean kill (CUDA, FFmpeg pipes and all). The process runs the same code as ``anahd``.

The worker puts tuples on a queue. Ongoing (any number, only the newest of each kind
matters): ``("stage", text)`` (what it is doing before the first frame),
``("progress", done, total, fps, time)`` and ``("frame", before_jpg, after_jpg, n, total)``
(a live before/after picture of output frame ``n``). Then exactly one ending:
``("done", frames, seconds, (x, y, w, h), notes)`` or ``("failed", message)``.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

POLL_MS = 200
ONGOING = ("stage", "progress", "frame")
ENDINGS = ("done", "failed")


def cli_args(
    src: str,
    cuts: list[tuple[float, float]],
    parts: bool = False,
    learned: bool = True,
    remaster: dict[str, Any] | None = None,
) -> list[str]:
    """The ``anahd`` command line for a clip: the defaults plus its cut markers,
    --parts when each kept part is to be its own file, --no-learned for the classic
    (hand-made) cleaning, and the remaster settings (``RemasterParams.to_dict()``) as
    --look-* flags: exactly what the app's preview used."""
    from anahd.cli import remaster_args
    from anahd.remaster import RemasterParams

    args = [src, *remaster_args(RemasterParams.from_dict(remaster or {}))]
    if not learned:
        args.append("--no-learned")
    for a, b in cuts:
        args += ["--cut", f"{a:.4f}-{b:.4f}"]
    if parts and cuts:
        args.append("--parts")
    return args


def work(
    src: str,
    out: str,
    messages: Queue[tuple[Any, ...]],
    cuts: list[tuple[float, float]] | None = None,
    parts: bool = False,
    learned: bool = True,
    remaster: dict[str, Any] | None = None,
) -> None:
    """The worker process's main: never raises, always ends with "done" or "failed"."""
    messages.put(("stage", "Starting"))
    from anahd import bundle, cli
    from anahd.errors import AnaHDError

    bundle.setup()
    try:
        args = cli._parser().parse_args(cli_args(src, cuts or [], parts, learned, remaster))
        with cli.keep_awake():
            result, took = cli.render(
                Path(src),
                Path(out),
                args,
                lambda done, total, fps: messages.put(
                    ("progress", done, total, fps, time.monotonic())
                ),
                lambda text: messages.put(("stage", text)),
                lambda before, after, n, total: messages.put(("frame", before, after, n, total)),
            )
        c = result.crop
        messages.put(("done", result.frames, took, (c.x, c.y, c.width, c.height), result.notes))
    except AnaHDError as exc:
        messages.put(("failed", str(exc)))
    except BaseException as exc:
        messages.put(("failed", f"{type(exc).__name__}: {exc}"))


class Runner(QObject):
    """Runs one clip at a time in a worker process and relays what it says as signals."""

    progress = Signal(object, int, int, float)  # source, frames done, total (0 = unknown), time
    stage = Signal(object, str)  # source, what it is doing before the first frame
    frame = Signal(object, bytes, bytes, int, int)  # source, before, after (JPEG), n, total
    finished = Signal(object, object)  # source, done tuple (frames, seconds, crop, notes)
    failed = Signal(object, str)  # source, message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc: BaseProcess | None = None
        self._queue: Queue[tuple[Any, ...]] | None = None
        self._src: Path | None = None
        self._out: Path | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self) -> bool:
        return self._proc is not None

    def start(
        self,
        src: Path,
        out: Path,
        cuts: list[tuple[float, float]] | None = None,
        parts: bool = False,
        learned: bool = True,
        remaster: dict[str, Any] | None = None,
    ) -> None:
        if self._proc is not None:
            raise RuntimeError("a clip is already being cleaned")
        self._src, self._out = src, out
        self._queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=work,
            args=(str(src), str(out), self._queue, list(cuts or []), parts, learned, remaster),
            daemon=True,
        )
        self._proc.start()
        self._timer.start()

    def stop(self) -> None:
        """Kill the worker and remove its unfinished output. The recording is untouched."""
        proc, out = self._proc, self._out
        if proc is None:
            return
        self._end()
        proc.kill()
        proc.join(5)
        if out is not None:
            from anahd.cli import partial_output

            partial = partial_output(out)
            for _ in range(20):  # FFmpeg may hold the file for a moment after the kill
                try:
                    partial.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.25)

    def _end(self) -> None:
        self._timer.stop()
        self._proc = None
        self._queue = None

    def _poll(self) -> None:
        q, proc, src = self._queue, self._proc, self._src
        if q is None or proc is None or src is None:
            return
        newest: dict[str, tuple[Any, ...]] = {}
        while True:
            try:
                msg = q.get_nowait()
            except queue.Empty:
                break
            if msg[0] in ONGOING:
                newest[msg[0]] = msg
                continue
            if msg[0] not in ENDINGS:
                continue  # a kind this version does not know: never an ending
            self._end()  # what was still ongoing in this poll is dropped with the clip
            proc.join(10)
            if msg[0] == "done":
                self.finished.emit(src, msg[1:])
            else:
                self.failed.emit(src, str(msg[1]))
            return
        if (m := newest.get("stage")) is not None:
            self.stage.emit(src, str(m[1]))
        if (m := newest.get("progress")) is not None:
            self.progress.emit(src, int(m[1]), int(m[2] or 0), float(m[4]))
        if (m := newest.get("frame")) is not None:
            self.frame.emit(src, bytes(m[1]), bytes(m[2]), int(m[3]), int(m[4]))
        if not proc.is_alive():
            # Its last message may still be in the pipe: wait for it briefly.
            try:
                while (msg := q.get(timeout=1.0))[0] not in ENDINGS:
                    pass
            except queue.Empty:
                msg = ("failed", f"the worker stopped unexpectedly (exit code {proc.exitcode})")
            self._end()
            if msg[0] == "done":
                self.finished.emit(src, msg[1:])
            else:
                self.failed.emit(src, str(msg[1]))
