"""``anahd flight.mp4``: write a cleaned HD 60 fps copy next to each recording.

By default on an NVIDIA GPU: breakup repair and the learned restorer (v6). The flags only
turn things off or point at another model. The recording itself is never changed.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from anahd import __version__, bundle
from anahd.cuts import keep_spans, parse_cut
from anahd.errors import AnaHDError, MediaError, OutputExistsError
from anahd.media import (
    NO_WINDOW,
    AudioSpan,
    add_audio,
    count_frames,
    first_frame_time,
    probe,
    require,
    sound_clock,
)
from anahd.pipeline import Measured, Options, Result, enhance, measure, verify
from anahd.remaster.params import NO_LOOK, RemasterParams, schema
from anahd.steady import SteadyConfig

MAX_UNREADABLE = 0.005  # share of source packets that may fail to decode


PREFIX = "AnaHD_"  # outputs are AnaHD_<recording>.mp4


def is_output(path: Path) -> bool:
    """An AnaHD output (current or the earlier <name>_anahd naming), not a recording."""
    return path.name.lower().startswith(PREFIX.lower()) or "_anahd" in path.stem.lower()


def default_output(src: Path, out_dir: Path | None = None) -> Path:
    return (out_dir or src.parent) / f"{PREFIX}{src.stem}.mp4"


def partial_output(out: Path) -> Path:
    """Where the render is written until it is complete and checked."""
    return out.with_name(f"{out.stem}.partial{out.suffix}")


@contextmanager
def keep_awake() -> Iterator[None]:
    """Keep Windows from sleeping while frames are rendered (the screen may still turn off).
    Elsewhere a no-op."""
    if sys.platform != "win32":
        yield
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        kernel.SetThreadExecutionState(es_continuous)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anahd", description=__doc__)
    p.add_argument("--version", action="version", version=f"AnaHD {__version__}")
    p.add_argument("input", type=Path, nargs="+", help="DVR recordings, cleaned one after another")
    p.add_argument("-o", "--output", type=Path, help="output file (one input only)")
    p.add_argument("--out-dir", type=Path, help="write outputs here instead of next to the input")
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    p.add_argument("--start", type=float, help="seconds into the recording to start")
    p.add_argument("--duration", type=float, help="seconds to process")
    p.add_argument(
        "--cut",
        type=parse_cut,
        action="append",
        metavar="START-END",
        help="leave this stretch out (seconds or m:ss, e.g. 0-4.5 or 2:10-2:31); repeatable",
    )
    p.add_argument(
        "--parts",
        action="store_true",
        help="with --cut: each kept part as its own file (AnaHD_<name>_part1.mp4, ...)",
    )
    p.add_argument("--model", type=Path, help="restorer weights (default: the shipped v6)")
    p.add_argument(
        "--no-learned",
        dest="learned",
        action="store_false",
        help="hand-made fusion, destripe and derainbow instead of the learned restorer",
    )
    p.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="leave breakup bands and lost colour as they are",
    )
    p.add_argument("--ai", action="store_true", help="with --no-learned: SPAN 2x upscale")
    p.add_argument(
        "--learned-strength",
        type=float,
        default=1.0,
        help="0 = repaired centre frame only, 1 = the restorer's full output",
    )
    p.add_argument("--nvenc", action="store_true", help="encode on the GPU (faster, softer)")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="clean on the CPU (slow, ~3 fps; hand-made steps, no repair, no restorer)",
    )
    p.add_argument("--steady", action="store_true", help="with --cpu: take out small jitter")
    p.add_argument("--gpu", action="store_true", help=argparse.SUPPRESS)  # the default now
    look = p.add_argument_group(
        "remaster settings (the app's sliders; defaults are the approved C03 look)"
    )
    look.add_argument(
        "--no-look",
        action="store_true",
        help="cleaning only: no colour clean-up, lens colour fix or grade (as before the look)",
    )
    for item in schema():
        flag = "--look-" + item["name"].replace("_", "-")
        if item["kind"] == "slider":
            unit = item["unit"]
            look.add_argument(
                flag,
                type=float,
                metavar="N",
                help=(
                    f"{item['label']}: {item['help']} "
                    f"({item['min']:g}..{item['max']:g}{unit}, default {item['default']:g}{unit})"
                ).replace("%", "%%"),
            )
        elif item["kind"] == "choice":
            look.add_argument(
                flag, choices=item["choices"], help=f"{item['label']} (default {item['default']})"
            )
        elif item["kind"] == "switch":
            look.add_argument(flag, action="store_true", default=None, help=item["label"])
        else:
            look.add_argument(flag, type=str, metavar="PATH", help=item["label"])
    return p


def remaster_params(args: argparse.Namespace) -> RemasterParams:
    """The settings these arguments ask for: the approved defaults (or --no-look), with
    every --look-<name> given on top."""
    base = NO_LOOK if getattr(args, "no_look", False) else RemasterParams()
    given = {}
    for item in schema():
        value = getattr(args, "look_" + item["name"], None)
        if value is not None:
            given[item["name"]] = value
    try:
        return base.with_(**given)
    except ValueError as exc:
        raise AnaHDError(f"--look-{exc}") from exc


def remaster_args(params: RemasterParams) -> list[str]:
    """Command-line arguments for ``params`` (only what differs from the defaults), for the
    app's worker, which runs the same command line."""
    out: list[str] = []
    default = RemasterParams()
    for item in schema():
        name = item["name"]
        value = getattr(params, name)
        if value == getattr(default, name):
            continue
        flag = "--look-" + name.replace("_", "-")
        if item["kind"] == "switch":
            if value:
                out.append(flag)
        else:
            out += [flag, f"{value:g}" if isinstance(value, float | int) else str(value)]
    return out


Report = Callable[[int, int | None, float], None]  # frames done, total if known, fps
REPORT_EVERY = 30  # frames


def print_progress(done: int, total: int | None, rate: float) -> None:
    line = f"{done} frames, {rate:.1f} fps"
    if total:
        left = max(total - done, 0) / max(rate, 1e-6)
        line = (
            f"{done}/{total} frames  {min(done / total, 1):4.0%}  "
            f"{rate:.1f} fps  {int(left // 60)}:{int(left % 60):02d} left"
        )
    print(f"\r{line}   ", end="", file=sys.stderr, flush=True)


def part_outputs(out: Path, count: int) -> list[Path]:
    """Where each kept part goes with --parts: AnaHD_x_part1.mp4, ... (one part: out)."""
    if count <= 1:
        return [out]
    return [out.with_name(f"{out.stem}_part{i}{out.suffix}") for i in range(1, count + 1)]


def planned_outputs(src: Path, out: Path, args: argparse.Namespace) -> list[Path]:
    """Every file ``render`` will write for these arguments (checked before cleaning)."""
    cuts = list(getattr(args, "cut", None) or [])
    if not (cuts and getattr(args, "parts", False)):
        return [out]
    info = probe(src)
    return part_outputs(out, len(keep_spans(cuts, count_frames(src), _fps(info.fps))))


def render(
    src: Path, out: Path, args: argparse.Namespace, report: Report | None = None
) -> tuple[Result, float]:
    """Clean ``src`` into ``out`` as ``args`` say (with --parts, one file per kept part:
    see :func:`planned_outputs`); the result and the seconds it took."""
    if not src.is_file():
        raise AnaHDError(f"{src} not found")
    if out.resolve() == src.resolve():
        raise AnaHDError("the output must not replace the recording")
    cuts = list(getattr(args, "cut", None) or [])
    if cuts and (args.start or args.duration):
        raise AnaHDError("--cut does not mix with --start/--duration (cut the start instead)")
    if not (cuts and getattr(args, "parts", False)):
        _check_free([out], args.force)  # before any work (part names need the frame count)
    whole = not (args.start or args.duration)
    packets = count_frames(src) if whole else None
    began = time.monotonic()
    done_before = 0  # frames of earlier parts, when cleaning a cut recording part by part
    expected = packets or 0

    def progress(done: int) -> None:
        done += done_before
        if report is not None and done % REPORT_EVERY == 0:
            total = expected if cuts else packets
            report(done, total, done / max(time.monotonic() - began, 1e-6))

    steady = SteadyConfig() if args.steady else None
    remaster = remaster_params(args)
    temps: list[Path] = []
    clock = sound_clock(src)  # None: the recording has no sound
    info = probe(src)
    rate = _fps(info.fps)

    def with_sound(video: Path, spans: list[AudioSpan]) -> None:
        """Put the recording's sound for ``spans`` into ``video`` (in place)."""
        if clock is None:
            return
        sounded = video.with_name(f"{video.stem}.sound{video.suffix}")
        temps.append(sounded)
        add_audio(video, src, spans, sounded)
        sounded.replace(video)

    def sound_of(seek: float | None, frames: int) -> AudioSpan:
        start = first_frame_time(clock, rate, seek) if clock is not None else 0.0
        return AudioSpan(start, frames / rate)

    try:
        if cuts:
            fps = rate
            spans = keep_spans(cuts, packets or 0, fps)
            if not spans:
                raise AnaHDError("the cuts leave nothing of the recording")
            separate = bool(getattr(args, "parts", False)) and len(spans) > 1
            finals = part_outputs(out, len(spans)) if separate else [out]
            _check_free(finals, args.force)
            expected = sum(s.frames for s in spans)
            measured = measure(src, info, Options())  # one look for every part
            frames, made, sounds = 0, [], []
            for n, span in enumerate(spans):
                part = (
                    partial_output(finals[n])
                    if separate
                    else out.with_name(f"{out.stem}.part{n}{out.suffix}")
                )
                temps.append(part)
                # From half a frame before the first kept frame: exactly that frame on.
                opts = Options(
                    start=None if span.start == 0 else (span.start - 0.5) / fps,
                    duration=span.frames / fps,
                    steady=steady,
                    remaster=remaster,
                )
                done_before = frames
                part_result = _clean_with_one_retry(src, part, opts, args, progress, measured)
                if not span.frames * (1 - MAX_UNREADABLE) - 1 <= part_result.frames <= span.frames:
                    raise AnaHDError(
                        f"{src}: read {part_result.frames} of {span.frames} frames "
                        f"at {span.start / fps:.2f} s"
                    )
                verify(part_result.frames, part)
                frames += part_result.frames
                made.append(part)
                sounds.append(sound_of(opts.start, part_result.frames))
            result = replace(part_result, frames=frames)
            if separate:
                _check_free(finals, args.force)  # none appeared while cleaning
                for part, sound in zip(made, sounds, strict=True):
                    with_sound(part, [sound])
                for part, final in zip(made, finals, strict=True):
                    part.replace(final)
                return result, time.monotonic() - began
            tmp = partial_output(out)
            temps.append(tmp)
            join(made, tmp)
            with_sound(tmp, sounds)
        else:
            _check_free([out], args.force)
            tmp = partial_output(out)
            temps.append(tmp)
            opts = Options(
                start=args.start, duration=args.duration, steady=steady, remaster=remaster
            )
            result = _clean_with_one_retry(src, tmp, opts, args, progress)
            # A packet the decoder rejects (a breakup burst) yields no frame, so the
            # source's packet count may be slightly above what could be read.
            if packets is not None and not (
                packets * (1 - MAX_UNREADABLE) - 1 <= result.frames <= packets
            ):
                raise AnaHDError(f"{src}: read {result.frames} of {packets} frames")
            with_sound(tmp, [sound_of(args.start, result.frames)])
        verify(result.frames, tmp)
        _check_free([out], args.force)  # it did not appear while cleaning
        tmp.replace(out)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)
    return result, time.monotonic() - began


def _check_free(outputs: list[Path], force: bool) -> None:
    for path in outputs:
        if path.exists() and not force:
            raise OutputExistsError(f"{path} exists; pass --force to overwrite it")


# What FFmpeg says when it cannot get memory: the rawvideo demuxer and libx264 both failed
# this way (2026-10-20) when Windows' commit charge (RAM + page file) was nearly used up.
OUT_OF_MEMORY = ("Cannot allocate memory", "Generic error in an external library", "ENOMEM")


def commit_charge() -> tuple[float, float] | None:
    """Windows' committed memory and its limit (RAM + page file), in GB; None elsewhere."""
    if sys.platform != "win32":
        return None

    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_ext", ctypes.c_ulonglong),
        ]

    st = Status()
    st.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return (st.total_page - st.avail_page) / 1e9, st.total_page / 1e9


def _out_of_memory(exc: Exception) -> bool:
    return any(sign in str(exc) for sign in OUT_OF_MEMORY)


def _clean_with_one_retry(
    src: Path,
    tmp: Path,
    opts: Options,
    args: argparse.Namespace,
    progress: Callable[[int], None],
    measured: Measured | None = None,
) -> Result:
    """Clean once; if FFmpeg ran out of memory, free what this process holds and try once
    more. A second failure is reported with the system's memory state."""
    try:
        return _clean(src, tmp, opts, args, progress, measured)
    except MediaError as exc:
        if not _out_of_memory(exc):
            raise
        print(f"\nanahd: FFmpeg ran out of memory ({exc}); retrying once", file=sys.stderr)
    gc.collect()
    if "torch" in sys.modules:
        import torch

        torch.cuda.empty_cache()
    try:
        return _clean(src, tmp, opts, args, progress, measured)
    except MediaError as exc:
        if not _out_of_memory(exc):
            raise
        charge = commit_charge()
        state = f" ({charge[0]:.1f} of {charge[1]:.1f} GB committed)" if charge else ""
        raise MediaError(
            f"Windows ran out of memory twice while encoding{state}; close other programs "
            f"(games, virtual machines) and try again. FFmpeg said: {exc}"
        ) from exc


def _clean(
    src: Path,
    tmp: Path,
    opts: Options,
    args: argparse.Namespace,
    progress: Callable[[int], None],
    measured: Measured | None = None,
) -> Result:
    if args.cpu:
        return enhance(src, tmp, opts, overwrite=True, progress=progress, measured=measured)
    if args.steady:
        raise AnaHDError("--steady runs on the CPU only; add --cpu")
    try:
        from anahd.gpu.pipeline import GpuConfig  # torch is imported only here
        from anahd.gpu.pipeline import enhance as enhance_gpu
    except ImportError as exc:
        raise AnaHDError(f"PyTorch is missing ({exc}); reinstall, or add --cpu") from exc
    from anahd.model import weights

    model = weights.resolve(args.model) if args.learned else None
    return enhance_gpu(
        src,
        tmp,
        opts,
        gcfg=GpuConfig(
            ai=args.ai and not args.learned,
            repair=args.repair,
            learned=model,
            learned_strength=args.learned_strength,
        ),
        overwrite=True,
        nvenc=args.nvenc,
        progress=progress,
        measured=measured,
    )


def _fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    return int(num) / int(den or 1)


def join(parts: list[Path], out: Path) -> None:
    """The cleaned parts, one after another, in one file (no re-encode)."""
    listing = out.with_name(f"{out.stem}.parts.txt")
    # concat list syntax: single quotes, a quote inside written as '\''
    quoted = (p.resolve().as_posix().replace("'", "'\\''") for p in parts)
    listing.write_text("".join(f"file '{q}'\n" for q in quoted), encoding="utf-8")
    try:
        cmd = [require("ffmpeg"), "-nostdin", "-v", "error", "-f", "concat", "-safe", "0"]
        cmd += ["-i", str(listing), "-c", "copy", "-movflags", "+faststart", "-y", str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
        if run.returncode != 0:
            raise MediaError(f"could not join the cleaned parts: {run.stderr.strip()}")
    finally:
        listing.unlink(missing_ok=True)


def run_one(src: Path, out: Path, args: argparse.Namespace) -> None:
    written = planned_outputs(src, out, args) if src.is_file() else [out]
    result, took = render(src, out, args, print_progress)
    names = ", ".join(str(w) for w in written)
    print(
        f"\r{names}: {result.frames} frames at {result.size[0]}x{result.size[1]} "
        f"in {took:.0f} s ({result.frames / took:.1f} fps)"
        + (f", steadied {result.steadied:.0%} zoom {result.zoom:.3f}" if args.steady else "")
        + (f"\n{result.notes}" if result.notes else ""),
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    bundle.setup()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output is not None and len(args.input) > 1:
        parser.error("-o/--output takes one input; use --out-dir for several")
    if args.out_dir is not None and not args.out_dir.is_dir():
        parser.error(f"--out-dir {args.out_dir} is not a folder")
    failed = 0
    with keep_awake():
        for i, src in enumerate(args.input, 1):
            if len(args.input) > 1:
                print(f"[{i}/{len(args.input)}] {src}", file=sys.stderr)
            try:
                run_one(src, args.output or default_output(src, args.out_dir), args)
            except AnaHDError as exc:
                print(f"\nanahd: {exc}", file=sys.stderr)
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
