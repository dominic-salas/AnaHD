"""End to end against the requirements, on a synthetic analog recording whose clean
picture is known (tests/synthetic.py). Nothing here compares AnaHD with itself: outputs are
measured against the truth and against what a plain Lanczos upscale of the same recording
gives. Needs torch and a CUDA GPU; about a minute.

Requirements covered:
- The product's: the default command needs no flags, writes AnaHD_<stem>.mp4 next to the
  input, 1440x1080 for a 4:3 picture at the recording's 60 fps, never overwrites without
  --force, never modifies the recording, the app's worker process can be stopped and
  leaves nothing behind. Speed is reported, not judged: it depends on the GPU (the target
  is ~18 fps, 10 min per 3-min flight).
- Functional: every frame kept, the output plays, noise is removed, detail is kept,
  breakup bands are repaired, the OSD stays legible, flat areas don't flicker, the picture
  area (black bars) is found, a broken input fails cleanly.
- Non-functional: GPU memory within an 8 GB card, no temporary files left.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import synthetic as syn  # noqa: E402
from anahd import cli  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
FRAMES = 300  # 5 s at 60 fps
SPEED_TARGET_FPS = 184 * 60 / 600  # a 184 s flight within 10 minutes = 18.4 fps
VRAM_LIMIT = 6 * 2**30  # bytes allocated by PyTorch: leaves room on an 8 GB card


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ffprobe(path: Path) -> dict[str, object]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames"]
    cmd += ["-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt,"]
    cmd[-1] += "color_space,nb_read_frames:format=duration"
    run = subprocess.run([*cmd, "-of", "json", str(path)], capture_output=True, text=True)
    data = json.loads(run.stdout)
    return {**data["streams"][0], **data["format"]}


@dataclass
class Run:
    src: Path
    out: Path
    code: int
    src_hash: str
    src_mtime: int
    reports: list[tuple[float, int]] = field(default_factory=list)  # (time, frames done)
    peak_vram: int = 0


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """``anahd flight.mp4``, exactly as a user types it, on the synthetic recording."""
    folder = tmp_path_factory.mktemp("flight")
    clip = syn.write(folder / "flight.mp4", FRAMES)
    r = Run(clip.path, folder / "AnaHD_flight.mp4", -1, sha256(clip.path), 0)
    r.src_mtime = clip.path.stat().st_mtime_ns
    shown = cli.print_progress

    def record(done: int, total: int | None, rate: float) -> None:
        r.reports.append((time.monotonic(), done))
        shown(done, total, rate)

    torch.cuda.reset_peak_memory_stats()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "print_progress", record)
        r.code = cli.main([str(clip.path)])
    r.peak_vram = int(torch.cuda.max_memory_allocated())
    return r


@dataclass
class Scores:
    psnr_out: list[float]  # per frame, after a global colour fit to the truth
    psnr_plain: list[float]
    noise_out: list[float]  # std in the flat patch
    noise_plain: list[float]
    flicker_out: list[float]  # mean |frame - previous frame| in the flat patch
    flicker_plain: list[float]
    band_out: list[float]  # mean abs error in the breakup rows of damaged frames
    band_plain: list[float]
    osd_ncc: list[float]  # correlation of the OSD area with the truth's
    detail_out: list[float]  # gradient energy over the truth's, textured lower half
    detail_plain: list[float]
    edge_out: list[float] = field(default_factory=list)  # mean abs error at hard edges
    edge_plain: list[float] = field(default_factory=list)


def fit(frame: NDArray[np.uint8], truth: NDArray[np.uint8]) -> NDArray[np.float64]:
    """``frame`` through the one global colour transform (3x3 matrix and offset) that best
    matches ``truth``. AnaHD grades tone and colour on purpose (a curve and vibrance, the
    same for the whole frame); a global transform undoes a grade but not noise, blur,
    halos or damage, which are what is compared."""
    x = np.c_[frame.reshape(-1, 3).astype(np.float64), np.ones(frame.shape[0] * frame.shape[1])]
    m, *_ = np.linalg.lstsq(x, truth.reshape(-1, 3).astype(np.float64), rcond=None)
    fitted: NDArray[np.float64] = (x @ m).reshape(frame.shape)
    return fitted


def psnr(a: NDArray[np.float64], b: NDArray[np.uint8]) -> float:
    mse = float(np.mean((a - b.astype(np.float64)) ** 2))
    return float(10 * np.log10(255**2 / max(mse, 1e-9)))


def detail(img: NDArray[np.float64] | NDArray[np.uint8]) -> float:
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g = g[syn.OUT_H // 2 :].astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(np.mean(np.hypot(gx, gy)))


@pytest.fixture(scope="module")
def cleaned(run: Run, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same recording cleaned with --no-look. The C03 look (on by default since
    2026-10-21) is a creative grade with a vignette and a radial colour fix: no global
    colour transform undoes it, so measuring restoration against the truth through
    :func:`fit` would count the grade as damage. The cleaning is the same either way."""
    out = tmp_path_factory.mktemp("flight_clean") / "AnaHD_flight.mp4"
    assert cli.main([str(run.src), "-o", str(out), "--no-look"]) == 0
    return out


@pytest.fixture(scope="module")
def scores(run: Run, cleaned: Path) -> Scores:
    assert run.code == 0, "the render failed; see the output above"
    from anahd.media import probe
    from anahd.pipeline import Options, measure

    c = measure(run.src, probe(run.src), Options()).crop  # the picture area AnaHD used
    assert abs(c.x - syn.PIC_X) <= 4 and abs(c.width - syn.PIC_W) <= 8, c
    area = (c.x, c.y, c.width, c.height)
    size = (syn.OUT_W, syn.OUT_H)
    outs = syn.read_bgr(cleaned, size)
    plains = syn.read_bgr(run.src, size, syn.plain_upscale_filter(*area))
    x, y, w, h = syn.FLAT
    r0, r1 = (round(r * syn.OUT_H / syn.PIC_H) for r in syn.BAND_ROWS)
    osd = syn.osd_mask()
    ys, xs = np.nonzero(osd)
    oy0, oy1 = int(np.min(ys)) - 8, int(np.max(ys)) + 8
    ox0, ox1 = int(np.min(xs)) - 8, int(np.max(xs)) + 8
    s = Scores([], [], [], [], [], [], [], [], [], [], [])
    prev: dict[str, NDArray[np.float64]] = {}
    pairs: Iterator[tuple[NDArray[np.uint8], NDArray[np.uint8]]] = zip(outs, plains, strict=False)
    for i, (out, plain) in enumerate(pairs):
        truth = syn.reframe(syn.clean_frame(i), *area)
        fo, fp = fit(out, truth), fit(plain, truth)
        if syn.damaged(i):
            band = truth[r0:r1].astype(np.float64)
            s.band_out.append(float(np.mean(np.abs(fo[r0:r1] - band))))
            s.band_plain.append(float(np.mean(np.abs(fp[r0:r1] - band))))
        else:
            s.psnr_out.append(psnr(fo, truth))
            s.psnr_plain.append(psnr(fp, truth))
        s.noise_out.append(float(fo[y : y + h, x : x + w].std(axis=(0, 1)).mean()))
        s.noise_plain.append(float(fp[y : y + h, x : x + w].std(axis=(0, 1)).mean()))
        for key, f, dest in (("o", fo, s.flicker_out), ("p", fp, s.flicker_plain)):
            patch = f[y : y + h, x : x + w]
            if key in prev and not syn.damaged(i) and not syn.damaged(i - 1):
                dest.append(float(np.mean(np.abs(patch - prev[key]))))
            prev[key] = patch
        a = out[oy0:oy1, ox0:ox1].astype(np.float64).ravel()
        b = truth[oy0:oy1, ox0:ox1].astype(np.float64).ravel()
        s.osd_ncc.append(float(np.corrcoef(a, b)[0, 1]))
        grey = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(grey, 50, 150), np.ones((5, 5), np.uint8)) > 0
        s.edge_out.append(float(np.abs(fo - truth)[edges].mean()))
        s.edge_plain.append(float(np.abs(fp - truth)[edges].mean()))
        s.detail_out.append(detail(fo) / detail(truth))
        s.detail_plain.append(detail(fp) / detail(truth))
    assert len(s.osd_ncc) == FRAMES, "could not compare every frame"
    print(  # the measurements, for the record
        f"\nPSNR out {np.mean(s.psnr_out):.2f} plain {np.mean(s.psnr_plain):.2f} dB; "
        f"noise out {np.mean(s.noise_out):.2f} plain {np.mean(s.noise_plain):.2f}; "
        f"flicker out {np.mean(s.flicker_out):.2f} plain {np.mean(s.flicker_plain):.2f}; "
        f"band MAE out {np.mean(s.band_out):.1f} plain {np.mean(s.band_plain):.1f}; "
        f"OSD NCC min {min(s.osd_ncc):.3f}; detail out {np.mean(s.detail_out):.2f} "
        f"plain {np.mean(s.detail_plain):.2f} of the truth; edge MAE out "
        f"{np.mean(s.edge_out):.1f} plain {np.mean(s.edge_plain):.1f}"
    )
    return s


# ----- the product's requirements -----


def test_default_command_writes_anahd_stem_next_to_the_input(run: Run) -> None:
    assert run.code == 0
    assert run.out == run.src.with_name("AnaHD_flight.mp4") and run.out.is_file()
    assert sorted(p.name for p in run.src.parent.iterdir()) == ["AnaHD_flight.mp4", "flight.mp4"]


def test_output_is_hd_at_native_aspect_and_the_recordings_60_fps(run: Run) -> None:
    info = ffprobe(run.out)
    assert (info["width"], info["height"]) == (1440, 1080)  # 4:3 picture -> 4:3 HD
    assert info["r_frame_rate"] == "60/1"
    assert info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p"
    assert info["color_space"] == "bt709"


def test_recording_is_never_modified(run: Run) -> None:
    assert sha256(run.src) == run.src_hash
    assert run.src.stat().st_mtime_ns == run.src_mtime


def test_existing_output_is_not_overwritten(run: Run) -> None:
    before = sha256(run.out)
    assert cli.main([str(run.src)]) == 1
    assert sha256(run.out) == before


def test_speed_is_reported(run: Run, record_property: Callable[[str, object], None]) -> None:
    """Steady throughput (start-up and the first 2 s excluded) on 1280x720 DVR input,
    reported against the 10-minute target; it depends on the GPU, so it is not a pass/fail."""
    steady = [(t, n) for t, n in run.reports if n >= 120]
    assert len(steady) >= 3, run.reports  # progress was reported while it ran
    (t0, n0), (t1, n1) = steady[0], steady[-1]
    fps = (n1 - n0) / (t1 - t0)
    gpu = torch.cuda.get_device_name(0)
    verdict = "meets" if fps >= SPEED_TARGET_FPS else "below"
    text = f"steady {fps:.1f} fps on {gpu}, {verdict} the {SPEED_TARGET_FPS:.1f} fps target"
    print("\n" + text)
    record_property("report", text)


# ----- functional -----


def test_every_frame_is_kept_and_the_output_plays(run: Run) -> None:
    info = ffprobe(run.out)
    assert int(str(info["nb_read_frames"])) == FRAMES
    assert abs(float(str(info["duration"])) - FRAMES / 60) < 1.5 / 60
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(run.out), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert decode.returncode == 0 and decode.stderr.strip() == ""


def test_noise_is_removed(scores: Scores) -> None:
    """In a flat area the truth is constant: what varies is noise. AnaHD must remove at
    least half of what a plain upscale keeps."""
    assert np.mean(scores.noise_out) <= 0.5 * np.mean(scores.noise_plain)


def test_closer_to_the_clean_picture_than_a_plain_upscale(scores: Scores) -> None:
    assert np.mean(scores.psnr_out) >= np.mean(scores.psnr_plain) + 1.0
    # and not by luck on a few frames: at least 90 % of the frames are better
    better = np.array(scores.psnr_out) > np.array(scores.psnr_plain)
    assert better.mean() >= 0.9


def test_detail_is_kept_not_smeared_or_oversharpened(scores: Scores) -> None:
    """Gradient energy of the textured half, relative to the truth's: a denoiser that
    smears loses it, one that oversharpens or leaves noise adds to it."""
    ratio = float(np.mean(scores.detail_out))
    assert 0.6 <= ratio <= 1.4
    assert abs(ratio - 1) <= abs(float(np.mean(scores.detail_plain)) - 1) + 0.1


@pytest.mark.xfail(
    strict=True,
    reason="known issue (README, Known limitations): faint dark halos along hard edges "
    "with the learned restorer. strict: this fails loudly once it is fixed",
)
def test_hard_edges_are_no_worse_than_a_plain_upscale(scores: Scores) -> None:
    assert np.mean(scores.edge_out) <= np.mean(scores.edge_plain)


def test_breakup_bands_are_repaired(scores: Scores) -> None:
    assert len(scores.band_out) >= 9
    assert np.mean(scores.band_out) <= 0.35 * np.mean(scores.band_plain)


def test_osd_stays_legible(scores: Scores) -> None:
    """The OSD text matches the truth in every frame (no ghosting from moving neighbours)."""
    assert min(scores.osd_ncc) >= 0.85


def test_flat_areas_do_not_flicker(scores: Scores) -> None:
    assert np.mean(scores.flicker_out) <= 0.5 * np.mean(scores.flicker_plain)


def test_broken_input_fails_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 64)
    assert cli.main([str(bad)]) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["broken.mp4"]


# ----- non-functional -----


def test_gpu_memory_fits_an_8_gb_card(run: Run) -> None:
    print(f"\npeak VRAM allocated {run.peak_vram / 2**30:.2f} GiB")
    assert 0 < run.peak_vram <= VRAM_LIMIT


# ----- the app's worker process -----


@pytest.fixture(scope="module")
def qapp() -> object:
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QCoreApplication

    return QCoreApplication.instance() or QCoreApplication(sys.argv[:1])


def wait_for(signal: object, seconds: float) -> list[object]:
    from PySide6.QtCore import QEventLoop, QTimer

    got: list[object] = []
    loop = QEventLoop()

    def done(*args: object) -> None:
        got.append(args)
        loop.quit()

    signal.connect(done)  # type: ignore[attr-defined]
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()
    signal.disconnect(done)  # type: ignore[attr-defined]
    return got


def test_app_worker_cleans_a_clip_and_finds_the_picture_area(qapp: object, tmp_path: Path) -> None:
    from anahd.app.worker import Runner

    clip = syn.write(tmp_path / "short.mp4", 90)
    out = tmp_path / "AnaHD_short.mp4"
    runner = Runner()
    failures = []
    runner.failed.connect(lambda *a: failures.append(a))
    runner.start(clip.path, out)
    got = wait_for(runner.finished, 180)
    assert not failures, failures
    assert got, "the worker did not finish in 3 minutes"
    frames, _took, crop, _notes, gpu = got[0][1]  # type: ignore[index]
    assert frames == 90
    assert isinstance(gpu, str) and gpu, "the worker did not name the GPU it cleaned on"
    x, y, w, h = crop
    assert abs(x - syn.PIC_X) <= 4 and abs(w - syn.PIC_W) <= 8 and y <= 4 and h >= 712
    assert out.is_file() and not runner.busy


def test_stopping_the_app_worker_leaves_nothing_behind(qapp: object, tmp_path: Path) -> None:
    from anahd.app.worker import Runner

    clip = syn.write(tmp_path / "long.mp4", 600)
    src_hash = sha256(clip.path)
    out = tmp_path / "AnaHD_long.mp4"
    runner = Runner()
    runner.start(clip.path, out)
    assert wait_for(runner.progress, 180), "no progress from the worker"
    began = time.monotonic()
    runner.stop()
    assert time.monotonic() - began < 10
    assert not runner.busy
    assert sorted(p.name for p in tmp_path.iterdir()) == ["long.mp4"]
    assert sha256(clip.path) == src_hash


# ----- fast rolls and flips pass through; a 16:9 picture stays whole -----

ROLL_FRAMES = 120  # still, a 3 deg/frame roll, a 12 deg/frame flip, still


def grey_small(img: NDArray[np.uint8] | NDArray[np.float64], scale: int = 4) -> NDArray[np.float32]:
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return np.asarray(g[::scale, ::scale], np.float32)


def residual_motion(
    truth: NDArray[np.uint8], frame: NDArray[np.float64], mode: int, scale: int = 4
) -> NDArray[np.float32]:
    """The warp (ECC, OSD masked out) that best maps ``frame`` onto ``truth``: identity
    when the output shows the camera exactly where the truth has it."""
    mask = np.full(truth.shape[:2], 255, np.uint8)
    mask[syn.OSD_ORG[1] - 60 :, : syn.OSD_ORG[0] + 420] = 0  # the OSD stays put on purpose
    start = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    _, found = cv2.findTransformECC(
        grey_small(truth, scale),
        grey_small(frame, scale),
        start,
        mode,
        criteria,
        mask[::scale, ::scale],
        5,
    )
    return np.asarray(found, np.float32)


@pytest.fixture(scope="module")
def roll(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """The recording, its default output and its output with --no-look (see
    :func:`cleaned`: fidelity to the truth is measured without the creative grade)."""
    folder = tmp_path_factory.mktemp("roll")
    clip = syn.write_with(folder / "roll.mp4", ROLL_FRAMES, syn.roll_truth, syn.FOUR_THREE)
    assert cli.main([str(clip.path)]) == 0
    clean = folder / "AnaHD_roll_clean.mp4"
    assert cli.main([str(clip.path), "-o", str(clean), "--no-look"]) == 0
    return clip.path, folder / "AnaHD_roll.mp4", clean


def test_fast_rolls_and_flips_are_not_smoothed_or_damped(
    roll: tuple[Path, Path, Path],
) -> None:
    """Every output frame shows the camera's own roll angle at that frame, through a
    180 deg/s roll and a 720 deg/s flip: nothing lags behind (smoothing) or falls short
    (damping) in the default output, and in the cleaning no frame is blended with its
    neighbours (each is closest to its own truth)."""
    from anahd.media import probe
    from anahd.pipeline import Options, measure

    src, out, clean = roll
    c = measure(src, probe(src), Options()).crop
    area = (c.x, c.y, c.width, c.height)
    outs = list(syn.read_bgr(out, (syn.OUT_W, syn.OUT_H)))
    cleans = list(syn.read_bgr(clean, (syn.OUT_W, syn.OUT_H)))
    assert len(outs) == len(cleans) == ROLL_FRAMES
    truths = [syn.reframe(syn.roll_truth(i), *area) for i in range(ROLL_FRAMES)]
    angles, wrong_frame = [], []
    # Frames whose camera angle equals a neighbour's (the first frame of the roll is still
    # at 0 deg) cannot tell their own frame from that neighbour, so they are left out.
    moving = [
        i
        for i in [*syn.ROLL, *syn.FLIP]
        if syn.roll_angle(i) not in (syn.roll_angle(i - 1), syn.roll_angle(i + 1))
    ]
    assert len(moving) >= 55
    for i in moving:
        fitted = fit(outs[i], truths[i])
        w = residual_motion(truths[i], fitted, cv2.MOTION_EUCLIDEAN)
        angles.append(abs(float(np.degrees(np.arctan2(w[1, 0], w[0, 0])))))
        err = {
            j: float(np.mean(np.abs(fit(cleans[i], truths[j]) - truths[j])))
            for j in (i - 1, i, i + 1)
        }
        if not err[i] <= 0.8 * min(err[i - 1], err[i + 1]):
            wrong_frame.append((i, err))
    print(f"\nroll residual: max {max(angles):.2f} deg, mean {np.mean(angles):.2f} deg")
    assert max(angles) <= 0.5, "the output's roll differs from the camera's"
    assert wrong_frame == [], "frames closer to a neighbouring frame than to their own"


def test_fast_motion_is_not_worse_than_a_plain_upscale(roll: tuple[Path, Path, Path]) -> None:
    """Through the flip, fusing neighbours that the flow cannot follow would ghost: the
    output must still be at least as close to the truth as a plain upscale."""
    from anahd.media import probe
    from anahd.pipeline import Options, measure

    src, _, out = roll  # the cleaning (--no-look), see :func:`roll`
    c = measure(src, probe(src), Options()).crop
    area = (c.x, c.y, c.width, c.height)
    size = (syn.OUT_W, syn.OUT_H)
    outs = list(syn.read_bgr(out, size))
    plains = list(syn.read_bgr(src, size, syn.plain_upscale_filter(*area)))
    po, pp = [], []
    for i in syn.FLIP:
        truth = syn.reframe(syn.roll_truth(i), *area)
        po.append(psnr(fit(outs[i], truth), truth))
        pp.append(psnr(fit(plains[i], truth), truth))
    print(f"\nflip PSNR out {np.mean(po):.2f} plain {np.mean(pp):.2f} dB, worst out {min(po):.2f}")
    assert np.mean(po) >= np.mean(pp)
    assert min(po) >= min(pp) - 1.0  # no single frame falls apart


def test_16_9_source_comes_out_1920x1080_without_bars_or_crop(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A 16:9 picture filling the DVR frame: 1920x1080 out, every edge is picture (no
    bars), and the whole picture is kept (no crop: no zoom, no shift against the truth)."""
    folder = tmp_path_factory.mktemp("wide")
    frames = 90
    clip = syn.write_with(folder / "wide.mp4", frames, syn.wide_truth, syn.WIDE)
    assert cli.main([str(clip.path)]) == 0
    out = folder / "AnaHD_wide.mp4"
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1920, 1080)
    assert int(str(info["nb_read_frames"])) == frames
    outs = list(syn.read_bgr(out, (1920, 1080)))
    for i in (10, 45, 80):
        truth = syn.wide_truth(i)
        fitted = fit(outs[i], truth)
        whole = float(np.mean(np.abs(fitted - truth)))
        strips = {
            "left": (slice(None), slice(0, 6)),
            "right": (slice(None), slice(-6, None)),
            "top": (slice(0, 6), slice(None)),
            "bottom": (slice(-6, None), slice(None)),
        }
        for side, (rows, cols) in strips.items():
            raw = outs[i][rows, cols].astype(np.float64)
            assert raw.mean() > 40, f"frame {i}: a dark bar on the {side}"
            edge = float(np.mean(np.abs(fitted[rows, cols] - truth[rows, cols])))
            assert edge <= 3 * whole + 4, f"frame {i}: the {side} edge is not the picture"
        w = residual_motion(truth, fitted, cv2.MOTION_AFFINE, scale=2)
        sx, sy = float(np.hypot(w[0, 0], w[1, 0])), float(np.hypot(w[0, 1], w[1, 1]))
        print(
            f"\nframe {i}: scale {sx:.4f} x {sy:.4f}, shift {w[0, 2] * 2:.2f}, {w[1, 2] * 2:.2f} px"
        )
        assert abs(sx - 1) <= 0.002 and abs(sy - 1) <= 0.002, "zoomed: part of the picture is cut"
        assert abs(w[0, 2]) * 2 <= 1.5 and abs(w[1, 2]) * 2 <= 1.5, "shifted against the truth"


# ----- cut markers: exactly the kept frames come out, in order, as one file -----


def test_cuts_keep_exactly_the_frames_between_them(tmp_path: Path) -> None:
    """Cut the first 0.5 s, 2.0-2.5 s and the last 0.25 s of a 5 s recording whose frames
    are told apart by their content: the output must be frames 30-119 then 150-284,
    nothing else, nothing twice, nothing from inside a cut."""
    clip = syn.write_with(tmp_path / "flight.mp4", 300, syn.jumping_truth, syn.FOUR_THREE)
    src_hash = sha256(clip.path)
    code = cli.main([str(clip.path), "--cut", "0-0.5", "--cut", "2-2.5", "--cut", "4.75-5"])
    assert code == 0
    out = tmp_path / "AnaHD_flight.mp4"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AnaHD_flight.mp4", "flight.mp4"]
    assert sha256(clip.path) == src_hash
    info = ffprobe(out)
    assert (info["width"], info["height"], info["r_frame_rate"]) == (1440, 1080, "60/1")
    expected = [*range(30, 120), *range(150, 285)]
    got = syn.identify(syn.read_bgr(out, (syn.OUT_W, syn.OUT_H)), 300, syn.jumping_truth)
    assert int(str(info["nb_read_frames"])) == len(expected) == len(got)
    wrong = [(k, e, g) for k, (e, g) in enumerate(zip(expected, got, strict=True)) if e != g]
    assert wrong == [], f"output frame, expected source frame, got: {wrong[:10]}"


def test_the_frame_identifier_works_on_the_recording_itself(tmp_path: Path) -> None:
    """The checks above are only as good as the identifier: it must name every frame of
    the noisy recording (plain upscale) correctly."""
    clip = syn.write_with(tmp_path / "flight.mp4", 120, syn.jumping_truth, syn.FOUR_THREE)
    frames = syn.read_bgr(clip.path, (syn.OUT_W, syn.OUT_H), syn.plain_upscale_filter())
    assert syn.identify(frames, 120, syn.jumping_truth) == list(range(120))


def test_parts_export_each_kept_part_as_its_own_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--parts (the app's "each part separately"): one file per kept part, named
    AnaHD_<name>_part<n>.mp4, each holding exactly its frames."""
    clip = syn.write_with(tmp_path / "flight.mp4", 300, syn.jumping_truth, syn.FOUR_THREE)
    assert cli.main([str(clip.path), "--cut", "2-2.5", "--parts"]) == 0
    said = capsys.readouterr().err
    assert "AnaHD_flight_part1.mp4" in said and "AnaHD_flight_part2.mp4" in said
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["AnaHD_flight_part1.mp4", "AnaHD_flight_part2.mp4", "flight.mp4"]
    size = (syn.OUT_W, syn.OUT_H)
    first = syn.identify(
        syn.read_bgr(tmp_path / "AnaHD_flight_part1.mp4", size), 300, syn.jumping_truth
    )
    second = syn.identify(
        syn.read_bgr(tmp_path / "AnaHD_flight_part2.mp4", size), 300, syn.jumping_truth
    )
    assert first == list(range(0, 120))
    assert second == list(range(150, 300))


def test_parts_never_overwrite_and_write_nothing_if_one_exists(tmp_path: Path) -> None:
    clip = syn.write_with(tmp_path / "flight.mp4", 120, syn.numbered_truth, syn.FOUR_THREE)
    (tmp_path / "AnaHD_flight_part2.mp4").write_bytes(b"earlier")
    assert cli.main([str(clip.path), "--cut", "1-1.5", "--parts"]) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AnaHD_flight_part2.mp4", "flight.mp4"]
    assert (tmp_path / "AnaHD_flight_part2.mp4").read_bytes() == b"earlier"


# ----- other analog DVRs: SD sizes with tall or wide pixels, PAL rates, MJPEG -----


@pytest.mark.parametrize(
    ("name", "layout", "fps", "sar", "mjpeg"),
    [
        ("ntsc_tagged", syn.NTSC, 60, "10/11", False),  # 720x480, pixels tagged
        ("ntsc_untagged", syn.NTSC, 30, None, False),  # 720x480, square pixels claimed
        ("pal_50fps", syn.PAL, 50, "12/11", False),  # 720x576 at 50 fps
        ("vga_mjpeg", syn.VGA, 30, None, True),  # 640x480 MJPEG 4:2:2, older DVRs
    ],
)
def test_other_dvr_formats_come_out_4_3_hd_with_true_geometry(
    tmp_path: Path, name: str, layout: syn.Layout, fps: int, sar: str | None, mjpeg: bool
) -> None:
    """A 4:3 analog picture recorded by an SD DVR comes out 1440x1080 at the recording's
    rate with every frame, and the picture's geometry is the camera's: not stretched by
    the DVR's pixel shape (scale against the truth within 1 %)."""
    from anahd.media import probe
    from anahd.pipeline import Options, measure

    ext = ".avi" if mjpeg else ".mp4"
    clip = syn.write_with(
        tmp_path / f"{name}{ext}", 60, syn.jumping_truth, layout, fps=fps, sar=sar, mjpeg=mjpeg
    )
    assert cli.main([str(clip.path)]) == 0
    out = tmp_path / f"AnaHD_{name}.mp4"
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1440, 1080), info
    assert info["r_frame_rate"] == f"{fps}/1"
    assert int(str(info["nb_read_frames"])) == 60
    c = measure(clip.path, probe(clip.path), Options()).crop
    frames = list(syn.read_bgr(out, (1440, 1080)))
    for i in (10, 50):
        truth = syn.reframe(syn.jumping_truth(i), c.x, c.y, c.width, c.height, layout)
        w = residual_motion(truth, fit(frames[i], truth), cv2.MOTION_AFFINE, scale=2)
        sx, sy = float(np.hypot(w[0, 0], w[1, 0])), float(np.hypot(w[0, 1], w[1, 1]))
        print(f"\n{name} frame {i}: scale {sx:.4f} x {sy:.4f}")
        assert abs(sx - 1) <= 0.01 and abs(sy - 1) <= 0.01, f"stretched: {sx:.3f} x {sy:.3f}"
    assert syn.identify(iter(frames), 60, syn.jumping_truth) == list(range(60))


def test_classic_cleaning_works_on_its_own(tmp_path: Path) -> None:
    """The app's "Classic" choice (--no-learned), for other pilots' cameras: a valid HD
    output with every frame, and the noise still clearly reduced."""
    clip = syn.write(tmp_path / "flight.mp4", 90)
    assert cli.main([str(clip.path), "--no-learned"]) == 0
    out = tmp_path / "AnaHD_flight.mp4"
    info = ffprobe(out)
    assert (info["width"], info["height"], info["r_frame_rate"]) == (1440, 1080, "60/1")
    assert int(str(info["nb_read_frames"])) == 90
    x, y, w, h = syn.FLAT
    size = (syn.OUT_W, syn.OUT_H)
    outs = list(syn.read_bgr(out, size))
    plains = list(syn.read_bgr(clip.path, size, syn.plain_upscale_filter()))
    noise = [
        (
            fit(o, t)[y : y + h, x : x + w].std(axis=(0, 1)).mean(),
            fit(p, t)[y : y + h, x : x + w].std(axis=(0, 1)).mean(),
        )
        for i, (o, p) in enumerate(zip(outs, plains, strict=True))
        if not syn.damaged(i)
        for t in [syn.clean_frame(i)]
    ]
    out_noise, plain_noise = (float(np.mean(v)) for v in zip(*noise, strict=True))
    print(f"\nclassic noise {out_noise:.2f} vs plain {plain_noise:.2f}")
    assert out_noise <= 0.6 * plain_noise
