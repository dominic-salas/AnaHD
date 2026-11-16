"""Digital or analog: the flag that keeps digital recordings out of a clean by default.
The sign is "no OSD" (digital cameras can be 4:3 too, so the shape tells nothing).
Measured on synthetic recordings like everything AnaHD sees (no GPU needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import synthetic as syn
from anahd.media import VideoInfo, probe
from anahd.origin import Origin, judge
from anahd.pipeline import Options, measure


def _judge(
    tmp_path: Path,
    truth: object,
    layout: syn.Layout,
    *,
    lines: int | None = 576,
    fps: int = syn.FPS,
) -> Origin:
    """``lines``: 576 upscaled, like an analog goggle's recording; None: sharp, as a digital
    link's picture is."""
    clip = syn.write_with(tmp_path / "rec.mp4", 120, truth, layout, lines=lines, fps=fps)  # type: ignore[arg-type]
    info = probe(clip.path)
    return judge(info, measure(clip.path, info, Options()))


def test_analog_with_osd_is_not_flagged(tmp_path: Path) -> None:
    o = _judge(tmp_path, syn.clean_frame, syn.FOUR_THREE)
    assert not o.digital and o.osd_share > 0.002, o


def test_a_full_frame_picture_with_an_osd_is_not_flagged(tmp_path: Path) -> None:
    o = _judge(tmp_path, syn.wide_truth, syn.WIDE)
    assert not o.digital, o


def test_a_4_3_recording_without_osd_is_flagged(tmp_path: Path) -> None:
    """A digital camera: 4:3 like analog, no OSD in the recording."""
    o = _judge(tmp_path, syn.without_osd(syn.FOUR_THREE, speed=10), syn.FOUR_THREE)
    assert o.digital and o.reasons == "no OSD", o


def test_a_16_9_recording_without_osd_is_flagged(tmp_path: Path) -> None:
    o = _judge(tmp_path, syn.without_osd(syn.WIDE, speed=10), syn.WIDE)
    assert o.digital, o


def test_known_limit_a_nearly_still_digital_clip_is_not_flagged(tmp_path: Path) -> None:
    """The OSD finder looks for bright detail that stays put, so a scene that barely moves
    (0.3 px a frame) shows some: the clip is not flagged and would just be cleaned. This
    pins the limit down; it is the safe way to be wrong."""
    o = _judge(tmp_path, syn.without_osd(syn.WIDE, speed=1), syn.WIDE, lines=None, fps=50)
    assert not o.digital, o


def test_a_nearly_still_hdzero_shaped_clip_is_caught_by_its_picture(tmp_path: Path) -> None:
    """The same scene at 720p60 is recognised by its detail instead (anahd.origin)."""
    o = _judge(tmp_path, syn.without_osd(syn.WIDE, speed=1), syn.WIDE, lines=None)
    assert o.digital and o.hdzero, o


def _info(w: int, h: int, fps: str) -> VideoInfo:
    return VideoInfo(w, h, fps, False, False, 10.0)


class _Meas:
    def __init__(self, osd: float) -> None:
        self.osd = np.full(4, osd)


def _j(info: VideoInfo, osd: float) -> Origin:
    return judge(info, _Meas(osd))  # type: ignore[arg-type]


def test_90_fps_is_hdzero_even_with_an_osd_found() -> None:
    o = _j(_info(1280, 720, "90/1"), 0.05)
    assert o.digital and o.reasons == "HDZero digital recording"


def test_1080_line_picture_is_hdzero() -> None:
    o = _j(_info(1920, 1080, "60/1"), 0.05)
    assert o.digital and o.hdzero


def test_analog_720p60_keeps_the_osd_rule() -> None:
    assert not _j(_info(1280, 720, "60/1"), 0.05).digital
    o = _j(_info(1280, 720, "60000/1001"), 0.0)
    assert o.digital and o.reasons == "no OSD" and not o.hdzero


def test_50_fps_4_3_is_not_hdzero() -> None:
    o = _j(_info(720, 576, "50/1"), 0.05)
    assert not o.digital and not o.hdzero


# ----- 720p60: HDZero and an analog DVR record the same shape; the picture decides -----


def _plane(kind: str, k: int = 0) -> np.ndarray:
    """A 960x720 luma picture cut from the synthetic scene: as sharp as a digital link
    ("sharp"), 480 lines upscaled like an analog goggle's recording ("analog"), or soft."""
    import cv2

    g = cv2.cvtColor(syn.SCENE.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    y0, x0 = 40 + 80 * k, 60 + 120 * k
    sharp = g[y0 : y0 + 720, x0 : x0 + 960]
    if kind == "sharp":
        rng = np.random.default_rng(k)
        return np.asarray(np.clip(sharp + rng.normal(0, 2, sharp.shape), 0, 255), np.uint8)
    if kind == "analog":
        low = cv2.resize(sharp, (640, 480), interpolation=cv2.INTER_AREA)
        return np.asarray(cv2.resize(low, (960, 720), interpolation=cv2.INTER_LINEAR), np.uint8)
    return np.asarray(cv2.GaussianBlur(sharp, (0, 0), 1.4), np.uint8)


class _Looked:
    def __init__(self, kind: str, osd: float = 0.05) -> None:
        self.osd = np.full(4, osd)
        self.samples = [_plane(kind, k % 3) for k in range(12)]


def test_a_sharp_720p60_picture_is_hdzero_even_with_an_osd_found() -> None:
    o = judge(_info(1280, 720, "60/1"), _Looked("sharp"))  # type: ignore[arg-type]
    assert o.digital and o.hdzero and o.reasons == "HDZero digital recording"


def test_an_analog_720p60_picture_is_left_to_the_osd_rule() -> None:
    for kind in ("analog", "soft"):
        o = judge(_info(1280, 720, "60/1"), _Looked(kind))  # type: ignore[arg-type]
        assert not o.digital and not o.hdzero, kind
        o = judge(_info(1280, 720, "60000/1001"), _Looked(kind, osd=0.0))  # type: ignore[arg-type]
        assert o.digital and not o.hdzero and o.reasons == "no OSD", kind


def test_the_picture_never_decides_for_other_shapes() -> None:
    for info in (_info(1280, 720, "50/1"), _info(720, 576, "50/1"), _info(1280, 960, "60/1")):
        assert not judge(info, _Looked("sharp")).hdzero  # type: ignore[arg-type]


def test_too_few_textured_frames_decide_nothing() -> None:
    flat = np.full((720, 960), 120, np.uint8)
    m = _Looked("sharp")
    m.samples = [flat] * 12
    assert not judge(_info(1280, 720, "60/1"), m).hdzero  # type: ignore[arg-type]
    m.samples = [_plane("sharp")] * 2  # fewer than the three scored frames needed
    assert not judge(_info(1280, 720, "60/1"), m).hdzero  # type: ignore[arg-type]
