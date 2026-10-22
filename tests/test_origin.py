"""Digital or analog: the flag that keeps digital recordings out of a clean by default.
The sign is "no OSD" (digital cameras can be 4:3 too, so the shape tells nothing).
Measured on synthetic recordings like everything AnaHD sees (no GPU needed)."""

from __future__ import annotations

from pathlib import Path

import synthetic as syn
from anahd.media import probe
from anahd.origin import Origin, judge
from anahd.pipeline import Options, measure


def _judge(tmp_path: Path, truth: object, layout: syn.Layout) -> Origin:
    clip = syn.write_with(tmp_path / "rec.mp4", 120, truth, layout)  # type: ignore[arg-type]
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
    assert o.digital and o.reasons == "NO OSD", o


def test_a_16_9_recording_without_osd_is_flagged(tmp_path: Path) -> None:
    o = _judge(tmp_path, syn.without_osd(syn.WIDE, speed=10), syn.WIDE)
    assert o.digital, o


def test_known_limit_a_nearly_still_digital_clip_is_not_flagged(tmp_path: Path) -> None:
    """The OSD finder looks for bright detail that stays put, so a scene that barely moves
    (0.3 px a frame) shows some: the clip is not flagged and would just be cleaned. This
    pins the limit down; it is the safe way to be wrong."""
    o = _judge(tmp_path, syn.without_osd(syn.WIDE, speed=1), syn.WIDE)
    assert not o.digital, o
