"""The OSD mask: the OSD's own strokes and outline, never a box, and never still scenery."""

from __future__ import annotations

import cv2
import numpy as np

from anahd.picture import OSD_FEATHER, OSD_GROW, detect_osd, osd_alpha

H, W = 240, 320
TEXT = "1      1"  # two elements on one line, a wide gap between them


def _scene(n: int, seed: int = 0) -> list[np.ndarray]:
    """A moving, textured scene: smooth noise drifting a few pixels per frame."""
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.uniform(40, 140, (H + 80, W + 80)).astype(np.float32), (0, 0), 3)
    return [big[3 * i % 60 : 3 * i % 60 + H, 2 * i % 60 : 2 * i % 60 + W].copy() for i in range(n)]


def _glyphs(frame: np.ndarray) -> np.ndarray:
    """Outlined OSD text drawn at a fixed place: dark outline, bright strokes."""
    out = np.clip(frame, 0, 255).astype(np.uint8)
    for colour, thick in ((10, 5), (235, 2)):
        cv2.putText(out, TEXT, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, thick)
    return out


def _strokes() -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.putText(m, TEXT, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1, 2)
    return m > 0


def _u8(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [np.clip(f, 0, 255).astype(np.uint8) for f in frames]


def test_still_scenery_with_edges_and_no_glyphs_is_not_osd() -> None:
    """A drone sitting on a desk: a still, detailed picture holds no OSD by itself."""
    still = _scene(1)[0]
    still[100:130, 150:210] = np.where((np.arange(60) // 6) % 2, 40, 140)  # hard edges
    assert not detect_osd(_u8([still + 0.0 for _ in range(12)])).any()


def test_a_glyph_is_its_strokes_and_outline_not_its_box() -> None:
    mask = detect_osd(_u8([_glyphs(f) for f in _scene(12)]))
    strokes = _strokes()
    assert mask[strokes].all(), "a stroke pixel is not in the mask"
    near = cv2.dilate(strokes.astype(np.uint8), np.ones((2 * OSD_GROW + 5,) * 2)) > 0
    assert not (mask & ~near).any(), "the mask reaches beyond the strokes' outline"
    ys, xs = np.nonzero(strokes)
    gap = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1][:, 40:-40]
    assert not gap.any(), "the space between elements on a line was filled"


def test_a_glyph_over_still_scenery_keeps_only_the_element() -> None:
    """Text over a still desk: the desk's edges away from the text are not OSD."""
    still = _scene(1)[0]
    still[120:200, 20:300] = np.where((np.arange(280) // 6) % 2, 40, 140)
    mask = detect_osd(_u8([_glyphs(still) for _ in range(12)]))
    assert mask[_strokes()].all()
    assert not mask[130:200, :].any(), "still scenery below the text became OSD"


def test_alpha_is_one_on_the_mask_and_fades_out_within_the_feather() -> None:
    osd = np.zeros((40, 40), bool)
    osd[15:25, 15:25] = True
    a = osd_alpha(osd)
    assert (a[osd] == 1.0).all()
    assert 0.0 < a[20, 25] < 1.0 and a[20, 25] > a[20, 27]  # just outside: partly kept
    assert a[20, 24 + OSD_FEATHER] > 0.0
    assert (a[20, 25 + OSD_FEATHER :] == 0.0).all()
    assert not osd_alpha(np.zeros((8, 8), bool)).any()
