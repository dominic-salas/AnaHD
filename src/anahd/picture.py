"""Where the picture is: the 4:3 area inside the DVR's black bars, and the burned-in OSD.

Adapted from FPVEdit's ``picture.py`` (same author, MIT). Both are estimated from frames
spread over the recording. The OSD mask covers each OSD element's strokes (every stroke
seen in any sample, so digits that change are covered too) and their outline, not its box.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from anahd.media import Crop, Plane

BoolMask = NDArray[np.bool_]

MIN_FRAMES = 6
OSD_GROW = 4  # px around the OSD's strokes: their dark outline and halo
OSD_FEATHER = 4  # px outside the OSD over which cleaning fades in
OSD_SHOWN_REACH = 24  # px: a stroke shown this close counts its whole element as shown


@dataclass(frozen=True)
class PictureConfig:
    # A column/row belongs to the picture if, in most frames, it is clearly brighter than
    # the bars or carries noise (analog video is never perfectly flat).
    level_above_black: float = 8.0
    noise_std: float = 2.0
    min_active_fraction: float = 0.5
    edge_inset_fraction: float = 0.0  # keep the whole picture; bars are snapped below
    # OSD glyphs: bright strokes with a dark outline, present in many sampled frames.
    # Tuned on the BoxPro+ DVR test set (2026-10-17): catches whole OSD lines, not fragments.
    glyph_level: int = 160
    glyph_outline: int = 70
    glyph_presence: float = 0.08
    # Other burned-in overlays (logos, crosshair): unchanging and detailed.
    static_stability: float = 0.8
    static_change: int = 10
    static_edge: int = 25
    max_overlay_box: float = 0.03  # larger static regions are scenery (props, frame arms)
    min_overlay_pixels: int = 30


def detect_crop(frames: list[Plane], config: PictureConfig | None = None) -> Crop:
    """Bounding box of the picture inside black bars (even offsets); the whole frame if unsure."""
    config = config or PictureConfig()
    height, width = frames[0].shape
    whole = Crop(0, 0, width, height)
    if len(frames) < 2:
        return whole
    stack = np.stack(frames).astype(np.float32)
    columns = _active_profile(stack, axis=1, config=config)
    rows = _active_profile(stack, axis=2, config=config)
    xs, ys = np.flatnonzero(columns), np.flatnonzero(rows)
    if len(xs) == 0 or len(ys) == 0:
        return whole
    x0, x1 = _inset(int(xs[0]), int(xs[-1]) + 1, width, config)
    y0, y1 = _inset(int(ys[0]), int(ys[-1]) + 1, height, config)
    if (
        x1 - x0 < config.min_active_fraction * width
        or y1 - y0 < config.min_active_fraction * height
    ):
        return whole
    return Crop(x0, y0, x1 - x0, y1 - y0)


def _active_profile(stack: NDArray[np.float32], *, axis: int, config: PictureConfig) -> BoolMask:
    means = stack.mean(axis=axis)
    noise = stack.std(axis=axis)
    black = float(np.median(np.percentile(means, 1, axis=1)))
    lit = (means > black + config.level_above_black) | (noise > config.noise_std)
    return np.asarray(lit.mean(axis=0) >= 0.5)


def _inset(start: int, end: int, size: int, config: PictureConfig) -> tuple[int, int]:
    """Skip the soft edge next to a bar; keep offsets and sizes even for 4:2:0."""
    inset = max(1, round(config.edge_inset_fraction * size))
    if start > 0:
        start += inset
    if end < size:
        end -= inset
    start += start % 2
    end -= (end - start) % 2
    return start, max(start + 2, end)


def detect_osd(frames: list[Plane], config: PictureConfig | None = None) -> BoolMask:
    """Mask of burned-in overlay pixels at the frames' size (all False if unsure)."""
    config = config or PictureConfig()
    shape = frames[0].shape
    if len(frames) < MIN_FRAMES:
        return np.zeros(shape, dtype=bool)
    stack = np.stack(frames)
    kernel = np.ones((3, 3), np.uint8)
    glyphs = np.zeros(shape, dtype=np.float64)
    for frame in stack:
        darkest_nearby = np.asarray(cv2.erode(frame, np.ones((5, 5), np.uint8)), dtype=np.int16)
        bright = frame >= config.glyph_level
        outlined = darkest_nearby < frame.astype(np.int16) - config.glyph_outline
        glyphs += bright & outlined
    glyph_mask = glyphs / len(stack) >= config.glyph_presence

    changes = np.abs(np.diff(stack.astype(np.int16), axis=0))
    stable = (changes < config.static_change).mean(axis=0) >= config.static_stability
    median = np.median(stack, axis=0).astype(np.uint8)
    edges = np.asarray(cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)) > config.static_edge

    # An OSD element is where glyphs are: the OSD font draws every element (digits, icons,
    # crosshair, horizon) as bright strokes with a dark outline. Join each element's strokes;
    # large regions are scenery.
    joined = np.asarray(
        cv2.morphologyEx(
            glyph_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15)),
        )
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    strokes = np.bincount(labels[glyph_mask].astype(np.intp), minlength=count)
    boxes = stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    ok = (boxes <= config.max_overlay_box * joined.size) & (strokes >= config.min_overlay_pixels)
    ok[0] = False
    element = ok[labels]
    # Only the element's own pixels, never its box: a box of untouched noise in a cleaned
    # picture shows. A stroke seen in any sample counts (digits change), and so does what
    # holds still with an edge in the element (outlines, icons); grown to take in
    # the outline and halo.
    mask = ((glyphs > 0) | (stable & edges)) & element
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OSD_GROW + 1, 2 * OSD_GROW + 1))
    return np.asarray(cv2.dilate(mask.astype(np.uint8), grow), dtype=bool)


def detect_osd_regions(frames: list[Plane], config: PictureConfig | None = None) -> BoolMask:
    """Where burned-in overlays may be, at the frames' size (all False if unsure): each
    element's whole box, and still detailed areas (which may be scenery). What the cleaning
    leaves out when judging the picture (breakup, motion, levels), never what it leaves
    uncleaned: that is ``detect_osd``."""
    config = config or PictureConfig()
    shape = frames[0].shape
    if len(frames) < MIN_FRAMES:
        return np.zeros(shape, dtype=bool)
    stack = np.stack(frames)
    kernel = np.ones((3, 3), np.uint8)
    glyphs = np.zeros(shape, dtype=np.float64)
    for frame in stack:
        darkest_nearby = np.asarray(cv2.erode(frame, np.ones((5, 5), np.uint8)), dtype=np.int16)
        bright = frame >= config.glyph_level
        outlined = darkest_nearby < frame.astype(np.int16) - config.glyph_outline
        glyphs += bright & outlined
    glyph_mask = glyphs / len(stack) >= config.glyph_presence

    changes = np.abs(np.diff(stack.astype(np.int16), axis=0))
    stable = (changes < config.static_change).mean(axis=0) >= config.static_stability
    median = np.median(stack, axis=0).astype(np.uint8)
    edges = np.asarray(cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)) > config.static_edge
    mask = (glyph_mask | (stable & edges)).astype(np.uint8)

    # Join each element's strokes and cover its whole box; large static regions are scenery.
    mask = np.asarray(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15)))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filled = np.zeros_like(mask)
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if w * h <= config.max_overlay_box * mask.size and area >= config.min_overlay_pixels:
            filled[y : y + h, x : x + w] = 1
    grown = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return np.asarray(grown, dtype=bool)


def osd_boxes(osd: BoolMask) -> BoolMask:
    """Each OSD element's whole box (its strokes joined, a margin around them): what to
    leave out when judging the picture itself, never what to leave uncleaned (a box of
    untouched noise shows)."""
    joined = np.asarray(
        cv2.morphologyEx(
            osd.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15)),
        )
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = np.zeros_like(joined)
    for x, y, w, h, _ in stats[1:]:
        boxes[y : y + h, x : x + w] = 1
    grown = cv2.dilate(boxes, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return np.asarray(grown, dtype=bool)


def osd_alpha(osd: BoolMask) -> NDArray[np.float32]:
    """How much of each pixel is OSD: 1 on the mask, falling to 0 over ``OSD_FEATHER`` px
    outside it, so the switch between the untouched OSD and the cleaned picture is soft and
    no OSD pixel is ever partly cleaned."""
    if not osd.any():
        return np.zeros(osd.shape, np.float32)
    outside = cv2.distanceTransform((~osd).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - outside / (OSD_FEATHER + 1), 0.0, 1.0).astype(np.float32)


STANDARD_ASPECTS = (4 / 3, 16 / 9)
ASPECT_SNAP = 0.03  # a detected picture within 3 % of a standard shape is that shape


SD_SIZES = ((720, 480), (704, 480), (720, 576), (704, 576))  # analog SD storage


def output_size(
    crop: Crop, height: int = 1080, sar: float = 1.0, frame: tuple[int, int] | None = None
) -> tuple[int, int]:
    """The picture's own shape at ``height`` lines: 4:3 -> 1440x1080, 16:9 -> 1920x1080.

    The shape is the one shown, not the one stored: ``sar`` is the file's pixel shape. SD
    DVRs store a 4:3 picture in 720x480 (NTSC) or 720x576 (PAL) and do not always say so;
    a ``frame`` of such a size with square pixels is taken as 4:3.
    """
    aspect = crop.width * sar / crop.height
    if sar == 1.0 and frame in SD_SIZES:
        aspect *= (4 / 3) / (frame[0] / frame[1])
    for standard in STANDARD_ASPECTS:
        if abs(aspect / standard - 1.0) <= ASPECT_SNAP:
            aspect = standard
    width = round(height * aspect / 2) * 2
    return width, height
