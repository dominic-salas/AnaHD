"""Crop detection: the picture's own area, found from the frames that hold a picture."""

from __future__ import annotations

import cv2
import numpy as np

from anahd.picture import detect_crop

H, W = 360, 640
PIC_X, PIC_W = 80, 480  # a 4:3 picture between black bars


def _scene(rng: np.random.Generator, width: int) -> np.ndarray:
    smooth = cv2.GaussianBlur(rng.uniform(30, 220, (H, width)).astype(np.float32), (0, 0), 6)
    return np.clip(smooth, 8, 255).astype(np.uint8)


def _picture(rng: np.random.Generator) -> np.ndarray:
    frame = np.zeros((H, W), np.uint8)
    frame[:, PIC_X : PIC_X + PIC_W] = _scene(rng, PIC_W)
    return frame


def _static(rng: np.random.Generator) -> np.ndarray:
    """A full-frame screen with no picture: grey noise over every column."""
    return np.clip(rng.normal(127, 12, (H, W)), 0, 255).astype(np.uint8)


def _black() -> np.ndarray:
    return np.ones((H, W), np.uint8)


def test_a_recording_mostly_static_still_finds_its_picture() -> None:
    rng = np.random.default_rng(1)
    frames = [_picture(rng) for _ in range(30)]
    frames += [_static(rng) for _ in range(50)] + [_black() for _ in range(10)]
    crop = detect_crop(frames)
    assert abs(crop.x - PIC_X) <= 4 and abs(crop.width - PIC_W) <= 8
    assert (crop.y, crop.height) == (0, H)


def test_a_full_frame_picture_stays_full_frame() -> None:
    rng = np.random.default_rng(2)
    crop = detect_crop([_scene(rng, W) for _ in range(20)])
    assert (crop.x, crop.width, crop.y, crop.height) == (0, W, 0, H)


def test_all_static_falls_back_to_the_whole_frame() -> None:
    rng = np.random.default_rng(3)
    crop = detect_crop([_static(rng) for _ in range(20)])
    assert (crop.x, crop.width, crop.height) == (0, W, H)
