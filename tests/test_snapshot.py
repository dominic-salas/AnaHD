"""The live view's pictures: colour conversion, the throttle, and (on a GPU) that taking
pictures leaves the cleaned output exactly as it was."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from anahd import snapshot as snap


def _i420(y: int, u: int, v: int, w: int = 64, h: int = 48) -> np.ndarray:
    luma, chroma = w * h, (w // 2) * (h // 2)
    buf = np.empty(luma + 2 * chroma, np.uint8)
    buf[:luma], buf[luma : luma + chroma], buf[luma + chroma :] = y, u, v
    return buf


@pytest.mark.parametrize("bt709", [False, True])
def test_limited_range_black_white_and_grey_stay_neutral(bt709: bool) -> None:
    for code, want in [(16, 0), (235, 255), (126, 128)]:
        bgr = snap.to_bgr(_i420(code, 128, 128), 64, 48, full_range=False, bt709=bt709)
        assert np.abs(bgr.astype(int) - want).max() <= 1, (code, bgr[0, 0])


def test_full_range_is_not_stretched() -> None:
    bgr = snap.to_bgr(_i420(200, 128, 128), 64, 48, full_range=True, bt709=False)
    assert np.abs(bgr.astype(int) - 200).max() <= 1


def test_matrix_matters_for_colour() -> None:
    """Pure red in BT.601 limited range (Y 81, Cb 90, Cr 240) is red only with the BT.601
    matrix; read as BT.709 it shifts, which is what the two sides of the view must not do."""
    red601 = snap.to_bgr(_i420(81, 90, 240), 64, 48, full_range=False, bt709=False)[0, 0]
    assert red601[2] >= 250 and red601[0] <= 6 and red601[1] <= 6, red601
    as709 = snap.to_bgr(_i420(81, 90, 240), 64, 48, full_range=False, bt709=True)[0, 0]
    assert np.abs(as709.astype(int) - red601.astype(int)).max() > 10


def test_size_keeps_the_shape_and_never_upscales() -> None:
    assert snap.to_bgr(
        _i420(100, 128, 128, 1440, 1080), 1440, 1080, full_range=False, bt709=True
    ).shape == (480, 640, 3)
    assert snap.to_bgr(_i420(100, 128, 128), 64, 48, full_range=False, bt709=True).shape == (
        48,
        64,
        3,
    )


def test_a_short_buffer_is_refused() -> None:
    with pytest.raises(ValueError, match="less than one"):
        snap.to_bgr(np.zeros(100, np.uint8), 64, 48, full_range=False, bt709=True)


def test_jpeg_round_trips() -> None:
    img = np.full((48, 64, 3), (20, 120, 220), np.uint8)
    back = cv2.imdecode(np.frombuffer(snap.jpeg(img), np.uint8), cv2.IMREAD_COLOR)
    assert back is not None
    assert back.shape == img.shape and np.abs(back.astype(int) - img).max() <= 4


def test_throttle_first_then_every_interval() -> None:
    t = snap.Throttle(every=1.5)
    due = [t.due(now) for now in (10.0, 10.5, 11.4, 11.5, 12.0, 13.0, 13.1)]
    assert due == [True, False, False, True, False, True, False]


# ----- on a GPU: the pictures leave the output alone -----


def _framemd5(path: Path) -> str:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return hashlib.sha256(
        "\n".join(line for line in out.splitlines() if not line.startswith("#")).encode()
    ).hexdigest()


@pytest.mark.gpu
def test_live_pictures_change_no_output_frame(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    import synthetic as syn
    from anahd.gpu.pipeline import enhance
    from anahd.pipeline import Options

    clip = syn.write(tmp_path / "dvr.mp4", 90)
    plain, watched = tmp_path / "plain.mp4", tmp_path / "watched.mp4"
    enhance(clip.path, plain, Options(), overwrite=True)
    pictures: list[tuple[bytes, bytes, int]] = []
    stages: list[str] = []
    enhance(
        clip.path,
        watched,
        Options(),
        overwrite=True,
        stage_report=stages.append,
        picture=lambda b, a, n: pictures.append((b, a, n)),
    )
    assert _framemd5(plain) == _framemd5(watched), "taking pictures changed the output"
    assert pictures and pictures[0][2] == 0, "the first output frame is always pictured"
    assert [n for _, _, n in pictures] == sorted({n for _, _, n in pictures})
    before = cv2.imdecode(np.frombuffer(pictures[0][0], np.uint8), cv2.IMREAD_COLOR)
    after = cv2.imdecode(np.frombuffer(pictures[0][1], np.uint8), cv2.IMREAD_COLOR)
    assert before is not None and after is not None
    assert before.shape == after.shape and after.shape[1] == snap.WIDTH
    assert stages[-1] == "Cleaning" and "Loading the models" in stages
