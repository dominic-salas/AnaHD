"""A synthetic analog FPV recording with a known clean picture, for end-to-end tests.

The clean scene is rendered at the output size (1440x1080, 4:3) and drifts slowly so the
neighbouring frames need real alignment. The "recording" is what a DVR would store: the
scene reduced to 960x720 and softened like analog video, with luma and chroma noise,
breakup bands in some frames and a static white OSD, pillarboxed into a 1280x720 frame and
encoded at 60 fps. Every clean frame can be regenerated, so outputs are compared with the
truth, not with what the code happens to produce.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

OUT_W, OUT_H = 1440, 1080  # the clean truth, and what AnaHD should output
PIC_W, PIC_H = 960, 720  # the analog picture inside the DVR frame
DVR_W, DVR_H = 1280, 720
PIC_X = (DVR_W - PIC_W) // 2  # black bars left and right, like a 4:3 picture in a 16:9 DVR
FPS = 60
NOISE_SIGMA = 9.0  # luma noise, levels (a noisy analog link)
BAND_EVERY, BAND_ROWS = 29, (300, 336)  # breakup: these picture rows in every 29th frame
OSD_TEXT, OSD_ORG = "12.6V 03:21", (60, 1010)  # at output scale


@dataclass(frozen=True)
class Clip:
    path: Path
    frames: int


def _scene(seed: int = 7, h: int = OUT_H + 200, w: int = OUT_W + 300) -> NDArray[np.float32]:
    """A large clean BGR image with flat areas, gradients, edges and fine texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = 90 + 60 * (xx / w)  # sky-like gradients
    img[..., 1] = 110 + 40 * (yy / h)
    img[..., 2] = 120 + 30 * np.sin(xx / 260)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    img[h // 2 :] += 22 * texture[h // 2 :, :, None]  # "grass" in the lower half
    for _ in range(40):  # buildings, poles, signs: hard edges
        x, y = int(rng.integers(0, w - 200)), int(rng.integers(0, h - 200))
        bw, bh = int(rng.integers(30, 200)), int(rng.integers(30, 200))
        colour = rng.uniform(30, 230, 3)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colour.tolist(), -1)
    for _ in range(25):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, c, int(rng.integers(10, 60)), rng.uniform(20, 240, 3).tolist(), -1)
    img[50:330, 60:520] = (120, 140, 150)  # a flat patch, free of edges, for noise checks
    return np.clip(img, 0, 255)


SCENE = _scene()
FLAT = (60, 60, 200, 120)  # x, y, w, h at output size: inside the flat patch for 300 frames


def _offset(i: int) -> tuple[float, float]:
    return 40 + 0.3 * i, 30 + 0.15 * i  # slow sub-pixel drift (300 frames: 90 x 45 px)


def clean_frame(i: int) -> NDArray[np.uint8]:
    """Frame ``i`` of the truth at 1440x1080, BGR, with the OSD."""
    ox, oy = _offset(i)
    m = np.array([[1, 0, -ox], [0, 1, -oy]], np.float32)
    frame = cv2.warpAffine(SCENE, m, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)
    out: NDArray[np.uint8] = np.clip(frame, 0, 255).astype(np.uint8)
    cv2.putText(out, OSD_TEXT, OSD_ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 9)
    cv2.putText(out, OSD_TEXT, OSD_ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return out


def osd_mask() -> NDArray[np.bool_]:
    """Where the OSD's white strokes are, at output size."""
    m = np.zeros((OUT_H, OUT_W), np.uint8)
    cv2.putText(m, OSD_TEXT, OSD_ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 3)
    return m > 0


def damaged(i: int) -> bool:
    return i % BAND_EVERY == BAND_EVERY // 2


def dvr_frame(i: int, rng: np.random.Generator) -> NDArray[np.uint8]:
    """What the DVR records for frame ``i``: 1280x720 BGR."""
    pic = cv2.resize(clean_frame(i), (PIC_W, PIC_H), interpolation=cv2.INTER_AREA)
    pic = cv2.GaussianBlur(pic.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=0.5)
    ycc8 = cv2.cvtColor(np.clip(pic, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    ycc: NDArray[np.float64] = ycc8.astype(np.float64)
    ycc[..., 0] += rng.normal(0, NOISE_SIGMA, (PIC_H, PIC_W))
    chroma = cv2.resize(rng.normal(0, 6, (PIC_H // 2, PIC_W // 2, 2)), (PIC_W, PIC_H))
    ycc[..., 1:] += chroma
    if damaged(i):  # a breakup band: rows of snow, shifted picture and colour garbage
        r0, r1 = BAND_ROWS
        ycc[r0:r1, :, 0] = rng.uniform(20, 235, (r1 - r0, PIC_W))
        ycc[r0:r1, :, 1:] = rng.uniform(60, 200, (r1 - r0, PIC_W, 2))
    pic = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    frame = np.zeros((DVR_H, DVR_W, 3), np.uint8)
    frame[:, PIC_X : PIC_X + PIC_W] = pic
    return frame


def write(path: Path, frames: int, seed: int = 1) -> Clip:
    """Encode the recording (H.264, 60 fps, near-lossless so the damage is as authored)."""
    rng = np.random.default_rng(seed)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    cmd += ["-s", f"{DVR_W}x{DVR_H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264"]
    cmd += ["-crf", "10", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(frames):
        proc.stdin.write(dvr_frame(i, rng).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg could not write the synthetic clip")
    return Clip(path, frames)


def read_bgr(path: Path, size: tuple[int, int], vf: str = "") -> Iterator[NDArray[np.uint8]]:
    """Decoded frames of ``path`` as BGR of ``size`` (after the filter ``vf``); FFmpeg
    applies each file's own colour matrix, so differently tagged files compare fairly."""
    w, h = size
    chain = f"{vf}," if vf else ""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"{chain}format=bgr24"]
    cmd += ["-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    n = w * h * 3
    while chunk := proc.stdout.read(n):
        if len(chunk) < n:
            break
        yield np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
    proc.wait()


def plain_upscale_filter(x: int = PIC_X, y: int = 0, w: int = PIC_W, h: int = PIC_H) -> str:
    """What an editor would do: crop the picture area, Lanczos to the output size."""
    return f"crop={w}:{h}:{x}:{y},scale={OUT_W}:{OUT_H}:flags=lanczos"


def reframe(
    truth: NDArray[np.uint8], x: int, y: int, w: int, h: int, layout: Layout | None = None
) -> NDArray[np.uint8]:
    """The truth as framed by a picture area of (x, y, w, h) in DVR pixels: AnaHD may trim
    a few edge pixels, and the comparison must use the same framing, not punish it."""
    lay = layout or Layout(OUT_W, OUT_H, PIC_W, PIC_H, PIC_X)
    sx, sy = lay.out_w / lay.pic_w, lay.out_h / lay.pic_h
    x0, y0 = (x - lay.pic_x) * sx, y * sy
    m = np.array([[w * sx / lay.out_w, 0, x0], [0, h * sy / lay.out_h, y0]], np.float32)
    framed = np.asarray(
        cv2.warpAffine(
            truth,
            m,
            (lay.out_w, lay.out_h),
            flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        ),
        np.uint8,
    )
    return framed


# ----- other shapes of recording: a fast roll and flip, and a full-frame 16:9 picture -----


@dataclass(frozen=True)
class Layout:
    """Where the analog picture sits in the DVR frame, and the HD size it maps to."""

    out_w: int
    out_h: int
    pic_w: int
    pic_h: int
    pic_x: int
    dvr_w: int = DVR_W
    dvr_h: int = DVR_H


FOUR_THREE = Layout(OUT_W, OUT_H, PIC_W, PIC_H, PIC_X)
WIDE = Layout(1920, 1080, DVR_W, DVR_H, 0)  # a 16:9 camera filling the DVR frame
# SD DVRs: the 4:3 picture fills a 720x480 (NTSC) or 720x576 (PAL) frame with tall or
# wide pixels; some others record square-pixel 640x480.
NTSC = Layout(OUT_W, OUT_H, 720, 480, 0, 720, 480)
PAL = Layout(OUT_W, OUT_H, 720, 576, 0, 720, 576)
VGA = Layout(OUT_W, OUT_H, 640, 480, 0, 640, 480)
_BIG: list[NDArray[np.float32]] = []


def big_scene() -> NDArray[np.float32]:
    """A square scene large enough to rotate a 1920x1080 window anywhere in it."""
    if not _BIG:
        _BIG.append(_scene(seed=11, h=2400, w=2400))
    return _BIG[0]


def _with_osd(frame: NDArray[np.float32]) -> NDArray[np.uint8]:
    out: NDArray[np.uint8] = np.clip(frame, 0, 255).astype(np.uint8)
    cv2.putText(out, OSD_TEXT, OSD_ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 9)
    cv2.putText(out, OSD_TEXT, OSD_ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return out


STILL, ROLL, FLIP = range(0, 30), range(30, 60), range(60, 90)  # frames; still after 90
ROLL_DEG, FLIP_DEG = 3.0, 12.0  # per frame: 180 deg/s, then a 720 deg/s flip (360 in 30)


def roll_angle(i: int) -> float:
    """Camera roll of frame ``i`` in degrees: still, a fast roll, a flip, still again."""
    roll = min(max(i - ROLL.start, 0), len(ROLL)) * ROLL_DEG
    flip = min(max(i - FLIP.start, 0), len(FLIP)) * FLIP_DEG
    return roll + flip


def roll_truth(i: int, layout: Layout = FOUR_THREE) -> NDArray[np.uint8]:
    """The scene rolled by :func:`roll_angle` about the picture centre; the OSD stays put."""
    scene = big_scene()
    cy, cx = scene.shape[0] / 2, scene.shape[1] / 2
    m = cv2.getRotationMatrix2D((cx, cy), roll_angle(i), 1.0)
    m[0, 2] -= cx - layout.out_w / 2
    m[1, 2] -= cy - layout.out_h / 2
    frame = cv2.warpAffine(scene, m, (layout.out_w, layout.out_h), flags=cv2.INTER_CUBIC)
    return _with_osd(np.asarray(frame, np.float32))


def wide_truth(i: int) -> NDArray[np.uint8]:
    """A slow drift like :func:`clean_frame`, at 1920x1080."""
    m = np.array([[1, 0, -(200 + 0.3 * i)], [0, 1, -(300 + 0.15 * i)]], np.float32)
    frame = cv2.warpAffine(big_scene(), m, (WIDE.out_w, WIDE.out_h), flags=cv2.INTER_CUBIC)
    return _with_osd(np.asarray(frame, np.float32))


def degrade(
    truth: NDArray[np.uint8], layout: Layout, rng: np.random.Generator, band: bool
) -> NDArray[np.uint8]:
    """What the DVR stores for a clean frame: reduced, softened, noisy, maybe a band."""
    w, h = layout.pic_w, layout.pic_h
    pic = cv2.resize(truth, (w, h), interpolation=cv2.INTER_AREA)
    pic = cv2.GaussianBlur(pic.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=0.5)
    ycc8 = cv2.cvtColor(np.clip(pic, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    ycc: NDArray[np.float64] = ycc8.astype(np.float64)
    ycc[..., 0] += rng.normal(0, NOISE_SIGMA, (h, w))
    ycc[..., 1:] += cv2.resize(rng.normal(0, 6, (h // 2, w // 2, 2)), (w, h))
    if band:
        r0, r1 = BAND_ROWS
        ycc[r0:r1, :, 0] = rng.uniform(20, 235, (r1 - r0, w))
        ycc[r0:r1, :, 1:] = rng.uniform(60, 200, (r1 - r0, w, 2))
    pic8 = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    frame = np.zeros((layout.dvr_h, layout.dvr_w, 3), np.uint8)
    frame[: layout.pic_h, layout.pic_x : layout.pic_x + w] = pic8
    return frame


def write_with(
    path: Path,
    frames: int,
    truth: Callable[[int], NDArray[np.uint8]],
    layout: Layout,
    *,
    fps: int = FPS,
    sar: str | None = None,
    mjpeg: bool = False,
) -> Clip:
    """Encode a recording of ``truth`` (no breakup bands) like :func:`write`. ``sar``
    tags the pixel shape (e.g. "10/11"); ``mjpeg`` records like older analog DVRs: MJPEG
    in 4:2:2, full range."""
    rng = np.random.default_rng(1)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    cmd += ["-s", f"{layout.dvr_w}x{layout.dvr_h}", "-r", str(fps), "-i", "-"]
    if sar:
        cmd += ["-vf", f"setsar={sar}"]
    if mjpeg:
        cmd += ["-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj422p", str(path)]
    else:
        cmd += ["-c:v", "libx264", "-crf", "10", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        cmd += [str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(frames):
        proc.stdin.write(degrade(truth(i), layout, rng, band=False).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg could not write the synthetic clip")
    return Clip(path, frames)


# ----- frames that carry their own number, to check exactly which frames come out -----

CODE_BITS, CODE_Y, CODE_X, CODE_SIZE, CODE_STEP = 10, 120, 360, 64, 76  # output pixels


def numbered_truth(i: int) -> NDArray[np.uint8]:
    """:func:`clean_frame` with frame number ``i`` drawn as 10 black/white blocks."""
    frame = clean_frame(i).copy()
    for bit in range(CODE_BITS):
        x = CODE_X + bit * CODE_STEP
        on = (i >> bit) & 1
        frame[CODE_Y : CODE_Y + CODE_SIZE, x : x + CODE_SIZE] = 235 if on else 20
    return frame


def read_number(frame: NDArray[np.uint8]) -> int:
    """The frame number in an output frame (block centres against mid-grey)."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m, n = CODE_SIZE // 4, 0
    for bit in range(CODE_BITS):
        x = CODE_X + bit * CODE_STEP
        centre = grey[CODE_Y + m : CODE_Y + CODE_SIZE - m, x + m : x + CODE_SIZE - m]
        n |= int(centre.mean() > 128) << bit
    return n


def without_osd(layout: Layout, speed: float = 1.0) -> Callable[[int], NDArray[np.uint8]]:
    """A recording's picture with no OSD, as a digital air unit records it, drifting
    ``speed`` times as fast as the other clips (1: 0.3 px a frame, nearly still; 10:
    3 px a frame, still slow for a flight)."""
    scene = big_scene()

    def frame(i: int) -> NDArray[np.uint8]:
        x, y = 200 + 0.3 * speed * i, 300 + 0.15 * speed * i
        m = np.array([[1, 0, -x], [0, 1, -y]], np.float32)
        f = cv2.warpAffine(scene, m, (layout.out_w, layout.out_h), flags=cv2.INTER_CUBIC)
        return np.asarray(np.clip(f, 0, 255), np.uint8)

    return frame


# ----- frames told apart by their content: the scene jumps to a new place every frame -----


def jumping_truth(i: int, layout: Layout = FOUR_THREE) -> NDArray[np.uint8]:
    """The scene 17 px further along every frame (unique for 900 frames), with the static
    OSD: each frame is recognisable by its content alone, whatever the cleaning does to
    detail, and nothing depends on what AnaHD takes for OSD."""
    x, y = 40 + (17 * i) % 900, 30 + 3 * (i % 5)
    m = np.array([[1, 0, -x], [0, 1, -y]], np.float32)
    f = cv2.warpAffine(big_scene(), m, (layout.out_w, layout.out_h), flags=cv2.INTER_CUBIC)
    return _with_osd(np.asarray(f, np.float32))


def identify(
    frames: Iterator[NDArray[np.uint8]], count: int, truth: Callable[[int], NDArray[np.uint8]]
) -> list[int]:
    """For each frame, the source frame (0..count-1) whose picture it matches best
    (small grey images, the OSD corner left out)."""

    def small(img: NDArray[np.uint8]) -> NDArray[np.float32]:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[: OSD_ORG[1] - 80]
        return np.asarray(cv2.resize(g, (180, 115), interpolation=cv2.INTER_AREA), np.float32)

    refs = np.stack([small(truth(k)) for k in range(count)])
    refs -= refs.mean(axis=(1, 2), keepdims=True)
    found = []
    for f in frames:
        s = small(f)
        s -= s.mean()
        found.append(int(np.argmin(np.abs(refs - s).mean(axis=(1, 2)))))
    return found
