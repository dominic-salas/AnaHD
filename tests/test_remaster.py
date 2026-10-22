"""The remaster settings and stage (anahd.remaster): defaults are the approved C03 look,
every part is the identity at 0, the sliders do what their labels say, measured against the
known clean picture of synthetic recordings (tests/synthetic.py), not against AnaHD itself.
Needs torch and a CUDA GPU for most tests; the end-to-end ones take about two minutes.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from anahd.remaster.params import GROUPS, NO_LOOK, RemasterParams, schema

# ----- the settings themselves (no GPU) -----

APPROVED = {  # the look approved 2026-10-21, on by default
    "cleanup_strength": 100,
    "scan_lines": 100,
    "noise": 100,
    "static_repair": 100,
    "colour_cleanup": 100,
    "stabilisation": 0,
    "sharpen": 0,
    "lens_colour_fix": 100,
    "warmth": 0,
    "tint": 0,
    "saturation": 100,
    "contrast": 100,
    "black_level": 0,
    "shadows": 100,
    "look_strength": 100,
    "vignette": 100,
    "bloom": 100,
    "motion_blur": 0,
    "superview": "off",
    "osd_redraw": False,
    "osd_redraw_live": False,
    "osd_font_path": "",
    "osd_cli_path": "",
}


def test_defaults_are_the_approved_look_with_effects_off() -> None:
    assert RemasterParams().to_dict() == APPROVED


def test_settings_round_trip_and_carry_what_the_interface_needs() -> None:
    p = RemasterParams(warmth=-35, superview="osd_out", osd_redraw=True)
    assert RemasterParams.from_dict(p.to_dict()) == p
    assert RemasterParams.from_json(p.to_json()) == p
    assert RemasterParams.from_dict({"warmth": 10, "from_the_future": 1}).warmth == 10
    for item in schema():
        assert item["group"] in GROUPS and item["label"] and item["help"], item
        if item["kind"] == "slider":
            assert item["min"] <= item["default"] <= item["max"] and item["step"] > 0
            assert "unit" in item
    with pytest.raises(ValueError):
        RemasterParams(saturation=250)
    with pytest.raises(ValueError):
        RemasterParams(superview="fisheye")


# ----- the stage on synthetic planes (GPU) -----

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
W, H = 1440, 1080
TEXT, ORG = "12.6V 03:21", (60, 1010)


def _scene(i: int) -> NDArray[np.float32]:
    """A moving, colourful scene (BGR, 0..1) at the output size."""
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(
        rng.uniform(0, 1, (H // 8, W // 8 + 40, 3)).astype(np.float32), (0, 0), 2
    )
    big = cv2.resize(base, (W + 320, H), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 0.08, (H // 2, (W + 320) // 2, 1)).astype(np.float32)  # texture
    big = big + cv2.resize(fine, (W + 320, H), interpolation=cv2.INTER_CUBIC)[..., None]
    return np.ascontiguousarray(np.clip(big[:, 4 * i : 4 * i + W], 0, 1))


def _with_osd(bgr: NDArray[np.float32]) -> NDArray[np.float32]:
    out = bgr.copy()
    outline = np.zeros((H, W), np.uint8)
    cv2.putText(outline, TEXT, ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 9)
    out[outline > 0] = 0.0
    out[_strokes()] = 1.0
    return out


def _strokes() -> NDArray[np.bool_]:
    m = np.zeros((H, W), np.uint8)
    cv2.putText(m, TEXT, ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 3)
    return m > 0


def _planes(bgr: NDArray[np.float32]):  # type: ignore[no-untyped-def]
    from anahd.remaster.grade import to_ycc

    rgb = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).cuda()
    y, cb, cr = to_ycc(rgb)
    pool = torch.nn.functional.avg_pool2d
    return y, pool(cb[None, None], 2)[0, 0], pool(cr[None, None], 2)[0, 0]


def _run_stage(  # type: ignore[no-untyped-def]
    p: RemasterParams, frames: int = 5, centre: int = 2, chroma_noise: float = 0.0
):
    """The stage over ``frames`` synthetic frames (static OSD on a scene panning 4 px a
    frame, with the true flow); returns the centre frame's R'G'B' and the input's."""
    from anahd.remaster.grade import to_rgb
    from anahd.remaster.stage import Frame, Remaster

    box = np.zeros((360, 480), bool)  # the OSD box at the source size, as detect_osd gives
    ys_, xs_ = np.nonzero(_strokes())
    ys, xs = ys_.tolist(), xs_.tolist()
    box[(min(ys) - 12) // 3 : (max(ys) + 12) // 3, (min(xs) - 12) // 3 : (max(xs) + 12) // 3] = 1
    stage = Remaster((W, H), torch.from_numpy(box).cuda(), p, torch.device("cuda"))
    stage.capture = centre
    inputs = []
    g = torch.Generator(device="cuda").manual_seed(2)
    for i in range(frames):
        y, u, v = _planes(_with_osd(_scene(i)))
        if chroma_noise and i == centre:  # what the clean-up should take out of this frame
            truth_uv = (u.clone(), v.clone())
        if chroma_noise:
            u = u + chroma_noise * torch.randn(u.shape, device="cuda", generator=g)
            v = v + chroma_noise * torch.randn(v.shape, device="cuda", generator=g)
        # The scene moves 4 px left a frame: centre -> neighbour d is -2d chroma pixels.
        step = torch.tensor([-2.0, 0.0], device="cuda").view(2, 1, 1)
        flow = {d: (d * step).expand(2, H // 2, W // 2) for d in (-2, -1, 1, 2)}
        uv = torch.stack((u, v))[None]
        full = torch.nn.functional.interpolate(uv, size=(H, W), mode="bilinear")[0]
        inputs.append(to_rgb(y, full[0], full[1]))
        stage.push(Frame(y, u, v, flow))
    stage.flush()
    assert stage.captured is not None
    if chroma_noise:
        stage.truth_uv = truth_uv  # type: ignore[attr-defined]
    return stage.captured, inputs[centre].clamp(0, 1), stage


OFF = NO_LOOK.with_(colour_cleanup=0, lens_colour_fix=0, look_strength=0)


@needs_gpu
def test_every_part_is_the_identity_at_0() -> None:
    out, inp, _ = _run_stage(OFF)
    assert float((out - inp).abs().max()) < 1e-5
    for name, on in [
        ("lens_colour_fix", 100), ("look_strength", 100),
        ("sharpen", 100), ("motion_blur", 39),
    ]:  # fmt: skip
        changed, _, _ = _run_stage(OFF.with_(**{name: on}))
        assert float((changed - inp).abs().mean()) > 1e-3, f"{name} does nothing"


@needs_gpu
def test_colour_cleanup_removes_colour_noise_and_leaves_brightness() -> None:
    """Chroma noise added to every frame (the truth known): the clean-up brings the centre's
    colour much closer to the truth and does not touch luma."""
    from anahd.remaster.grade import to_ycc

    noisy, _, stage = _run_stage(OFF, chroma_noise=0.03)
    cleaned, _, _ = _run_stage(OFF.with_(colour_cleanup=100), chroma_noise=0.03)
    pool = torch.nn.functional.avg_pool2d
    tu, tv = stage.truth_uv

    def err(rgb: Tensor) -> float:
        _, cb, cr = to_ycc(rgb)
        return float((pool(cb[None, None], 2)[0, 0] - tu).abs().mean()
                     + (pool(cr[None, None], 2)[0, 0] - tv).abs().mean())  # fmt: skip

    print(f"\ncolour error vs truth: noisy {err(noisy):.4f}, cleaned {err(cleaned):.4f}")
    assert err(cleaned) < 0.6 * err(noisy)
    ly = to_ycc(noisy)[0]
    assert float((to_ycc(cleaned)[0] - ly).abs().max()) < 0.02


def test_cleaner_sliders_at_0_give_back_the_plain_frame() -> None:
    """The clean-up mix: 0 = the plain centre frame; scan_lines/noise = 0 put back exactly
    the row-structured / the other part of what the restorer took out."""
    from anahd.gpu.pipeline import Cleaner

    g = torch.Generator().manual_seed(1)
    centre = torch.rand(3, 120, 160, generator=g) * 255
    rows = torch.randn(1, 120, 1, generator=g).expand(1, 120, 160) * 6  # stripes
    grain = torch.randn(1, 120, 160, generator=g) * 4
    noisy = centre + torch.cat((rows + grain, torch.zeros(2, 120, 160)))
    cleaned = centre  # a restorer that removed exactly the stripes and the grain

    def mix(**kw: float) -> Tensor:
        p = RemasterParams.from_dict(kw)
        fake = SimpleNamespace(params=p, cleanup=p.cleanup_strength / 100)
        return Cleaner._mix(fake, cleaned, noisy)  # type: ignore[arg-type]

    assert torch.allclose(mix(cleanup_strength=0), noisy)
    assert torch.allclose(mix(), cleaned)
    back = mix(scan_lines=0)[0] - centre[0]  # the stripes, not the grain
    inner = (slice(20, 100), slice(70, 90))  # away from the filters' edges
    assert float(np.corrcoef(back[inner].ravel(), rows[0][inner].ravel())[0, 1]) > 0.9
    assert float(back[inner].std()) < 1.3 * float(rows[0][inner].std())
    back = mix(noise=0)[0] - centre[0]
    assert float(np.corrcoef(back[inner].ravel(), grain[0][inner].ravel())[0, 1]) > 0.9


@needs_gpu
def test_look_strength_scales_the_look_monotonically() -> None:
    base, _, _ = _run_stage(OFF)
    steps = []
    for s in (25, 50, 75, 100):
        out, _, _ = _run_stage(OFF.with_(look_strength=s))
        steps.append(float((out - base).abs().mean()))
    assert all(a < b for a, b in itertools.pairwise(steps)), steps
    assert steps[-1] > 0.02, steps


@needs_gpu
def test_osd_pixels_are_not_graded_and_no_box_shows_around_them() -> None:
    out, inp, _ = _run_stage(RemasterParams())
    core = torch.from_numpy(cv2.erode(_strokes().astype(np.uint8), np.ones((3, 3)))).cuda() > 0
    assert float((out - inp).abs().amax(0)[core].max()) < 2 / 255, "the OSD was graded"
    ys_, xs_ = np.nonzero(_strokes())
    ys, xs = ys_.tolist(), xs_.tolist()
    box = torch.zeros(H, W, dtype=torch.bool, device="cuda")
    box[min(ys) - 8 : max(ys) + 8, min(xs) - 8 : max(xs) + 8] = True
    grown = torch.from_numpy(cv2.dilate(_strokes().astype(np.uint8), np.ones((15, 15)))).cuda() > 0
    behind = box & ~grown  # inside the OSD box, away from the text: the moving scene
    outside = torch.zeros_like(box)
    outside[min(ys) - 80 : min(ys) - 20, min(xs) : max(xs)] = True
    graded_behind = float((out - inp).abs().mean(0)[behind].mean())
    graded_outside = float((out - inp).abs().mean(0)[outside].mean())
    assert graded_behind > 0.6 * graded_outside, "the scene behind the OSD is left ungraded"


@needs_gpu
def test_superview_is_1920x1080_natural_in_the_centre_stretched_at_the_edges() -> None:
    from anahd.remaster.stage import superview_grid

    out, _, stage = _run_stage(OFF.with_(superview="stretch"))
    assert stage.out_size == (1920, 1080) and tuple(out.shape) == (3, 1080, 1920)
    gx = superview_grid(W, H, "stretch", torch.device("cuda"))[0, 540, :, 0]  # type: ignore[index]
    src_px = (gx[1:] - gx[:-1]) * (W - 1) / 2  # source pixels per output pixel
    centre = float(src_px[958:962].mean())
    edge = float(src_px[:20].mean())
    assert 0.9 <= 1 / centre <= 1.1, f"centre stretched {1 / centre:.2f}x"
    assert 1 / edge > 1.6, f"edges stretched only {1 / edge:.2f}x"
    tall, _, _ = _run_stage(OFF.with_(superview="osd_out"))
    assert not torch.allclose(tall, out), "osd_out does not push the corners out"


# ----- end to end on synthetic recordings (slow) -----

import synthetic as syn  # noqa: E402

if torch.cuda.is_available():
    from anahd import cli
    from test_requirements import fit, residual_motion


def _striped(path: Path, frames: int) -> Path:
    """syn.clean_frame recorded like syn.write_with, plus analog line noise: every picture
    row offset by its own few levels, new in each frame."""
    import subprocess

    rng = np.random.default_rng(5)
    lay = syn.FOUR_THREE
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    cmd += ["-s", f"{lay.dvr_w}x{lay.dvr_h}", "-r", str(syn.FPS), "-i", "-", "-c:v", "libx264"]
    cmd += ["-crf", "10", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(frames):
        frame = syn.degrade(syn.clean_frame(i), lay, rng, band=False).astype(np.float32)
        offsets = rng.normal(0, 6.0, (lay.pic_h, 1, 1))
        frame[:, lay.pic_x : lay.pic_x + lay.pic_w] += offsets
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def _stripes_and_grain(path: Path, frames: int, vf: str = "") -> tuple[float, float]:
    """Against the truth: the row-structured error (std of the row means of the error,
    high-passed across rows) and the grain (error std in the flat patch, rows removed)."""
    size = (syn.OUT_W, syn.OUT_H)
    stripes, grain = [], []
    x, y, w, h = syn.FLAT
    for i, out in enumerate(syn.read_bgr(path, size, vf)):
        if i < 4 or i >= frames - 4:
            continue
        truth = syn.clean_frame(i)
        err = cv2.cvtColor(np.clip(fit(out, truth), 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        err = err.astype(np.float64) - cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
        rows = err[40:940, 40:1400].mean(axis=1)
        smooth = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 0), sigmaX=0.1, sigmaY=8).ravel()
        stripes.append(float(np.std(rows - smooth)))
        patch = err[y : y + h, x : x + w]
        grain.append(float((patch - patch.mean(axis=1, keepdims=True)).std()))
    return float(np.mean(stripes)), float(np.mean(grain))


@needs_gpu
@pytest.mark.gpu
@pytest.mark.slow
def test_scan_line_and_noise_sliders_put_back_what_they_say(tmp_path: Path) -> None:
    frames = 90
    src = _striped(tmp_path / "striped.mp4", frames)
    plain = _stripes_and_grain(src, frames, syn.plain_upscale_filter())
    got = {}
    for name, extra in [("default", []), ("scan0", ["--look-scan-lines", "0"]),
                        ("noise0", ["--look-noise", "0"])]:  # fmt: skip
        out = tmp_path / f"{name}.mp4"
        assert cli.main([str(src), "-o", str(out), *extra]) == 0
        got[name] = _stripes_and_grain(out, frames)
    print(
        f"\nstripes / grain vs truth: plain {plain}, "
        + ", ".join(f"{k} {v}" for k, v in got.items())
    )
    (sd, gd), (s0, g_s0), (s_n0, gn0) = got["default"], got["scan0"], got["noise0"]
    assert sd < 0.5 * plain[0], "the stripes are not removed by default"
    assert s0 > 0.6 * plain[0] and s0 > 2 * sd, "scan_lines=0 does not bring the stripes back"
    assert g_s0 < 1.5 * gd, "scan_lines=0 also brought the grain back"
    assert gn0 > 0.6 * plain[1] and gn0 > 2 * gd, "noise=0 does not bring the grain back"
    assert s_n0 < 0.6 * plain[0], "noise=0 also brought the stripes back"


@needs_gpu
@pytest.mark.gpu
@pytest.mark.slow
def test_preview_matches_the_same_frame_of_a_full_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preview of 1.0 s is the render's frame 60, compared before the encoder (the H.264
    encode of this very grainy scene alone moves pixels by ~3.5 levels on average)."""
    from anahd.gpu import pipeline as gp
    from anahd.remaster.grade import to_rgb
    from anahd.remaster.preview import PreviewCache, preview_frame

    clip = syn.write(tmp_path / "flight.mp4", 120)
    sent: list[Tensor] = []
    put = gp._Writer.put

    def keep(self: gp._Writer, packed: Tensor) -> None:
        sent.append(packed.clone())
        put(self, packed)

    monkeypatch.setattr(gp._Writer, "put", keep)
    assert cli.main([str(clip.path)]) == 0
    b = sent[60].float()
    y = (b[: W * H].view(H, W) - 16) / 219
    q = W * H // 4
    uv = torch.stack(((b[W * H : W * H + q] - 128) / 224, (b[W * H + q :] - 128) / 224))
    uv = torch.nn.functional.interpolate(
        uv.view(1, 2, H // 2, W // 2), size=(H, W), mode="bilinear"
    )
    rendered = (to_rgb(y, uv[0, 0], uv[0, 1]).clamp(0, 1) * 255).round().permute(1, 2, 0)
    cache = PreviewCache()
    pv = preview_frame(clip.path, 1.0, RemasterParams(), cache)
    again = preview_frame(clip.path, 1.0, RemasterParams(warmth=30), cache)
    diff = np.abs(pv.result.astype(np.float64) - rendered.cpu().numpy())
    print(
        f"\npreview vs render: mean {diff.mean():.2f}, p99 {np.percentile(diff, 99):.1f}; "
        f"timings cold {pv.timings}, warm {again.timings}"
    )
    assert pv.result.shape == pv.source.shape == (1080, 1440, 3)
    assert diff.mean() < 1.0 and np.percentile(diff, 99) < 8
    assert sum(again.timings.values()) < 1500, "a warm preview takes over 1.5 s"


@needs_gpu
@pytest.mark.gpu
@pytest.mark.slow
def test_full_smoothness_never_damps_a_720_deg_s_flip(tmp_path: Path) -> None:
    """Stabilisation at 100: through the 180 deg/s roll and the 720 deg/s flip, every frame
    still shows the camera's own angle (as the default output must, test_requirements)."""
    from anahd.media import probe
    from anahd.pipeline import Options, measure

    n = 120
    clip = syn.write_with(tmp_path / "roll.mp4", n, syn.roll_truth, syn.FOUR_THREE)
    out = tmp_path / "AnaHD_roll.mp4"
    assert cli.main([str(clip.path), "--look-stabilisation", "100"]) == 0
    c = measure(clip.path, probe(clip.path), Options()).crop
    area = (c.x, c.y, c.width, c.height)
    outs = list(syn.read_bgr(out, (syn.OUT_W, syn.OUT_H)))
    moving = [
        i
        for i in [*syn.ROLL, *syn.FLIP]
        if syn.roll_angle(i) not in (syn.roll_angle(i - 1), syn.roll_angle(i + 1))
    ]
    angles = []
    for i in moving:
        truth = syn.reframe(syn.roll_truth(i), *area)
        w = residual_motion(truth, fit(outs[i], truth), cv2.MOTION_EUCLIDEAN)
        angles.append(abs(float(np.degrees(np.arctan2(w[1, 0], w[0, 0])))))
    print(f"\nroll residual with smoothness 100: max {max(angles):.2f} deg")
    assert max(angles) <= 0.5


@needs_gpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("look", [True, False])
def test_classic_cleaning_works_with_the_look_on_and_off(tmp_path: Path, look: bool) -> None:
    clip = syn.write(tmp_path / "flight.mp4", 60)
    out = tmp_path / "AnaHD_flight.mp4"
    assert cli.main([str(clip.path), "--no-learned", *([] if look else ["--no-look"])]) == 0
    outs = list(syn.read_bgr(out, (syn.OUT_W, syn.OUT_H)))
    assert len(outs) == 60
    x, y, w, h = syn.FLAT
    plains = list(syn.read_bgr(clip.path, (syn.OUT_W, syn.OUT_H), syn.plain_upscale_filter()))
    noise = [
        (fit(o, t)[y : y + h, x : x + w].std(axis=(0, 1)).mean(),
         fit(p, t)[y : y + h, x : x + w].std(axis=(0, 1)).mean())
        for i, (o, p) in enumerate(zip(outs, plains, strict=True))
        if not syn.damaged(i)
        for t in [syn.clean_frame(i)]
    ]  # fmt: skip
    out_noise, plain_noise = (float(np.mean(v)) for v in zip(*noise, strict=True))
    assert out_noise <= 0.6 * plain_noise


# ----- the defaults look like the approved renders (needs the dev clips) -----

APPROVED_LOOK = Path(__file__).parent / "data" / "approved_look.json"
DEV_CLIPS = Path(os.environ.get("ANAHD_DEV_CLIPS", "dev_clips"))
# What the retuned defaults reach (2026-10-21, analog_look/recheck): the corrected pipeline
# renders dark clips' shadows brighter than the approved (double-converted) renders, and no
# setting in the look moves the black level, hence the wider luma tolerance.
LOOK_TOLERANCE = {"p5": 0.03, "p50": 0.03, "p95": 0.02, "sat": 0.006, "cb": 0.015, "cr": 0.015}


def _appearance(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    x = cv2.resize(
        cv2.GaussianBlur(rgb.astype(np.float32) / 255, (0, 0), 3),
        (360, 270),
        interpolation=cv2.INTER_AREA,
    )
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.stack((y, (b - y) / 1.8556, (r - y) / 1.5748), -1)


@needs_gpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not DEV_CLIPS.is_dir(), reason="the dev clips are not on this machine")
def test_defaults_look_like_the_renders_dom_approved() -> None:
    """The default output, on the frames of the dev clips the look was approved on,
    measured against those approved renders: tone (luma quantiles), saturation and mean
    colour within LOOK_TOLERANCE."""
    import json

    from anahd.remaster.preview import PreviewCache, preview_frame

    want = json.loads(APPROVED_LOOK.read_text(encoding="utf-8"))
    cache = PreviewCache(keep=8)
    for clip, ref in want["clips"].items():
        frames = [
            _appearance(preview_frame(DEV_CLIPS / f"{clip}.mp4", t, RemasterParams(), cache).result)
            for t in want["times"]
        ]
        a = np.stack(frames)
        y, cb, cr = a[..., 0], a[..., 1], a[..., 2]
        p5, p50, p95 = np.percentile(y, [5, 50, 95])
        got = {"p5": p5, "p50": p50, "p95": p95, "sat": np.hypot(cb, cr).mean(),
               "cb": cb.mean(), "cr": cr.mean()}  # fmt: skip
        off = {k: float(got[k] - ref[k]) for k in LOOK_TOLERANCE}
        print(f"\n{clip}: " + ", ".join(f"{k} {v:+.4f}" for k, v in off.items()))
        for k, v in off.items():
            assert abs(v) <= LOOK_TOLERANCE[k], f"{clip}: {k} off by {v:+.4f}"


def test_black_level_lifts_or_crushes_the_shadows_and_leaves_them_alone_by_default() -> None:
    """Shadows must not get darker by default, there should be more to see
    in them, not less. 0 changes nothing; lift raises the shadows and keeps their local
    contrast visible; crush darkens them and costs shadow detail."""
    from anahd.remaster.stage import black_level

    g = torch.Generator().manual_seed(4)
    shadow = 0.05 + 0.1 * torch.rand(3, 64, 64, generator=g)  # dark, textured
    assert RemasterParams().black_level == 0

    def detail(x: Tensor) -> float:  # local contrast: what is left to see
        blur = torch.nn.functional.avg_pool2d(x[None], 5, 1, 2)[0]
        return float((x - blur).abs().mean())

    lifted, crushed = black_level(shadow, -100), black_level(shadow, 100)
    assert float(lifted.mean()) > float(shadow.mean()) > float(crushed.mean())
    assert detail(lifted) >= 0.9 * detail(shadow)  # lift keeps the shadow detail
    assert detail(crushed) < detail(shadow)  # crush loses some (clipped to black)
    white = torch.ones(3, 4, 4)
    assert torch.allclose(black_level(white, -100), white) and torch.allclose(
        black_level(white, 100), white
    )


@needs_gpu
def test_the_default_black_level_changes_nothing() -> None:
    out, _, _ = _run_stage(RemasterParams())
    same, _, _ = _run_stage(RemasterParams(black_level=0))
    assert torch.equal(out, same)
