# Changelog

## 0.1.0 (2026-10-22)

First public version.

- Cleaning: breakup and colour-snow repair from aligned neighbouring frames; the Learned
  restorer (v6, a 5-frame U-Net distilled from SeedVR2 7B on the author's footage) or the
  Classic hand-made cleaning; OSD protection; upscale to 1440x1080 (4:3) or 1920x1080
  (16:9) at the recording's frame rate.
- Any DVR format FFmpeg reads, including SD sizes with non-square pixels and MJPEG.
- Desktop app: preview, cut markers, one file per clip or per kept part, digital-recording
  check, before/after results, tray while working, never quits by itself.
- Remaster settings: the C03 lens colour fix and a film-style look (on by default), colour
  sliders, and optional stabilisation, sharpening, SuperView and motion blur, with a
  before/after preview of the frame, each effect's speed cost and a speed estimate.
- OSD redraw from a Betaflight font (experimental, off by default).
- Command line: `anahd`, with `--cut`, `--parts`, `--no-learned`, `--look-*` and more.
- Windows installer with FFmpeg, PyTorch's CUDA libraries and the model. Runs on GeForce
  GTX 16 / RTX 20 series cards and newer; an older card is named with a clear message.
