# Changelog

## 0.1.2 (2026-10-28)

- **Watch the clean as it happens.** While a clip is cleaned, the preview shows the frame
  just cleaned against the recording's, updated every second and a half: drag the line
  across it as on the Results page. The frame number and the point in the clip are under
  it, with a Stop right there, so a setting that looks wrong in motion costs a minute, not
  the whole clip. Pressing Clean shows the clip being cleaned, and the view follows the
  batch from clip to clip unless you have picked another clip to look at.
- Before the first frame, the clip's card says what is happening (counting frames,
  measuring the picture area and OSD, loading the models) instead of "Starting...".
- The header and the tray say when the whole batch should be done ("done about 6:41 PM"),
  from the speed measured on your own GPU.
- While a clip is cleaned, other clips show their plain frame: the before/after preview
  waits, so it never competes with the clean for the GPU.
- The output is unchanged, bit for bit.

## 0.1.1 (2026-10-22)

- The app's look now matches dvsalas.com: charcoal surfaces, an amber accent, Space
  Grotesk and Geist Mono (bundled, SIL Open Font License), sentence-case labels. Layout
  and features are unchanged.
- Outputs keep the recording's sound, in sync with the picture after cuts, joins and
  parts (the DVR's late-starting sound is padded with silence, not slid early).
- `.ts` recordings no longer fail to start ("could not count frames").
- The settings panel is never cut off on the right.

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
