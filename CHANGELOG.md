# Changelog

## [Unreleased]

- **The mouse side buttons move between pages.** Back goes from Results to Clips and Forward
  from Clips to Results (once a clip is cleaned), in the app and when it runs inside okFPV.

## 0.2.0 (2026-11-16)

- **Compiling is off.** Compiling the flow step (Triton) measured no faster than TensorRT
  alone on a 3-minute flight (671 s against 651 s), took 95 s for each new picture size and
  could not run in the installed app. It now runs only with `ANAHD_COMPILE=1` (a test
  option), Triton is an optional extra (`compile`) instead of a dependency, and the installer
  no longer carries it.
- **TensorRT's first build is announced.** When an engine for a new picture size is being
  built the status line says "Preparing TensorRT for this picture size (first time only)",
  in the app and on the command line (which now prints each stage as it starts). Engines
  already built are loaded without it.
- **The picture area is found from frames that hold a picture.** Black frames and static or
  noise screens no longer count when AnaHD looks for the black bars, so a recording that
  spends most of its time in breakup keeps its 4:3 picture instead of being taken as full
  frame. A recording whose camera was in a full-frame mode throughout still comes out full frame.
- **A quiet suggestion for the RaceCam R1 Mini.** A 4:3 recording at 50 fps looks like a camera
  the Learned cleaning was not trained on (it smears fine detail such as bare branches), so
  the app says Classic may keep fine detail better. It only suggests; nothing is switched.
- The README's before/after pictures are regenerated from the 1440-line output.
- **HDZero recordings are upscaled, not left out.** A recording judged HDZero skips the
  analog repair and cleaning (no alignment, no breakup repair, no restorer, no C03 lens or
  analog colour corrections) and keeps the 2x upscale to 1440 lines and the default look.
  The clip list says "HDZero digital recording: upscaled without analog cleaning", and
  "Clean as analog" (or `--analog` on the command line) cleans it the old way. `--digital`
  forces the new path.
- **HDZero at 720p60 is recognised from the picture.** An HDZero recording and an analog
  DVR file are both 1280x720 at 60 fps, so the picture decides: HDZero carries detail close
  to the pixel grid, an analog picture is band-limited. Checked on 111 labelled analog and 45 HDZero recordings; it
  is tuned to call no analog flight digital, so a dark or blurry HDZero clip can still be
  cleaned as analog.

- HDZero recordings at 90 fps or 1920x1080 are recognised from the file alone.
- **TensorRT is found and used when present.** AnaHD looks for NVIDIA's free TensorRT
  (10.15, CUDA 12) next to its own install and in your AnaHD folder, and runs without it
  when it is missing. It is too big for the installer (about 2.2 GB), so the app shows a
  quiet strip with a "Download TensorRT" button that fetches it from NVIDIA; the README
  lists it as a recommended requirement with the download link. Built engines are cached
  per graphics card, driver and TensorRT version.
- **1440-line output.** Pictures now come out at 1440 lines in their own shape: 1920x1440
  for 4:3 and 2560x1440 for 16:9 (a 16:9 DVR frame with the analog line's blanking at the
  sides counts as 16:9). The learned restorer's 2x output is used as it is instead of being
  shrunk to 1080 lines.
- **A line pattern is gone.** At 1:1 the restorer's output carried faint horizontal lines
  (a 4 px period, its pixel-shuffle phase). A narrow vertical filter on the luma removes
  them without softening the picture.
- **The classic path upscales with SPAN.** Without the learned restorer, the 2x SPAN
  upscaler (`--no-ai` for Lanczos) replaces the Lanczos resize: TensorRT when present, else
  PyTorch FP16, else Lanczos.
- **Faster cleaning on longer clips.** The U-Net runs on TensorRT and RAFT's update loop is
  compiled (Triton, cached under `%LOCALAPPDATA%\AnaHD`), each falling back to the plain
  path on any failure. The first compile of a picture size takes about 95 s, so it is only
  done on a clip of 9 minutes or more; a size compiled before is reused from 10 s.

## 0.1.5 (2026-11-09)

- **AnaHD can open inside okFPV.** okFPV's AnaHD tab now shows AnaHD in its own window,
  without a second title bar. On its own, AnaHD is unchanged.

## 0.1.4 (2026-11-08)

- **Fast flips no longer lose picture.** Repair compared each frame with neighbouring
  frames that could not be lined up during a fast flip, took good picture for damage and
  replaced it, sometimes with wrong blocks and washed-out colour or OSD text. Repair now
  needs the neighbouring frames to agree with each other first. In exchange, a little more
  static can stay in the most heavily damaged bands during fast moves.
- The window buttons line up with the top bar's own row.
- **A new text font.** The app's text is set in Instrument Sans (SIL Open Font License,
  bundled) instead of the previous face.

## 0.1.3 (2026-10-29)

- **No more boxes around the OSD.** In still moments (on the pad, in menus, landing) faint
  lighter or darker rectangles showed around the OSD and on still parts of the scene that
  were taken for OSD. The OSD is now found by its own strokes, the cleaned picture fades in
  around it over a few pixels instead of at a hard edge, and an OSD element keeps the plain
  frame only while it is on screen, so a menu or a flight timer leaves nothing behind once
  it is gone. Still scenes also pulse less.
- Fewer copies on the GPU and frames sent from pinned memory: about 3% faster with the
  same output, which the box fix spends again; 0.1.3 cleans as fast as 0.1.2.
- The speed estimate under the settings learns your own PC's speed from finished cleans
  (600 frames or more), per graphics card, app version and engine; Reset to defaults
  forgets it.
- A cleaned clip's card shows the cleaned picture.
- A wrong `--cut` says what is expected: START-END in seconds or m:ss, e.g. 0-4.5 or
  1:05-1:20.

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
