# TODOS

## Cleaning

### Rectangular boxes in still scenes (found 2026-10-28)

**What:** Faint lighter or darker rectangles in the cleaned output where the scene doesn't move: on the propeller guard and around the holes in the desk in `hdz_30s.mp4` (drone sitting on a desk, DVR menu showing). See `docs/designs/evidence/osd-boxes-hdz30s.png`: output frames 0 and 26, at 2x.

**Why:** This is the kind of artifact the owner has caught before ("noticable boxes behind OSD elements"). It is present in 0.1.1 and in 0.1.2 alike: 0.1.2's output is bit-identical. Likely cause, not yet verified: in a scene that barely moves, still detail is taken for OSD (README, "OSD protection"). OSD pixels keep the current noisy frame, while the area around them is fused from the neighbours, so the edge of the mask's boxes shows.

**Pros:** It would remove a visible flaw from every still moment: menus, landing, waiting on the pad.

**Cons:** Changes the output, so it needs the relative gate and an eye check; the OSD mask is shared with OSD protection and the redraw.

**Context:** Start in `anahd.picture.detect_osd` (how the mask is made) and in the GPU pipeline's OSD protection (where OSD pixels keep the current frame). Check whether the boxes match the mask's tiles. A fix might feather the mask edge, or require the OSD to be bright, edged detail rather than anything that stays still.

**Effort:** M (human) → S (CC+gstack)
**Priority:** P1
**Depends on:** none

## App

### Play the cleaned output while it renders

**What:** Write the `.partial.mp4` as fragmented MP4 so the Results player can play the cleaned flight up to wherever the render has reached.

**Why:** It turns the wait into watching. At ~32 fps a 3-minute flight renders at about half real time, so playback lags only a few minutes behind.

**Pros:** The strongest "whoa" version of the live view; you see real motion, not snapshots.

**Cons:** Playing a growing file in Qt, `verify()` and the frame-count check, and the final remux with sound all need rework.

**Context:** Approach C in `docs/designs/live-view-and-speed.md`, deferred in office hours and again in the CEO review (E5) on 2026-10-28. Start after the live view (0.1.2) ships. The encoder flags are at `src/anahd/media.py:330` (`+faststart`).

**Effort:** L (human) → M (CC+gstack)
**Priority:** P3
**Depends on:** the live view (T1-T3 in the design doc)

### Whole-step fusion (torch.compile/Triton) and the U-Net in TensorRT

**What:** Fuse the clean step's element-wise chains with torch.compile/Triton, and run the U-Net through TensorRT.

**Why:** Possibly the largest remaining speed gain once the layered plan (docs/designs/faster-cleaning-and-todos.md) has shipped.

**Pros:** Fewer kernels and less memory traffic; TensorRT convs are often much faster than PyTorch's.

**Cons:** Native runtimes inside the PyInstaller installer; TensorRT engines are built per GPU architecture, so it needs a build cache, a first-run cost and a PyTorch fallback.

**Context:** Approach C plus lever 6, deferred in the 2026-10-29 plan reviews. Reopen if the shipped gain stays under 20% (503 s to 402 s or less).

**Effort:** L (human) → M (CC+gstack)
**Priority:** P3
**Depends on:** waves 1-3 of the faster-cleaning plan shipped

## Completed

- **`--cut` error names an internal function**: Done 2026-10-29 (0.1.3): the error now ends "a cut is START-END in seconds or m:ss, e.g. 0-4.5 or 1:05-1:20".
- **Speed estimate calibrated to the user's own GPU**: Done 2026-10-29 (0.1.3): after a completed run of 600+ frames the estimate learns this PC's speed per GPU, app version and engine (half old, half new, kept within 0.5-2x); Reset to defaults forgets it.
- **Card thumbnail shows the cleaned frame when done**: Done 2026-10-29 (0.1.3): the card shows the last live "after" picture; without one, or when stopped or failed, it keeps the recording's.
- **Live caption total counts packets, not frames**: Closed, won't do (2026-10-29): off by one frame at most; the decoded count would cost an extra probe.
- **5-second motion check before committing**: Closed, won't do (2026-10-29): the live view shows motion a minute in, with Stop under it.
