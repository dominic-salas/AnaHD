# TODOS

## Cleaning

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

**Context:** Approach C plus lever 6, deferred in the 2026-10-29 plan reviews. Reopened 2026-10-29: 0.1.3 cleans at 0.1.2's speed, and Wave 2's levers measured no gain (below), so this is the remaining route to the 20% goal (503 s to 402 s or less). Needs its own design (Triton and TensorRT inside the PyInstaller installer, per-GPU engines) before any build.

**Effort:** L (human) → M (CC+gstack)
**Priority:** P2
**Depends on:** its own design and plan review

## Completed

- **`--cut` error names an internal function**: Done 2026-10-29 (0.1.3): the error now ends "a cut is START-END in seconds or m:ss, e.g. 0-4.5 or 1:05-1:20".
- **Speed estimate calibrated to the user's own GPU**: Done 2026-10-29 (0.1.3): after a completed run of 600+ frames the estimate learns this PC's speed per GPU, app version and engine (half old, half new, kept within 0.5-2x); Reset to defaults forgets it.
- **Card thumbnail shows the cleaned frame when done**: Done 2026-10-29 (0.1.3): the card shows the last live "after" picture; without one, or when stopped or failed, it keeps the recording's.
- **Live caption total counts packets, not frames**: Closed, won't do (2026-10-29): off by one frame at most; the decoded count would cost an extra probe.
- **5-second motion check before committing**: Closed, won't do (2026-10-29): the live view shows motion a minute in, with Stop under it.
- **Rectangular boxes in still scenes**: Fixed 2026-10-29 (0.1.3): the OSD mask is the OSD's own strokes, the switch to the cleaned picture is feathered, and an element keeps the plain frame only while it shows. Evidence: docs/designs/evidence/osd-boxes-fixed-hdz30s.jpg.
- **Wave 2 speed levers (LK box filter, fp16 warp/blend)**: Closed 2026-10-29, measured no gain before any gate run. LK refine's two separable avg_pool2d passes take 1.125 ms per iteration on the real (6, 5, 720, 956) fp16 tensor; one grouped 5x5 conv over a count map takes 1.147 ms, a separable grouped conv 1.298 ms. The warp's four grid_samples take 0.995 ms in fp32 and 1.000 ms in fp16 (index arithmetic, not memory, is the cost); the fp16 blend only touches the Classic path.
