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

### 5-second motion check before committing

**What:** A "Preview this stretch" action that renders 5 seconds around the playhead (~15 s of work) with the current settings, before a full Clean.

**Why:** A still frame can't show breakup repair, flicker or fast-roll smearing; this shows the settings in motion before an 8-minute commitment.

**Pros:** It catches a wrong look before any long render; it reuses `cli.render` with `--start/--duration`.

**Cons:** It overlaps with the live view (which lets you Stop early). That makes it a second way to get a motion preview.

**Context:** Office-hours idea, deferred in the CEO review (E6) on 2026-10-28. Revisit after using the live view for a while: if Stop-at-minute-one feels good enough, skip it.

**Effort:** M (human) → S (CC+gstack)
**Priority:** P3
**Depends on:** live view shipped

### Speed estimate calibrated to the user's own GPU

**What:** After the first clean on a machine, scale `app/speed.py`'s estimate by the ratio of the measured fps to the predicted fps, stored in QSettings.

**Why:** `speed.py`'s constants are from one RTX 4070 Ti; on any other card the red "past ten minutes" warning and the estimate are wrong.

**Pros:** Honest estimates for every user; small.

**Cons:** One clean is a noisy sample; it needs a reset when the settings change engine.

**Context:** CEO review E2, 2026-10-28. The measured fps is already known at `window._finished` (frames / took).

**Effort:** S (human) → S (CC+gstack)
**Priority:** P3
**Depends on:** none

### Card thumbnail shows the cleaned frame when done

**What:** When a clip finishes, its card thumbnail switches to the last live snapshot (cleaned).

**Why:** You can see at a glance in the list which clips are done, and what they look like.

**Pros:** Reuses the live snapshot; tiny.

**Cons:** The thumbnail no longer matches the recording, which may confuse.

**Context:** CEO review E3, 2026-10-28.

**Effort:** S (human) → S (CC+gstack)
**Priority:** P4
**Depends on:** live view shipped

## Completed
