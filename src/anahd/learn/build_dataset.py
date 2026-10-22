"""Build the training set: clean 5-frame windows plus RAFT flows from each centre to its
neighbours, and a bank of real residual patches and breakup strips.

Clean targets are the GPU cleaner's own picture (``Cleaner.restore``: fusion after breakup
repair, destripe, derainbow; before the look and the upscale), downscaled so its leftover noise
drops below visibility (960x720 -> 480x360). Windows are skipped where the breakup detector
repaired anything in any of their frames, or where a cleaned frame still differs from its
source by row spikes, so targets are breakup-free; residuals and breakup strips are harvested
from every frame. Flows are computed on the clean frames; training perturbs them so the model
also copes with RAFT's errors on noisy input. Degradation happens later, on the fly, on GPU
(`degrade.py`), before the warp, as in the real pipeline where noisy neighbours are warped.

Writes into `out/`:
  frames.u8.npy   (N, 5, 3, 360, 480) uint8 YUV 4:4:4, memmap
  flows.f16.npy   (N, 4, 2, 180, 240) float16, half-res flows in full-res pixels, memmap
  osd.u8.npy      (N, 360, 480) uint8, 1 = OSD (excluded from the loss)
  bank.f16.npy    (M, 3, 96, 96) float16 residual patches (Y, U, V)
  bands.npz       real breakup strips
  meta.json       shapes, the spans used and what was skipped
Never writes next to the sources.

    python -m anahd.learn.build_dataset SPANS.json OUT --max-windows 3000
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from anahd.gpu.pipeline import Cleaner, GpuConfig
from anahd.media import probe, read_raw
from anahd.pipeline import Measured, Options, _windows, measure

from .noisebank import harvest, harvest_bands, row_spikes

TW, TH = 480, 360
WIN = 5


@dataclass(frozen=True)
class Pair:
    src: Tensor  # (3, H, W) YUV 4:4:4 in 0..1, the source frame (full range)
    clean: Tensor  # (3, H, W) the cleaner's picture of it
    damaged: bool  # the breakup detector repaired rows or colour in it
    rows: Tensor  # (H,) bool, the rows it repaired


def _full444(y: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Planes in 0..255 (chroma at half size) -> (3, H, W) in 0..1."""
    h, w = y.shape

    def up(c: Tensor) -> Tensor:
        return F.interpolate(c[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]

    return torch.stack((y, up(u), up(v))).clamp(0, 255) / 255.0


def cleaned_pairs(
    src: Path, start: float, duration: float, dev: torch.device, cache: dict[Path, Measured]
) -> tuple[Iterator[Pair], Tensor]:
    """Run the GPU cleaner (with repair) over a span; (source, cleaned) per frame at the
    active-area size, and the cleaner's OSD mask (H, W) bool. ``cache`` keeps each
    recording's measurement (crop, OSD, levels: from the whole recording) across spans."""
    info = probe(src)
    if src not in cache:
        cache[src] = measure(src, info, Options())
    meas = cache[src]
    cleaner = Cleaner(
        meas.crop,
        meas.osd,
        meas.curve,
        meas.size,
        info,
        Options().clean,
        Options().look,
        GpuConfig(repair=True),
        dev,
    )
    osd = cleaner.osd.clone()

    def gen() -> Iterator[Pair]:
        raws = read_raw(src, info, meas.crop, start=start, duration=duration)
        prepared = (cleaner.prepare(r) for r in raws)
        last = cleaner.stats
        last_frames = last.frames
        none_rows = torch.zeros(meas.crop.height, dtype=torch.bool, device=dev)
        for centre, nbs in _windows(prepared, Options().clean.radius):
            s = _full444(centre.y, centre.u, centre.v)  # before repair edits the centre
            c = _full444(*cleaner.restore(centre, nbs))
            st = cleaner.stats
            # Not ``pixels``: the per-pixel snow repair touches over 1% of most frames.
            damaged = (
                st.frames != last.frames
                or st.whole != last.whole
                or st.left != last.left
                or st.colour != last.colour
            )
            last = st
            rows = cleaner.repaired_rows.clone() if st.frames != last_frames else none_rows
            last_frames = st.frames
            yield Pair(s, c, damaged, rows)

    return gen(), osd


def _flows(raft: torch.nn.Module, win: np.ndarray, dev: torch.device) -> np.ndarray:
    """Flow from the centre frame to each neighbour (backward-warp convention), half res."""
    y = torch.from_numpy(win[:, 0]).to(dev).float() / 255.0  # luma only, as grey RGB
    img = (y * 2 - 1).unsqueeze(1).repeat(1, 3, 1, 1)
    c = WIN // 2
    nb = [i for i in range(WIN) if i != c]
    a = img[c : c + 1].repeat(len(nb), 1, 1, 1)
    fl = raft(a, img[nb], num_flow_updates=12)[-1]
    fl = F.interpolate(fl, scale_factor=0.5, mode="bilinear", align_corners=False)
    return np.asarray(fl.half().cpu().numpy())


def _save_bands(path: Path, bands: list[np.ndarray]) -> None:
    """Strips have different heights: store them concatenated along rows with an index."""
    if not bands:
        np.savez(path, rows=np.zeros((3, 0, 1), np.float16), starts=np.zeros(1, np.int64))
        return
    starts = np.cumsum([0] + [b.shape[1] for b in bands])
    np.savez(path, rows=np.concatenate(bands, axis=1), starts=starts)


def build(
    spans: list[dict[str, object]],
    out: Path,
    max_windows: int,
    dev: torch.device,
    bank_per_frame: int = 4,
    max_bank: int = 8000,
    max_bands: int = 3000,
) -> None:
    out.mkdir(parents=True, exist_ok=False)
    raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(dev).eval()
    frames = np.lib.format.open_memmap(
        out / "frames.u8.npy", "w+", np.uint8, (max_windows, WIN, 3, TH, TW)
    )
    flows = np.lib.format.open_memmap(
        out / "flows.f16.npy", "w+", np.float16, (max_windows, WIN - 1, 2, TH // 2, TW // 2)
    )
    osd = np.lib.format.open_memmap(out / "osd.u8.npy", "w+", np.uint8, (max_windows, TH, TW))
    bank: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    n = 0
    skipped = {"damaged": 0, "bands": 0}
    used = []
    cache: dict[Path, Measured] = {}
    t0 = time.monotonic()
    for sp in spans:
        if n >= max_windows:
            break
        src = Path(str(sp["source"]))
        start, dur = float(str(sp["start_s"])), float(str(sp["dur_s"]))
        stride = int(str(sp["stride"]))
        pairs, mask_full = cleaned_pairs(src, start, dur, dev, cache)
        mask_small = (
            (F.interpolate(mask_full.float()[None, None], size=(TH, TW), mode="area")[0, 0] > 0.05)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        ring: deque[np.ndarray] = deque(maxlen=WIN)
        damaged: deque[bool] = deque(maxlen=WIN)
        spiky: deque[bool] = deque(maxlen=WIN)
        frames_seen, before = 0, n
        for i, p in enumerate(pairs):
            frames_seen += 1
            if len(bands) < max_bands and p.damaged:
                bands += harvest_bands(p.src, p.clean, mask_full, p.rows)
            spike = bool(row_spikes(p.src, p.clean, mask_full).any())
            if i % 15 == 0 and len(bank) < max_bank:
                bank += harvest(p.src, p.clean, mask_full, max_patches=bank_per_frame)
            small = F.interpolate(p.clean[None], size=(TH, TW), mode="area")[0]
            ring.append((small * 255).round().clamp(0, 255).byte().cpu().numpy())
            damaged.append(p.damaged)
            spiky.append(spike)
            if len(ring) == WIN and (i - WIN // 2) % stride == 0 and n < max_windows:
                if any(damaged) or any(spiky):
                    skipped["damaged" if any(damaged) else "bands"] += 1
                    continue
                win = np.stack(ring)
                frames[n] = win
                flows[n] = _flows(raft, win, dev)
                osd[n] = mask_small
                n += 1
        used.append({**sp, "frames": frames_seen, "windows": n - before})
        mins = (time.monotonic() - t0) / 60
        print(
            f"{src.name} @{start:.0f}s: +{n - before} windows (total {n}), "
            f"bank {len(bank)}, bands {len(bands)}, skipped {skipped}, {mins:.1f} min",
            flush=True,
        )
        del pairs
        gc.collect()
        torch.cuda.empty_cache()
    for mm in (frames, flows, osd):
        mm.flush()
    np.save(out / "bank.f16.npy", np.stack(bank).astype(np.float16))
    _save_bands(out / "bands.npz", bands)
    meta = {
        "windows": n,
        "bank": len(bank),
        "bands": len(bands),
        "size": [TW, TH],
        "skipped": skipped,
        "spans": used,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spans", type=Path, help="JSON list from anahd.learn.spans")
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-windows", type=int, default=3000)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N spans")
    ap.add_argument("--stride", type=int, default=0, help="frames between windows (0: per span)")
    ap.add_argument("--limit", type=int, default=0, help="use only N spans (bursts, tests)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    spans = json.loads(a.spans.read_text())
    # Grouped by recording (one measurement each) but recordings in a random order, so a
    # window budget that runs out early does not drop the same flights every time.
    order: dict[str, list[dict[str, object]]] = {}
    for sp in spans:
        order.setdefault(str(sp["source"]), []).append(sp)
    groups = list(order.values())
    np.random.default_rng(1).shuffle(groups)  # type: ignore[arg-type, unused-ignore]
    spans = [sp for g in groups for sp in g]
    if a.stride:
        spans = [{**sp, "stride": a.stride} for sp in spans]
    spans = spans[a.offset :]
    if a.limit:
        spans = spans[: a.limit]
    with torch.inference_mode():
        build(spans, a.out, a.max_windows, torch.device(a.device))


if __name__ == "__main__":
    main()
