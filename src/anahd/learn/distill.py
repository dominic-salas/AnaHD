"""Learn a slow teacher's look (SeedVR2, see ``teacher_data``) with the fast restorer.

    python -m anahd.learn.distill prep DATA            # steady targets from the teacher frames
    python -m anahd.learn.distill train DATA RUN --init CKPT --minutes 120

``prep`` turns the teacher's frames t-1, t, t+1 around each window centre into one target:
the neighbours are warped onto the centre with the cleaner's own flow and averaged in where
they agree. Detail the teacher draws the same way in consecutive frames survives; texture it
redraws differently every frame (walls and roofs "boiling") averages out, so the student never
learns it. ``train`` fits an upscaling restorer (``m2``) from the real input stacks to those
targets: its 2x output is resampled to the teacher's scale (1.5x) for the loss, as the
pipeline resamples it to 1440x1080.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.gpu.pipeline import _ycc_matrix
from anahd.model.net import build

from .train import charb
from .warp import backward_warp

SIGMA = 0.02  # luma difference (0..1) at which a teacher neighbour's weight falls off


def _ycc_from_rgb(bt709: bool) -> Tensor:
    kr, kb = (0.2126, 0.0722) if bt709 else (0.299, 0.114)
    return torch.from_numpy(np.linalg.inv(_ycc_matrix(kr, kb))).float()


def _teacher_frame(path: Path, m: Tensor, dev: torch.device) -> Tensor:
    """PNG (RGB) -> (3, H, W) YUV full range in 0..1, chroma centred on 0.5."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(dev).float() / 255.0
    ycc = torch.einsum("ij,hwj->ihw", m.to(dev), rgb)
    return torch.stack((ycc[0], ycc[1] + 0.5, ycc[2] + 0.5)).clamp(0, 1)


def _blur3(x: Tensor) -> Tensor:
    return F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)


def _gx(t: Tensor) -> Tensor:
    return t[..., :, 1:] - t[..., :, :-1]


def _gy(t: Tensor) -> Tensor:
    return t[..., 1:, :] - t[..., :-1, :]


def prep_span(span: Path, dev: torch.device) -> int:
    meta = json.loads((span / "meta.json").read_text())
    frames = sorted((span / "teacher_png" / "feed").glob("*.png"))
    # Centres whose next teacher frame exists (all of them once the teacher has finished).
    centres = [c for c in meta["centres"] if c + 1 < len(frames)]
    if not centres:
        return 0
    m = _ycc_from_rgb(bool(meta["bt709"]))
    flows = np.load(span / "flows.npy", mmap_mode="r")
    first = _teacher_frame(frames[0], m, dev)
    th, tw = first.shape[-2:]
    out = np.lib.format.open_memmap(span / "targets.npy", "w+", np.uint8, (len(centres), 3, th, tw))
    for k, c in enumerate(centres):
        t = torch.stack([_teacher_frame(frames[i], m, dev) for i in (c - 1, c, c + 1)])
        fl = torch.from_numpy(np.asarray(flows[k], np.float32)).to(dev)  # (2, 2, h/2, w/2)
        sx = tw / (fl.shape[-1] * 2)  # full-size source pixels -> teacher pixels
        fl = F.interpolate(fl, size=(th, tw), mode="bilinear", align_corners=False) * sx
        nb = backward_warp(t[[0, 2]], fl)
        ref = _blur3(t[1:2, :1])
        w = torch.exp(-(((_blur3(nb[:, :1]) - ref) / SIGMA) ** 2))  # (2, 1, H, W)
        tgt = (t[1] + (w * nb).sum(0)) / (1.0 + w.sum(0))
        out[k] = (tgt * 255.0).round().clamp(0, 255).byte().cpu().numpy()
    out.flush()
    return len(centres)


def prep_raw(span: Path, dev: torch.device) -> int:
    """The teacher's own luma at each target's centre (``raws.npy``), unaveraged: its texture
    energy is what the student must keep (the steady average softens texture, the student
    more so)."""
    if not (span / "targets.npy").exists():
        return 0
    meta = json.loads((span / "meta.json").read_text())
    frames = sorted((span / "teacher_png" / "feed").glob("*.png"))
    n = len(np.load(span / "targets.npy", mmap_mode="r"))
    centres = meta["centres"][:n]
    m = _ycc_from_rgb(bool(meta["bt709"]))
    first = _teacher_frame(frames[0], m, dev)
    th, tw = first.shape[-2:]
    out = np.lib.format.open_memmap(span / "raws.npy", "w+", np.uint8, (n, 1, th, tw))
    for k, c in enumerate(centres):
        y = _teacher_frame(frames[c], m, dev)[:1]
        out[k] = (y * 255.0).round().clamp(0, 255).byte().cpu().numpy()
    out.flush()
    return n


class Pairs:
    """Real input stacks and steady teacher targets from every prepared span."""

    def __init__(self, root: Path) -> None:
        self.spans = []
        for d in sorted(root.glob("span_*")):
            if not (d / "targets.npy").exists():
                continue
            st = np.load(d / "stacks.npy", mmap_mode="r")
            tg = np.load(d / "targets.npy", mmap_mode="r")
            n = min(len(st), len(tg))  # targets cover a prefix of the centres
            if n == 0:
                continue
            raw = np.load(d / "raws.npy", mmap_mode="r")[:n] if (d / "raws.npy").exists() else None
            self.spans.append((st[:n], tg[:n], np.load(d / "osd.npy"), raw))
        self.index = [(i, k) for i, sp in enumerate(self.spans) for k in range(len(sp[0]))]
        self.rng = np.random.default_rng()

    def batch(
        self, bs: int, crop: int, dev: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """(stack (B,5,3,c,c), target (B,3,1.5c,1.5c), keep (B,1,1.5c,1.5c), raw teacher luma
        (B,1,1.5c,1.5c) or None when not prepared); c even."""
        xs, ys, ms, rs = [], [], [], []
        for j in self.rng.integers(0, len(self.index), bs):
            i, k = self.index[int(j)]
            st, tg, osd, raw = self.spans[i]
            h, w = st.shape[-2:]
            th, tw = tg.shape[-2:]
            y0 = int(self.rng.integers(0, h - crop + 1)) & ~1
            x0 = int(self.rng.integers(0, w - crop + 1)) & ~1
            ty, tx = round(y0 * th / h), round(x0 * tw / w)
            tc = crop * 3 // 2
            ty, tx = min(ty, th - tc), min(tx, tw - tc)
            xs.append(np.asarray(st[k, :, :, y0 : y0 + crop, x0 : x0 + crop]))
            ys.append(np.asarray(tg[k, :, ty : ty + tc, tx : tx + tc]))
            if raw is not None:
                rs.append(np.asarray(raw[k, :, ty : ty + tc, tx : tx + tc]))
            o = cv2.resize(
                osd[y0 : y0 + crop, x0 : x0 + crop].astype(np.uint8),
                (tc, tc),
                interpolation=cv2.INTER_NEAREST,
            )
            ms.append(1 - o)
        stack = torch.from_numpy(np.stack(xs)).to(dev).float() / 255.0
        tgt = torch.from_numpy(np.stack(ys)).to(dev).float() / 255.0
        keep = torch.from_numpy(np.stack(ms)).to(dev).float()[:, None]
        flip = torch.rand(bs, device=dev) < 0.5
        stack = torch.where(flip.view(bs, 1, 1, 1, 1), stack.flip(-1), stack)
        tgt = torch.where(flip.view(bs, 1, 1, 1), tgt.flip(-1), tgt)
        keep = torch.where(flip.view(bs, 1, 1, 1), keep.flip(-1), keep)
        rawt = None
        if len(rs) == bs:
            rawt = torch.from_numpy(np.stack(rs)).to(dev).float() / 255.0
            rawt = torch.where(flip.view(bs, 1, 1, 1), rawt.flip(-1), rawt)
        return stack, tgt, keep, rawt


def to_teacher_scale(out: Tensor, size: int) -> Tensor:
    return F.interpolate(out, size=(size, size), mode="bicubic", antialias=True)


def texture(y: Tensor) -> Tensor:
    """Local fine-detail energy of luma (B,1,H,W): mean |y - blur(y)| over 8x8, stride 4."""
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=y.device)
    k = (k[:, None] * k[None, :]) / 256.0
    blur = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="replicate"), k.view(1, 1, 5, 5))
    return F.avg_pool2d((y - blur).abs(), 8, 4)


def texture_loss(out_y: Tensor, raw_y: Tensor, m: Tensor) -> Tensor:
    """The student's texture energy against the raw teacher's, where both are picture."""
    mm = F.avg_pool2d(m, 8, 4)
    d = (texture(out_y) - texture(raw_y)).abs() * (mm > 0.99)
    return d.sum() / (mm > 0.99).sum().clamp(min=1)


def loss_fn(out: Tensor, tgt: Tensor, m: Tensor) -> Tensor:
    w = torch.tensor([1.0, 0.5, 0.5], device=out.device).view(1, 3, 1, 1)
    l1 = charb(out * w, tgt * w, m)
    lg = charb(_gx(out[:, :1]), _gx(tgt[:, :1]), m[..., :, 1:]) + charb(
        _gy(out[:, :1]), _gy(tgt[:, :1]), m[..., 1:, :]
    )
    return l1 + 1.0 * lg


def train(a: argparse.Namespace) -> None:
    from .yuvimg import save_rgb_png

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    a.run.mkdir(parents=True, exist_ok=True)
    data = Pairs(a.data)
    print(f"{len(data.index)} windows from {len(data.spans)} spans", flush=True)
    net = build(a.size).to(dev).to(memory_format=torch.channels_last)  # type: ignore[call-overload, unused-ignore]
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    it = 0
    ck = a.run / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        it = int(st["it"])
        print(f"resumed at {it}", flush=True)
    elif a.init is not None:
        net.load_state_dict(torch.load(a.init, map_location=dev)["net"])
        print(f"initialised from {a.init}", flush=True)
    stop = False

    def _sig(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)

    def save() -> None:
        tmp = a.run / "last.tmp"
        torch.save(
            {"net": net.state_dict(), "opt": opt.state_dict(), "it": it, "size": a.size}, tmp
        )
        tmp.replace(ck)

    t_end, t_save = time.monotonic() + a.minutes * 60, time.monotonic()
    run_loss, t0, i0 = 0.0, time.monotonic(), it
    tc = a.crop * 3 // 2
    with (a.run / "log.txt").open("a") as log:
        while not stop and time.monotonic() < t_end and it < a.iters:
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.iters)) * min(1.0, (it + 1) / 500)
            for g in opt.param_groups:
                g["lr"] = lr
            stack, tgt, keep, raw = data.batch(a.bs, a.crop, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(stack)
            out = to_teacher_scale(out.float(), tc)
            loss = loss_fn(out, tgt, keep)
            if a.tex > 0 and raw is not None:
                loss = loss + a.tex * texture_loss(out[:, :1], raw, keep)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            it += 1
            run_loss += float(loss.detach())
            if it % 200 == 0:
                ips = (it - i0) / (time.monotonic() - t0)
                msg = f"it {it} loss {run_loss / 200:.5f} lr {lr:.2e} {ips:.1f} it/s"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                run_loss = 0.0
            if it % 2000 == 0:
                inp = F.interpolate(stack[:, 2], size=(tc, tc), mode="bicubic", align_corners=False)
                rows = [
                    torch.cat((inp[i], out[i].clamp(0, 1), tgt[i]), -1) for i in range(min(4, a.bs))
                ]
                save_rgb_png(a.run / f"preview_{it:06d}.png", torch.cat(rows, -2).detach())
            if time.monotonic() - t_save > 300:
                save()
                t_save = time.monotonic()
    save()
    print(f"stopped at {it}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("data", type=Path)
    t = sub.add_parser("train")
    t.add_argument("data", type=Path)
    t.add_argument("run", type=Path)
    t.add_argument("--size", default="m2")
    t.add_argument("--init", type=Path)
    t.add_argument("--minutes", type=float, default=120.0)
    t.add_argument("--iters", type=int, default=60_000)
    t.add_argument("--bs", type=int, default=8)
    t.add_argument("--crop", type=int, default=128)
    t.add_argument("--lr", type=float, default=1.5e-4)
    t.add_argument("--tex", type=float, default=0.0, help="texture-energy loss weight")
    r = sub.add_parser("prep-raw")
    r.add_argument("data", type=Path)
    a = ap.parse_args()
    if a.cmd == "prep":
        dev = torch.device("cuda")
        with torch.inference_mode():
            for d in sorted(a.data.glob("span_*")):
                if (d / "targets.npy").exists():
                    continue
                print(f"{d.name}: {prep_span(d, dev)} targets", flush=True)
    elif a.cmd == "prep-raw":
        dev = torch.device("cuda")
        with torch.inference_mode():
            for d in sorted(a.data.glob("span_*")):
                if not (d / "raws.npy").exists():
                    print(f"{d.name}: {prep_raw(d, dev)} raw frames", flush=True)
    else:
        train(a)


if __name__ == "__main__":
    main()
