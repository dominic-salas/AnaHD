"""Train the restorer in time-boxed bursts (the GPU may be needed for other work in between).

    python -m anahd.learn.train DATA RUN --size m --minutes 30

Resumes from RUN/last.pt if it exists, saves every few minutes and on Ctrl+C, and writes a
preview strip (degraded | output | clean) to RUN/preview_XXXXXX.png.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.model.net import build

from .degrade import AnalogDegrader, DegradeConfig
from .warp import backward_warp

CROP = 192
MARGIN = 16


class Windows:
    """Training windows from one or more dataset folders (built in bursts)."""

    def __init__(self, roots: list[Path], max_bank: int = 8000, max_bands: int = 3000) -> None:
        self.frames, self.flows, self.osd = [], [], []
        bank, self.bands = [], []
        for root in roots:
            n = int(json.loads((root / "meta.json").read_text())["windows"])
            if n == 0:
                continue
            self.frames.append(np.load(root / "frames.u8.npy", mmap_mode="r")[:n])
            self.flows.append(np.load(root / "flows.f16.npy", mmap_mode="r")[:n])
            self.osd.append(np.load(root / "osd.u8.npy", mmap_mode="r")[:n])
            bank.append(np.load(root / "bank.f16.npy"))
            bz = np.load(root / "bands.npz")
            rows, starts = bz["rows"], bz["starts"]
            self.bands += [
                torch.from_numpy(rows[:, starts[j] : starts[j + 1]].astype(np.float32))
                for j in range(len(starts) - 1)
            ]
        self.rng = np.random.default_rng()
        allbank = np.concatenate(bank)
        pick = self.rng.permutation(len(allbank))[:max_bank]
        self.bank = torch.from_numpy(allbank[pick].astype(np.float32))
        if len(self.bands) > max_bands:
            keep = self.rng.permutation(len(self.bands))[:max_bands]
            self.bands = [self.bands[int(j)] for j in keep]
        self.sizes = [len(f) for f in self.frames]
        self.n = sum(self.sizes)
        self.start = np.cumsum([0, *self.sizes])

    def batch(self, bs: int, dev: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """(clean (B,5,3,S,S), flows (B,4,2,S,S), osd (B,1,S,S)) with S = CROP + 2*MARGIN."""
        s = CROP + 2 * MARGIN
        _, _, _, h, w = self.frames[0].shape
        fr, fl, om = [], [], []
        for g in self.rng.integers(0, self.n, bs):
            k = int(np.searchsorted(self.start, g, side="right")) - 1
            i = int(g - self.start[k])
            frames, flows, osd = self.frames[k], self.flows[k], self.osd[k]
            y0 = int(self.rng.integers(0, h - s + 1)) & ~1
            x0 = int(self.rng.integers(0, w - s + 1)) & ~1
            fr.append(np.asarray(frames[i, :, :, y0 : y0 + s, x0 : x0 + s]))
            fl.append(np.asarray(flows[i, :, :, y0 // 2 : (y0 + s) // 2, x0 // 2 : (x0 + s) // 2]))
            om.append(np.asarray(osd[i, y0 : y0 + s, x0 : x0 + s]))
        clean = torch.from_numpy(np.stack(fr)).to(dev, non_blocking=True).float() / 255.0
        flows = torch.from_numpy(np.stack(fl).astype(np.float32)).to(dev)
        b = flows.shape[0]
        flows = F.interpolate(
            flows.view(b * 4, 2, s // 2, s // 2), size=(s, s), mode="bilinear", align_corners=False
        )
        flows = flows.view(b, 4, 2, s, s)
        osd = torch.from_numpy(np.stack(om)).to(dev).float().unsqueeze(1)
        return clean, flows, osd


def augment(clean: Tensor, flows: Tensor, osd: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Augmented (clean, true flows, flows with RAFT-like errors, osd)."""
    b = clean.shape[0]
    dev = clean.device
    # Horizontal flip (flow x component flips sign).
    flip = torch.rand(b, device=dev) < 0.5
    fv = flip.view(b, 1, 1, 1, 1)
    clean = torch.where(fv, clean.flip(-1), clean)
    fflip = flows.flip(-1) * torch.tensor([-1.0, 1.0], device=dev).view(1, 1, 2, 1, 1)
    flows = torch.where(fv, fflip, flows)
    osd = torch.where(flip.view(b, 1, 1, 1), osd.flip(-1), osd)
    # Time reversal: frames reversed, neighbour order (t-2, t-1, t+1, t+2) reversed too.
    rev = (torch.rand(b, device=dev) < 0.5).view(b, 1, 1, 1, 1)
    clean = torch.where(rev, clean.flip(1), clean)
    flows = torch.where(rev, flows.flip(1), flows)
    # Exposure: brighter/darker scenes, more or less saturated.
    g = torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(0.6, 1.15)
    sat = torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    clean = torch.cat((clean[:, :, :1] * g, (clean[:, :, 1:] - 0.5) * sat + 0.5), dim=2)
    clean = clean.clamp(0, 1)
    # RAFT on noisy input is less accurate than on clean frames: smooth flow errors.
    amp = torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(0, 0.7)
    lo = torch.randn(b, 4, 2, 6, 6, device=dev) * amp
    s = flows.shape[-1]
    err = F.interpolate(lo.view(b * 8, 1, 6, 6), size=(s, s), mode="bicubic", align_corners=False)
    return clean, flows, flows + err.view(b, 4, 2, s, s), osd


def _blur3(t: Tensor) -> Tensor:
    return F.avg_pool2d(t, 3, 1, 1, count_include_pad=False)


def steady_target(clean: Tensor, flows: Tensor, sigma: float = 0.02) -> Tensor:
    """The centre averaged with its two nearest clean neighbours warped onto it, each
    weighted by how well it agrees (luma, lightly blurred): a target that holds still from
    frame to frame where the cleaner's own output still shimmers. clean (B,5,3,S,S), true
    flows (B,4,2,S,S) from the centre to frames 0, 1, 3, 4."""
    b, _, c, h, w = clean.shape
    centre = clean[:, 2]
    nb = backward_warp(
        clean[:, [1, 3]].reshape(b * 2, c, h, w), flows[:, [1, 2]].reshape(b * 2, 2, h, w)
    ).view(b, 2, c, h, w)
    ref = _blur3(centre[:, :1])
    wts = [torch.exp(-(((_blur3(nb[:, k, :1]) - ref) / sigma) ** 2)) for k in range(2)]
    total = 1.0 + wts[0] + wts[1]
    return (centre + wts[0] * nb[:, 0] + wts[1] * nb[:, 1]) / total


def charb(x: Tensor, y: Tensor, m: Tensor) -> Tensor:
    d = torch.sqrt((x - y) ** 2 + 1e-6) * m
    return d.sum() / (m.sum() * x.shape[1] / m.shape[1] + 1e-6)


def _gx(t: Tensor) -> Tensor:
    return t[..., :, 1:] - t[..., :, :-1]


def _gy(t: Tensor) -> Tensor:
    return t[..., 1:, :] - t[..., :-1, :]


def loss_fn(out: Tensor, tgt: Tensor, m: Tensor) -> Tensor:
    w = torch.tensor([1.0, 0.7, 0.7], device=out.device).view(1, 3, 1, 1)
    l1 = charb(out * w, tgt * w, m)
    lg = charb(_gx(out[:, :1]), _gx(tgt[:, :1]), m[..., :, 1:]) + charb(
        _gy(out[:, :1]), _gy(tgt[:, :1]), m[..., 1:, :]
    )
    return l1 + 0.5 * lg


def step_inputs(deg: AnalogDegrader, clean: Tensor, flows: Tensor) -> Tensor:
    noisy = deg(clean)
    b, _, c, h, w = noisy.shape
    nb = [0, 1, 3, 4]
    warped = backward_warp(noisy[:, nb].reshape(b * 4, c, h, w), flows.reshape(b * 4, 2, h, w))
    stack = noisy.clone()
    stack[:, nb] = warped.view(b, 4, c, h, w)
    return stack


def downscale(clean: Tensor, flows: Tensor, s: int) -> tuple[Tensor, Tensor]:
    """Clean windows and flows at 1/s size (flows in the smaller pixels), the input side
    of an upscaling network."""
    if s == 1:
        return clean, flows
    b, t, c, h, w = clean.shape
    small = F.avg_pool2d(clean.reshape(b * t, c, h, w), s).view(b, t, c, h // s, w // s)
    n = flows.shape[1]
    f = F.avg_pool2d(flows.reshape(b * n, 2, h, w), s).view(b, n, 2, h // s, w // s) / s
    return small, f


def save_preview(path: Path, stack: Tensor, out: Tensor, tgt: Tensor) -> None:
    """Rows of degraded centre | output | target."""
    from .yuvimg import save_rgb_png

    if stack.shape[-1] != out.shape[-1]:
        s = out.shape[-1] // stack.shape[-1]
        stack = stack.repeat_interleave(s, -1).repeat_interleave(s, -2)
    c = stack.shape[1] // 2
    rows = [torch.cat((stack[i, c], out[i], tgt[i]), dim=-1) for i in range(min(4, out.shape[0]))]
    save_rgb_png(path, torch.cat(rows, dim=-2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="dataset folder (or several, comma-separated)")
    ap.add_argument("run", type=Path)
    ap.add_argument("--size", default="m")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=120_000, help="schedule length (total)")
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", type=Path, help="start from these weights (a new run only)")
    ap.add_argument("--steady", action="store_true", help="temporally averaged targets")
    ap.add_argument("--p-burst", type=float, default=0.0, help="multi-frame breakup bursts")
    ap.add_argument(
        "--soft",
        action="store_true",
        help="always soften the input like real analog detail (for upscaling networks)",
    )
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    a.run.mkdir(parents=True, exist_ok=True)
    data = Windows([Path(p) for p in str(a.data).split(",")])
    print(f"{data.n} windows, bank {len(data.bank)}, bands {len(data.bands)}", flush=True)
    dcfg = DegradeConfig(p_burst=a.p_burst)
    if a.soft:
        dcfg = replace(dcfg, p_soften=1.0, soft_w=(0.7, 1.8), soft_v=(0.3, 1.0))
    deg = AnalogDegrader(dcfg, data.bank.to(dev), [x.to(dev) for x in data.bands])
    net = build(a.size).to(dev).to(memory_format=torch.channels_last)  # type: ignore[call-overload, unused-ignore]
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    it = 0
    ck = a.run / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        it = int(st["it"])
        print(f"resumed at {it}")
    elif a.init is not None:
        init = torch.load(a.init, map_location=dev)["net"]
        own = net.state_dict()
        keep = {k: v for k, v in init.items() if k in own and own[k].shape == v.shape}
        net.load_state_dict(keep, strict=False)
        print(f"initialised from {a.init} ({len(keep)} of {len(own)} tensors)")
    scale = int(net.cfg.scale)

    stop = False

    def _sig(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    def save() -> None:
        tmp = a.run / "last.tmp"
        state = {"net": net.state_dict(), "opt": opt.state_dict(), "it": it, "size": a.size}
        torch.save(state, tmp)
        tmp.replace(ck)

    t_end = time.monotonic() + a.minutes * 60
    t_save = time.monotonic()
    run_loss, t0, i0 = 0.0, time.monotonic(), it
    with (a.run / "log.txt").open("a") as log:
        while not stop and time.monotonic() < t_end and it < a.iters:
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.iters)) * min(1.0, (it + 1) / 1000)
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            clean, true_flows, flows, osd = augment(*data.batch(a.bs, dev))
            stack = step_inputs(deg, *downscale(clean, flows, scale))
            sl = slice(MARGIN, -MARGIN)
            si = slice(MARGIN // scale, -MARGIN // scale)
            stack = stack[..., si, si].contiguous()
            full_tgt = steady_target(clean, true_flows) if a.steady else clean[:, 2]
            tgt = full_tgt[..., sl, sl]
            m = 1.0 - osd[..., sl, sl]
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = net(stack)
            loss = loss_fn(out.float(), tgt, m)
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
                save_preview(
                    a.run / f"preview_{it:06d}.png",
                    stack.float(),
                    out.float().clamp(0, 1),
                    tgt,
                )
            if time.monotonic() - t_save > 300:
                save()
                t_save = time.monotonic()
    save()
    print(f"stopped at {it}", flush=True)


if __name__ == "__main__":
    main()
