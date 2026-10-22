"""The learned restorer's wiring into the GPU cleaner. Skipped without torch and a CUDA GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

from torch import Tensor  # noqa: E402

from anahd.gpu.pipeline import Aligned, Cleaner, Prepared  # noqa: E402
from anahd.model.infer import LearnedRestorer  # noqa: E402
from anahd.model.net import build  # noqa: E402
from anahd.remaster.params import RemasterParams  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda")


def _stub(h: int, w: int) -> Cleaner:
    """A Cleaner with only the fields the learned step reads (no RAFT, no measurement)."""
    c = Cleaner.__new__(Cleaner)
    c.h, c.w = h, w
    c.osd = torch.zeros(h, w, dtype=torch.bool, device=DEV)
    c.osd[:8, :16] = True
    c.osd_half = c.osd[::2, ::2].clone()
    c._near_cache = {}
    c.params = RemasterParams()  # the clean-up sliders at their defaults
    c.cleanup = 1.0
    return c


def test_nearest_neighbour_slots() -> None:
    c = _stub(16, 16)
    assert c._near([-3, -2, -1, 1, 2, 3]) == [1, 2, 3, 4]
    # Clip start: no frames before the centre, so -2 and -1 fall back on the centre itself.
    assert c._near([1, 2, 3, 3, 3, 3]) == [None, None, 0, 1]
    # Clip end: +2 is missing, the nearest is +1.
    assert c._near([-3, -2, -1, 1, 1, 1]) == [1, 2, 3, 3]


def test_untrained_restorer_returns_the_centre(tmp_path: Path) -> None:
    """A fresh network predicts no residual: the centre comes back, the OSD is the centre."""
    ck = tmp_path / "net.pt"
    torch.save({"net": build("s").state_dict(), "size": "s"}, ck)
    net = LearnedRestorer(ck, DEV)
    h, w = 64, 92  # not a multiple of 8, like the 956-wide DVR area
    c = _stub(h, w)
    g = torch.Generator(device=DEV).manual_seed(0)

    def planes(n: int) -> tuple[Tensor, ...]:
        y = torch.rand(n, h, w, device=DEV, generator=g) * 255
        u = torch.rand(n, h // 2, w // 2, device=DEV, generator=g) * 255
        v = torch.rand(n, h // 2, w // 2, device=DEV, generator=g) * 255
        return y, u, v

    cy = planes(1)[0][0]
    # Analog chroma is smooth; a ramp survives the trip to full size and back.
    ramp = torch.linspace(60, 200, w // 2, device=DEV).expand(h // 2, w // 2)
    cu, cv = ramp.clone(), ramp.flip(-1).clone()
    wy, wu, wv = planes(6)
    offsets = [-3, -2, -1, 1, 2, 3]
    z = torch.zeros(6, 1, 1, device=DEV)
    a = Aligned(z, wy, wu, wv, z, offsets, z, wy, wu, wv)
    centre = Prepared(cy, cu, cv, None, cy)  # type: ignore[arg-type, unused-ignore]
    with torch.inference_mode():
        y, u, v = c._learned(net, centre, a)
    keep, keep_h = ~c.osd, ~c.osd_half
    # FP16 inside the network: about 0.1 level of rounding on 0..255.
    assert (y - cy)[keep].abs().max() < 0.2
    assert torch.equal(y[c.osd], cy[c.osd])
    assert (u - cu)[keep_h].abs().max() < 0.5
    assert (v - cv)[keep_h].abs().max() < 0.5
    assert torch.equal(u[c.osd_half], cu[c.osd_half])
    assert torch.equal(v[c.osd_half], cv[c.osd_half])
