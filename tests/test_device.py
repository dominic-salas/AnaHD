"""Which NVIDIA cards the PyTorch build can run on, and the message for the ones it can't."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anahd.errors import AnaHDError  # noqa: E402
from anahd.gpu import device  # noqa: E402

# The shipped build (PyTorch 2.11, CUDA 12.8).
SHIPPED = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.parametrize(
    ("card", "capability", "runs"),
    [
        ("GTX 1060 (Pascal)", (6, 1), False),
        ("GTX 980 (Maxwell)", (5, 2), False),
        ("GTX 1660 (Turing)", (7, 5), True),
        ("RTX 3060 (Ampere)", (8, 6), True),
        ("RTX 4070 Ti (Ada)", (8, 9), True),
        ("RTX 5090 (Blackwell)", (12, 0), True),
    ],
)
def test_shipped_build_runs_from_turing_on(
    card: str, capability: tuple[int, int], runs: bool
) -> None:
    assert device.runs_on(capability, SHIPPED) is runs, card


def test_ptx_runs_on_later_cards_only() -> None:
    assert device.runs_on((9, 0), ["sm_75", "compute_80"])
    assert not device.runs_on((7, 5), ["compute_80"])


def test_too_old_a_card_is_named_in_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (6, 1))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: SHIPPED)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce GTX 1060 6GB")
    with pytest.raises(AnaHDError, match="GTX 1060 6GB is older than AnaHD supports"):
        device.cuda_device()


def test_no_card_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AnaHDError, match="no NVIDIA graphics card found"):
        device.cuda_device()
