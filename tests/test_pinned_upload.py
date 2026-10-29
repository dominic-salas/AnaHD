"""The pinned upload (``GpuConfig.pinned_upload``) gives the same output, bit for bit, and the
same repair counts as the blocking copy: it only stops the host waiting on the GPU. Needs a
CUDA GPU."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import synthetic as syn  # noqa: E402
from anahd.gpu.pipeline import GpuConfig, enhance  # noqa: E402
from anahd.pipeline import Options  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _frames(path: Path) -> list[str]:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if not line.startswith("#")]


@pytest.mark.parametrize("learned", [False, True], ids=["classic", "learned"])
def test_pinned_upload_changes_no_frame_and_no_count(tmp_path: Path, learned: bool) -> None:
    from anahd.model import weights

    clip = syn.write(tmp_path / "dvr.mp4", 120)
    assert any(syn.damaged(i) for i in range(120)), "the clip must have breakup to repair"
    model = weights.resolve(None) if learned else None
    results, frames = [], []
    for pinned in (False, True):
        out = tmp_path / f"out_{pinned}.mp4"
        gcfg = GpuConfig(repair=True, learned=model, pinned_upload=pinned)
        results.append(enhance(clip.path, out, Options(), gcfg=gcfg, overwrite=True))
        frames.append(_frames(out))
    assert len(frames[0]) == 120
    diff = [i for i, (a, b) in enumerate(zip(*frames, strict=True)) if a != b]
    assert not diff, f"frames differ: {diff[:10]}"
    assert results[0].notes == results[1].notes, "the repair counts must not change"
    assert "repaired" in results[0].notes and " 0 rows" not in results[0].notes
    digest = hashlib.sha256("".join(frames[0]).encode()).hexdigest()
    assert digest == hashlib.sha256("".join(frames[1]).encode()).hexdigest()
