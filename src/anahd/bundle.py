"""Find what the standalone Windows build ships next to the program: FFmpeg, the model and
RAFT's weights. Running from source, none of these folders exist and nothing changes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def root() -> Path | None:
    """The standalone build's folder (where AnaHD.exe is), or None when run from source."""
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else None


def models() -> Path | None:
    base = root()
    return base / "models" if base is not None else None


def setup() -> None:
    """Put the bundled FFmpeg first on PATH and let torchvision find the bundled RAFT
    weights (so the first run needs no download). Call once at start-up."""
    base = root()
    if base is None:
        return
    ffmpeg = base / "ffmpeg"
    if (ffmpeg / "ffmpeg.exe").is_file():
        os.environ["PATH"] = f"{ffmpeg}{os.pathsep}{os.environ.get('PATH', '')}"
    torch_home = base / "models" / "torch"
    if torch_home.is_dir():
        os.environ.setdefault("TORCH_HOME", str(torch_home))
