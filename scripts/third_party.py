"""Third-party components of AnaHD and their licences.

    uv run python scripts/third_party.py            # rewrite THIRD_PARTY_LICENSES.md
    uv run python scripts/third_party.py --check    # fail if it is out of date (CI)

The Python packages are found by walking the installed dependency graph of
``anahd[app]`` (what the standalone build bundles), so the list follows the lock file.
``scripts/build_windows.py`` copies the same packages' licence files into the installer.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "THIRD_PARTY_LICENSES.md"

# Not Python packages, or not visible in package metadata.
OTHERS = (
    (
        "FFmpeg",
        "GPL-3.0-or-later (the bundled build)",
        "Decodes and encodes video; runs as a "
        "separate program. Source: https://ffmpeg.org/download.html",
    ),
    (
        "NVIDIA CUDA runtime, cuDNN, cuBLAS and related libraries",
        "NVIDIA software licence (redistributable)",
        "Shipped inside PyTorch's CUDA build",
    ),
    (
        "RAFT-small optical-flow weights (torchvision)",
        "BSD-3-Clause",
        "Teed and Deng, 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow'",
    ),
    ("Python", "PSF License", "The interpreter inside the standalone build"),
    (
        "PyInstaller bootloader",
        "GPL-2.0 with an exception for bundled programs",
        "Starts the standalone build",
    ),
    ("Inno Setup", "Inno Setup licence (permissive)", "Makes the installer"),
    (
        "SeedVR2 7B (ByteDance Seed)",
        "Apache-2.0",
        "Teacher the AnaHD model was distilled from; not distributed",
    ),
)


NOTICES = (
    "## FFmpeg (GPL)",
    "",
    'The installer includes `ffmpeg.exe` and `ffprobe.exe` from the FFmpeg 9.0.2 "full"',
    "Windows build of https://www.gyan.dev/ffmpeg/builds/. That build enables GPL parts",
    "(x264, x265 and others), so as distributed it is licensed under the GNU General Public",
    "License, version 3 or later. Its licence text and the build's README (every library it",
    "contains, with versions) are next to it in the install folder's `ffmpeg` folder",
    "(`FFmpeg-LICENSE`, `FFmpeg-README.txt`).",
    "",
    "- AnaHD runs FFmpeg as a separate program and does not link to it; AnaHD's own code",
    "  stays under the MIT licence.",
    "- Source code: FFmpeg's source is at https://ffmpeg.org/download.html (release 9.0.2 at",
    "  https://ffmpeg.org/releases/); the build's scripts and library versions are at",
    "  https://www.gyan.dev/ffmpeg/builds/.",
    "- Written offer: for three years after each AnaHD release, the complete corresponding",
    "  source of the FFmpeg build bundled with it is available on request, at no charge",
    "  beyond the cost of providing it. Ask through this repository's issues.",
    "- You may replace the bundled FFmpeg with your own build: put your `ffmpeg.exe` and",
    "  `ffprobe.exe` in the install folder's `ffmpeg` folder.",
    "",
    "FFmpeg is a trademark of Fabrice Bellard, originator of the FFmpeg project.",
    "",
    "## NVIDIA CUDA",
    "",
    "PyTorch's CUDA build, bundled in the installer, includes NVIDIA runtime libraries (the",
    "CUDA runtime, cuBLAS, cuFFT, cuDNN and related DLLs). They are © NVIDIA Corporation and",
    "are redistributed unmodified, as their licences allow for redistributable files:",
    "",
    "- CUDA Toolkit End User License Agreement (redistributable runtime files, Attachment A):",
    "  https://docs.nvidia.com/cuda/eula/",
    "- cuDNN Software License Agreement:",
    "  https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html",
    "",
    "They are not covered by AnaHD's MIT licence and are only for use as part of AnaHD.",
    "NVIDIA, CUDA and cuDNN are trademarks of NVIDIA Corporation; AnaHD is not made,",
    "endorsed or supported by NVIDIA. AnaHD needs an NVIDIA graphics card and driver, which",
    "are installed separately under NVIDIA's own terms.",
)


def _licence(dist: md.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return str(expr)
    classifiers = [
        c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")
    ]
    if classifiers:
        return ", ".join(sorted(set(classifiers)))
    text = (meta.get("License") or "").strip().splitlines()
    return text[0][:80] if text else "see its licence file"


def runtime_distributions(
    root: str = "anahd", extras: tuple[str, ...] = ("app",)
) -> list[md.Distribution]:
    """Every installed distribution ``root[extras]`` needs at run time (not itself)."""
    env = default_environment()
    seen: dict[str, md.Distribution] = {}
    todo: list[tuple[str, tuple[str, ...]]] = [(root, extras)]
    while todo:
        name, wanted = todo.pop()
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        for line in dist.requires or []:
            req = Requirement(line)
            ok = req.marker is None or any(
                req.marker.evaluate({**env, "extra": e}) for e in ("", *wanted)
            )
            if not ok:
                continue
            key = canonicalize_name(req.name)
            if key not in seen:
                try:
                    seen[key] = md.distribution(req.name)
                except md.PackageNotFoundError:
                    continue
                todo.append((req.name, tuple(req.extras)))
    return sorted(seen.values(), key=lambda d: canonicalize_name(d.metadata["Name"]))


def markdown() -> str:
    lines = [
        "# Third-party licences",
        "",
        "AnaHD itself, model weights included, is MIT licensed ([LICENSE](LICENSE)). The",
        "Windows installer bundles the components below; their licence texts are in the",
        "install folder under `licenses`. This file is generated by",
        "`scripts/third_party.py` from the locked dependencies.",
        "",
        "## Python packages",
        "",
        "| Package | Version | Licence |",
        "|---|---|---|",
    ]
    for dist in runtime_distributions():
        name = dist.metadata["Name"]
        lines.append(f"| {name} | {dist.version} | {_licence(dist)} |")
    lines += ["", "## Other components", "", "| Component | Licence | Notes |", "|---|---|---|"]
    for name, licence, note in OTHERS:
        lines.append(f"| {name} | {licence} | {note} |")
    lines += ["", *NOTICES]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the file is out of date")
    text = markdown()
    if ap.parse_args().check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUT.name} is out of date: run scripts/third_party.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
