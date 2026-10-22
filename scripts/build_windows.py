"""Build the standalone Windows app: one folder with AnaHD.exe, anahd-cli.exe (the command
line), FFmpeg, the model and licences (after FPV Editor's build script).

    uv sync --extra app --group build
    uv run python scripts/build_windows.py --ffmpeg C:/path/to/ffmpeg/bin

The folder runs without Python, uv or the project; it needs an NVIDIA GPU and driver.
`packaging/installer.iss` (Inno Setup) turns it into an installer. ``--ffmpeg`` is a
folder with ffmpeg.exe and ffprobe.exe (and their DLLs for a shared build); without it the
app uses the FFmpeg on PATH. The model comes from ``--model`` or the user's model cache.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from third_party import runtime_distributions

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "AnaHD"
CLI_NAME = "anahd-cli"  # not "anahd": Windows file names ignore case (AnaHD.exe)
# Never needed by the product: training (and the SeedVR2 teacher), the optional SPAN path.
EXCLUDE = (
    "tkinter",
    "anahd.learn",
    "tensorrt",
    "spandrel",
    "onnx",
    "matplotlib",
    "pytest",
    "mypy",
    "IPython",
)
RAFT = "raft_small_C_T_V2-01064c6d.pth"


def write_icon(path: Path) -> None:
    """A multi-size .ico with PNG images (Windows Vista and later), drawn by the app."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtWidgets import QApplication

    from anahd.app.look import SIZES, icon_pixmap

    _ = QApplication.instance() or QApplication([])
    images = []
    for size in SIZES:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        icon_pixmap(size).save(buffer, "PNG")
        images.append((size, bytes(buffer.data().data())))
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blobs = b"", b""
    for size, png in images:
        side = 0 if size >= 256 else size  # 0 means 256 in the ICO format
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png), offset)
        blobs += png
        offset += len(png)
    path.write_bytes(header + entries + blobs)


def copy_ffmpeg(source: Path, target: Path) -> str:
    """Copy ffmpeg and ffprobe (and DLLs of a shared build); return its licence."""
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        if not (source / name).is_file():
            raise SystemExit(f"{source} has no {name}")
    target.mkdir(parents=True, exist_ok=True)
    for path in [source / "ffmpeg.exe", source / "ffprobe.exe", *source.glob("*.dll")]:
        shutil.copy2(path, target / path.name)
    for name in ("LICENSE", "LICENSE.txt", "COPYING.GPLv3", "COPYING.LGPLv2.1", "README.txt"):
        for folder in (source, source.parent):
            if (folder / name).is_file():
                shutil.copy2(folder / name, target / f"FFmpeg-{name}")
    version = subprocess.run(
        [str(target / "ffmpeg.exe"), "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return "GPL" if "--enable-gpl" in version else "LGPL"


def copy_models(target: Path, model: Path) -> None:
    """The restorer and RAFT-small's weights, so the first run needs no download."""
    from anahd.model import weights

    if weights.sha256(model) != weights.MODEL_SHA256:
        raise SystemExit(f"{model} is not the pinned {weights.MODEL_NAME}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, target / weights.MODEL_NAME)
    import torch.hub

    raft = Path(torch.hub.get_dir()) / "checkpoints" / RAFT
    if not raft.is_file():
        from torchvision.models.optical_flow import Raft_Small_Weights

        Raft_Small_Weights.DEFAULT.get_state_dict(progress=False)
    hub = target / "torch" / "hub" / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raft, hub / RAFT)


def copy_licences(target: Path, ffmpeg_licence: str | None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "LICENSE", target / "AnaHD-LICENSE.txt")
    shutil.copy2(ROOT / "THIRD_PARTY_LICENSES.md", target / "THIRD_PARTY_LICENSES.md")
    # Every bundled package (the dependency graph of anahd[app]), as listed in that file.
    for distribution in runtime_distributions():
        package = distribution.metadata["Name"]
        for file in distribution.files or []:
            name = file.name.upper()
            if "LICEN" in name or "COPYING" in name or "NOTICE" in name:
                source = Path(str(file.locate()))
                if source.is_file():  # flattened path: packages ship several LICENSE files
                    flat = str(file).replace("\\", "/").replace("/", "_")
                    shutil.copy2(source, target / f"{package}-{flat}")
    ffmpeg = (
        f"FFmpeg ({ffmpeg_licence} build) is included in the ffmpeg folder and runs as a "
        "separate program. Its source code is available from https://ffmpeg.org/download.html "
        "(see the version printed by ffmpeg -version)."
        if ffmpeg_licence
        else "FFmpeg is not included; the FFmpeg installed on this computer is used."
    )
    (target / "THIRD-PARTY-NOTICES.txt").write_text(
        "AnaHD is MIT licensed (AnaHD-LICENSE.txt). It includes:\n"
        "- Python (PSF License) and the PyInstaller bootloader (GPL-2.0 with an exception "
        "that allows distribution with any program)\n"
        "- Qt for Python: PySide6 and shiboken6 (LGPL-3.0), used unmodified as shared "
        "libraries; you may replace them with your own build of the same version\n"
        "- PyTorch and torchvision (BSD-3-Clause) with NVIDIA's CUDA runtime libraries "
        "(NVIDIA's licence terms, redistributed as PyTorch ships them)\n"
        "- RAFT-small optical-flow weights from torchvision (BSD-3-Clause)\n"
        "- NumPy (BSD-3-Clause), OpenCV (Apache-2.0)\n"
        "- The AnaHD restorer weights (MIT), trained with SeedVR2 (Apache-2.0) as teacher\n"
        f"- {ffmpeg}\n"
        "The licence texts of these packages are in this folder.\n",
        encoding="utf-8",
    )


def pyinstaller(
    build: Path, out: Path, name: str, module: str, icon: Path, *, windowed: bool
) -> Path:
    """One PyInstaller folder build of ``module``'s main(); returns the folder."""
    launcher = build / f"{name}_main.py"
    launcher.write_text(
        f"import sys\n\nfrom {module} import main\n\nsys.exit(main())\n", encoding="utf-8"
    )
    excludes = [arg for mod in EXCLUDE for arg in ("--exclude-module", mod)]
    subprocess.run(
        [
            *(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean"),
            "--windowed" if windowed else "--console",
            *("--name", name, "--icon", str(icon)),
            *("--distpath", str(out), "--workpath", str(build / "work" / name)),
            *("--specpath", str(build), *excludes),
            str(launcher),
        ],
        check=True,
        cwd=ROOT,
    )
    return out / name


def main() -> int:
    from anahd import __version__
    from anahd.model import weights

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ffmpeg", type=Path, help="folder with ffmpeg.exe and ffprobe.exe")
    parser.add_argument("--model", type=Path, help=f"{weights.MODEL_NAME} (default: the cache)")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output folder")
    args = parser.parse_args()
    model = args.model or weights.models_dir() / weights.MODEL_NAME
    build = ROOT / "build" / "windows"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    icon = build / "anahd.ico"
    write_icon(icon)
    app = pyinstaller(build, args.out, APP_NAME, "anahd.app", icon, windowed=True)
    # The command-line tool, in the same folder: its own small exe, the app's libraries.
    cli = pyinstaller(build, build / "cli", CLI_NAME, "anahd.cli", icon, windowed=False)
    shutil.copy2(cli / f"{CLI_NAME}.exe", app / f"{CLI_NAME}.exe")
    shutil.copytree(cli / "_internal", app / "_internal", dirs_exist_ok=True)
    # OpenCV's own video I/O (an FFmpeg build of ~30 MB) is never used: FFmpeg does that.
    for unused in (app / "_internal" / "cv2").glob("opencv_videoio_ffmpeg*.dll"):
        unused.unlink()
    licence = copy_ffmpeg(args.ffmpeg, app / "ffmpeg") if args.ffmpeg else None
    copy_models(app / "models", model)
    copy_licences(app / "licenses", licence)
    size = sum(p.stat().st_size for p in app.rglob("*") if p.is_file()) / 1e6
    print(f"Built {app} {__version__} ({size:.0f} MB, FFmpeg: {licence or 'not included'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
