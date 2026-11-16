"""Finding, installing and caching NVIDIA TensorRT, which AnaHD uses when it is there.

TensorRT is optional: ``available()`` says whether it can be used, and ``get_engine()``
returns None on any failure, so callers fall back to plain PyTorch. The packaged app does
not bundle it (its libraries are about 2.2 GB, over GitHub's 2 GB file limit), so it is
looked for in this order:

1. an importable ``tensorrt`` (source installs with the ``ai`` extra),
2. the folder ``%LOCALAPPDATA%\\AnaHD\\tensorrt`` (or ``ANAHD_TENSORRT``), which is where
   ``install()`` unpacks NVIDIA's free ``tensorrt-cu12`` wheels. Nothing else is touched.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

from anahd.errors import AnaHDError

log = logging.getLogger("anahd.trt")

#: The version AnaHD was tested with (PyTorch 2.11 + CUDA 12.8 build).
VERSION = "10.15.1.29"
DOWNLOAD_PAGE = "https://developer.nvidia.com/tensorrt-getting-started"
_BINDINGS_JSON = f"https://pypi.org/pypi/tensorrt-cu12-bindings/{VERSION}/json"
_LIBS_INDEX = "https://pypi.nvidia.com/tensorrt-cu12-libs/"
_SHIM = (
    "from tensorrt_bindings import *  # noqa: F403\n"
    "from tensorrt_bindings import __version__  # noqa: F401\n"
)

_warned: set[str] = set()
_runtimes: list[Any] = []  # an engine needs its runtime alive


@dataclass(frozen=True)
class Status:
    ok: bool
    reason: str  # why it is not usable, or where it was found
    version: str = ""


def _base() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "AnaHD"


def tensorrt_dir() -> Path:
    """Where ``install()`` puts TensorRT for the packaged app."""
    override = os.environ.get("ANAHD_TENSORRT")
    return Path(override) if override else _base() / "tensorrt"


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message)


def prepare() -> None:
    """Make a TensorRT in ``tensorrt_dir()`` importable (a plain ``import tensorrt`` then
    works for every caller). Cheap: imports nothing, and does nothing when TensorRT is
    already importable or the folder does not exist."""
    folder = tensorrt_dir()
    if str(folder) in sys.path or not (folder / "tensorrt_bindings").is_dir():
        return
    if importlib.util.find_spec("tensorrt") is not None:
        return
    _add_folder(folder)


def _add_folder(folder: Path) -> None:
    sys.path.insert(0, str(folder))
    libs = folder / "tensorrt_libs"
    if libs.is_dir() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(libs))
    os.environ["PATH"] = f"{libs}{os.pathsep}{os.environ.get('PATH', '')}"


@cache
def _probe() -> tuple[Status, ModuleType | None]:
    try:
        module = importlib.import_module("tensorrt")
        where = "installed"
        if str(tensorrt_dir()).lower() in str(getattr(module, "__file__", "")).lower():
            where = str(tensorrt_dir())
        return Status(True, where, str(module.__version__)), module
    except ImportError:
        pass
    except Exception as exc:  # a broken install must not stop the app
        return Status(False, f"TensorRT is installed but did not load ({exc})"), None
    folder = tensorrt_dir()
    if not (folder / "tensorrt_bindings").is_dir():
        return Status(False, "TensorRT is not installed"), None
    added = str(folder) not in sys.path
    if added:
        _add_folder(folder)
    try:
        importlib.invalidate_caches()
        module = importlib.import_module("tensorrt")
        return Status(True, str(folder), str(module.__version__)), module
    except Exception as exc:
        if added:
            sys.path.remove(str(folder))
        return Status(False, f"TensorRT in {folder} did not load ({exc})"), None


def status() -> Status:
    return _probe()[0]


def available() -> bool:
    return status().ok


def refresh() -> Status:
    """Look again (after ``install()``)."""
    _probe.cache_clear()
    return status()


def module() -> ModuleType | None:
    """The ``tensorrt`` module, or None."""
    return _probe()[1]


@cache
def _driver() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def cache_key(gpu: str, driver: str, trt_version: str) -> str:
    """A folder name for engines built for this GPU, driver and TensorRT."""
    raw = f"{gpu}_driver{driver}_trt{trt_version}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def engine_cache_dir(gpu: str | None = None) -> Path:
    """``%LOCALAPPDATA%\\AnaHD\\engines\\<gpu>_driver<d>_trt<v>``: an engine only runs on the
    GPU, and with the TensorRT, it was built with, and a new driver can change what it does."""
    if gpu is None:
        import torch

        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "nogpu"
    return _base() / "engines" / cache_key(gpu, _driver(), status().version or "none")


def get_engine(
    name: str, build: Callable[[ModuleType], bytes | None], *, gpu: str | None = None
) -> Any | None:
    """A deserialized TensorRT engine for ``name``, built with ``build(tensorrt)`` (which
    returns the serialized engine) the first time and loaded from the cache after. Returns
    None on any failure, logged once, so the caller runs its PyTorch path instead."""
    trt = module()
    if trt is None:
        _warn_once("missing", f"TensorRT not used: {status().reason}")
        return None
    try:
        path = engine_cache_dir(gpu) / f"{name}.engine"
        if not path.is_file():
            data = build(trt)
            if data is None:
                raise AnaHDError("TensorRT could not build the engine")
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_bytes(bytes(data))
            part.replace(path)
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            path.unlink(missing_ok=True)  # damaged or from another build: rebuilt next time
            raise AnaHDError("TensorRT could not load the cached engine")
        _runtimes.append(runtime)
        return engine
    except Exception as exc:
        _warn_once(name, f"TensorRT not used for {name}: {exc}")
        return None


# ----- installing for the packaged app -----


def _get(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            return bytes(resp.read())
    except OSError as exc:
        raise AnaHDError(f"could not reach {url}: {exc}") from exc


def _download(url: str, dest: Path, sha256: str, progress: Callable[[float], None] | None) -> None:
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total)
    except OSError as exc:
        raise AnaHDError(f"TensorRT download failed: {exc}") from exc
    if digest.hexdigest() != sha256:
        raise AnaHDError("the downloaded TensorRT file does not match its checksum")


def _wheel_urls() -> list[tuple[str, str]]:
    """(url, sha256) of the Windows bindings wheel for this Python and the libraries wheel."""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}-none-win_amd64"
    files = json.loads(_get(_BINDINGS_JSON))["urls"]
    bindings = next((f for f in files if tag in f["filename"]), None)
    if bindings is None:
        raise AnaHDError(f"no TensorRT {VERSION} bindings for Python {sys.version_info[:2]}")
    wanted = f"tensorrt_cu12_libs-{VERSION}-py2.py3-none-win_amd64.whl"
    html = _get(_LIBS_INDEX).decode("utf-8", "replace")
    found = re.search(rf'href="([^"#]*{re.escape(wanted)})#sha256=([0-9a-f]+)"', html)
    if found is None:
        raise AnaHDError(f"NVIDIA's package index has no {wanted}")
    libs = urllib.parse.urljoin(_LIBS_INDEX, found.group(1))
    return [(bindings["url"], bindings["digests"]["sha256"]), (libs, found.group(2))]


def install(progress: Callable[[str, float], None] | None = None) -> Status:
    """Download NVIDIA's free TensorRT (about 2.2 GB) into ``tensorrt_dir()``. Raises
    AnaHDError with a plain message on failure; leaves nothing half-installed."""
    if sys.platform != "win32":
        raise AnaHDError("the TensorRT download is for Windows; install tensorrt-cu12 with pip")
    target = tensorrt_dir()
    stage = target.with_name(target.name + ".new")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        urls = _wheel_urls()
        for i, (url, sha) in enumerate(urls):
            wheel = stage / f"part{i}.whl"

            def report(fraction: float, i: int = i) -> None:
                if progress:
                    progress("Downloading TensorRT", (i + fraction) / len(urls))

            _download(url, wheel, sha, report)
            with zipfile.ZipFile(wheel) as zf:
                zf.extractall(stage)
            wheel.unlink()
        shim = stage / "tensorrt"
        shim.mkdir(exist_ok=True)
        (shim / "__init__.py").write_text(_SHIM, encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)
        stage.replace(target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return refresh()
