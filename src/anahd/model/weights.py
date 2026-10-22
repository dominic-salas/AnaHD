"""Find the shipped restorer's weights: ``--model``, then ``ANAHD_MODEL``, then the user
cache, then the copy inside the standalone build, then a download from the project's GitHub
release (checked against a pinned SHA-256).

The weights are not in git (the repository stays small and footage-free). A file given with
``--model`` or ``ANAHD_MODEL`` is used as it is, so a retrained model can be tried without
touching the cache; the cached and downloaded file must match :data:`MODEL_SHA256`.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from anahd import bundle
from anahd.errors import AnaHDError

MODEL_NAME = "anahd-v6.pt"  # v6: m2 (3 M params, 2x), distilled from SeedVR2 7B, FP16 weights
MODEL_SHA256 = "cb9f331f6e42c8ac8f9a53293a617e727db370aa46be28313ce71776f2c59e4c"
# Placeholder until the model release is published.
MODEL_URL = f"https://github.com/dominic-salas/AnaHD/releases/download/model-v6/{MODEL_NAME}"
ENV = "ANAHD_MODEL"


def models_dir() -> Path:
    """%LOCALAPPDATA%\\AnaHD\\models on Windows, ~/.cache/AnaHD/models elsewhere."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "AnaHD" / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def resolve(explicit: Path | None = None, *, download: bool = True) -> Path:
    """The weights file to load, downloading it once if there is none."""
    given = explicit or (Path(os.environ[ENV]) if os.environ.get(ENV) else None)
    if given is not None:
        if not given.is_file():
            raise AnaHDError(f"model {given} not found")
        return given
    cached = models_dir() / MODEL_NAME
    if cached.is_file():
        if sha256(cached) == MODEL_SHA256:
            return cached
        cached.unlink()  # damaged or an older model under the same name: fetch it again
    shipped = bundle.models()  # the standalone build carries the model
    if (
        shipped is not None
        and (shipped / MODEL_NAME).is_file()
        and sha256(shipped / MODEL_NAME) == MODEL_SHA256
    ):
        return shipped / MODEL_NAME
    if not download:
        raise AnaHDError(f"no model in {cached.parent}; pass --model PATH")
    return _fetch(cached)


def _fetch(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    print(f"anahd: downloading the model to {dest} (once)")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, part.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise AnaHDError(
            f"could not download the model from {MODEL_URL} ({exc}); "
            f"put {MODEL_NAME} in {dest.parent} or pass --model PATH"
        ) from exc
    if sha256(part) != MODEL_SHA256:
        part.unlink()
        raise AnaHDError("the downloaded model does not match its known checksum")
    part.replace(dest)
    return dest
