"""Desktop app: drop recordings in, clean them one after another, compare before/after.

Runs entirely on this computer. Needs the ``app`` extra (PySide6): ``uv sync --extra app``.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import sys


def main() -> int:
    # In the standalone build the worker process starts this same program;
    # freeze_support() runs the worker there instead of a second window.
    multiprocessing.freeze_support()
    if importlib.util.find_spec("PySide6") is None:
        print("The desktop app needs PySide6: uv sync --extra app", file=sys.stderr)
        return 1
    from anahd.app.window import run

    return run(sys.argv)
