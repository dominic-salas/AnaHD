"""Strip a training checkpoint down to the shipped model file (weights only, FP16).

python tools/export_model.py lab/run_v6/v6_final.pt anahd-v6.pt

The optimizer state is dropped and the weights are stored in FP16: inference casts the
network to FP16 anyway, so the output is bit-identical and the file is ~6x smaller.
Prints the SHA-256 to pin in ``anahd.model.weights``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from anahd.model.weights import sha256


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    st = torch.load(src, map_location="cpu", weights_only=True)
    net = {k: v.half() if v.is_floating_point() else v for k, v in st["net"].items()}
    torch.save({"net": net, "size": str(st["size"])}, dst)
    print(f"{dst}: {dst.stat().st_size / 1e6:.1f} MB, sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
