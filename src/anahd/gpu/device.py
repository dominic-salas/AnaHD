"""The CUDA device, checked against the GPU generations this PyTorch build has code for."""

from __future__ import annotations

import torch

from anahd.errors import AnaHDError

OLDEST = "a GeForce GTX 16 or RTX 20 series card or newer"


def runs_on(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Whether a build compiled for ``arch_list`` (``sm_75``, ``compute_90``, ...) runs on a
    card of this compute capability: same-major binaries run on later minors, PTX
    (``compute_``) on any later card."""
    for arch in arch_list:
        kind, _, digits = arch.partition("_")
        built = (int(digits[:-1]), int(digits[-1]))
        if kind == "sm" and built[0] == capability[0] and built[1] <= capability[1]:
            return True
        if kind == "compute" and built <= capability:
            return True
    return False


def cuda_device() -> torch.device:
    """The GPU to clean on, or a plain explanation of why there is none."""
    if not torch.cuda.is_available():
        raise AnaHDError(
            "no NVIDIA graphics card found, or its driver is too old (CUDA 12.8 or newer "
            "is needed): update the driver, or use the command line's --cpu (slow)"
        )
    capability = torch.cuda.get_device_capability()
    if not runs_on(capability, torch.cuda.get_arch_list()):
        name = torch.cuda.get_device_name()
        raise AnaHDError(
            f"{name} is older than AnaHD supports (compute {capability[0]}.{capability[1]}); "
            f"it needs {OLDEST}, or use the command line's --cpu (slow)"
        )
    return torch.device("cuda")
