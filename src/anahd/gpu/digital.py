"""The digital path: an HDZero recording is upscaled without any analog cleaning.

A digital FPV link has none of the analog damage AnaHD repairs: no snow, no line tearing,
no colour bleed, no sync loss. Its own trouble is compression, and the damage when the
link breaks up is made of coded blocks (a different thing, which the analog repair would
misread). So the clip skips alignment, repair, destriping and derainbowing and is not
blended with its neighbours; what is left is the upscale and the look.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

from anahd.pipeline import Measured
from anahd.remaster.params import RemasterParams

if TYPE_CHECKING:
    from anahd.gpu.pipeline import GpuConfig


def config(gcfg: GpuConfig) -> GpuConfig:
    """The GPU settings for the digital path: no repair, no restorer (it is trained on
    analog damage), the SPAN upscaler for the 2x."""
    return replace(gcfg, digital=True, repair=False, learned=None, ai=True)


def restore(y: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The planes as recorded (full range 0..255, chroma at half size)."""
    return y, u, v


def without_overlay(meas: Measured) -> Measured:
    """A digital recording carries no OSD; what the finder reported on a still scene is
    scenery, so nothing is protected from the upscale."""
    none = np.zeros_like(meas.osd)
    regions = None if meas.osd_regions is None else np.zeros_like(meas.osd_regions)
    return replace(meas, osd=none, osd_regions=regions)


def params_for(params: RemasterParams) -> RemasterParams:
    """The remaster settings without what only analog video needs: no OSD redraw, and none of
    the analog colour corrections (the C03 lens fix, colour noise, purple shadows)."""
    return params.with_(
        osd_redraw=False,
        osd_redraw_live=False,
        lens_colour_fix=0,  # the C03's lens, not this camera's
        colour_cleanup=0,  # analog colour noise
        shadows=0,  # the analog purple cast
    )
