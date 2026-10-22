"""The remaster settings: every slider and switch the app shows, with its label, range and
default, so the interface can be built from this one list.

Values are in the units the sliders show (mostly percent). 100 % is the approved look
(2026-10-21) for strengths that can go above it; 0 turns that part off where that makes
sense. Defaults reproduce the approved remaster exactly.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import Any

GROUPS = ("Clean", "Colour", "Look", "Effects", "Advanced")
SUPERVIEW = ("off", "stretch", "osd_out")

# How the app shows each group (the panel is generated from this and the fields' metadata).
GROUP_INFO: dict[str, dict[str, Any]] = {
    "Clean": {"title": "Clean"},
    "Colour": {"title": "Colour"},
    "Look": {"title": "Look"},
    "Effects": {"title": "Effects"},
    "Advanced": {"title": "Advanced / Experimental", "collapsed": True},
}


def _slider(
    label: str,
    default: float,
    lo: float,
    hi: float,
    group: str,
    help_: str,
    step: float = 1.0,
    unit: str = "%",
    toggle: dict[str, Any] | None = None,
) -> Any:
    """``toggle``: shown as a checkbox (``label``) that sets the value to ``off`` or back
    to the last value it had when on (first time: ``on``); the slider shows when on."""
    meta: dict[str, Any] = {
        "label": label,
        "kind": "slider",
        "min": lo,
        "max": hi,
        "step": step,
        "default": default,
        "unit": unit,
        "group": group,
        "help": help_,
    }
    if toggle:
        meta["toggle"] = toggle
    return field(default=default, metadata=meta)


def _choice(
    label: str,
    default: str,
    choices: tuple[str, ...],
    group: str,
    help_: str,
    toggle: dict[str, Any] | None = None,
    names: dict[str, str] | None = None,
) -> Any:
    """``toggle`` as for a slider; ``names``: what each choice is called on screen."""
    meta: dict[str, Any] = {
        "label": label,
        "kind": "choice",
        "choices": list(choices),
        "default": default,
        "group": group,
        "help": help_,
    }
    if toggle:
        meta["toggle"] = toggle
    if names:
        meta["names"] = names
    return field(default=default, metadata=meta)


def _switch(
    label: str,
    default: bool,
    group: str,
    help_: str,
    experimental: bool = False,
    needs: str = "",
) -> Any:
    """``needs``: as for a file."""
    meta = {
        "label": label,
        "kind": "switch",
        "default": default,
        "group": group,
        "help": help_,
        "experimental": experimental,
        "needs": needs,
    }
    return field(default=default, metadata=meta)


def _path(label: str, group: str, help_: str, file_filter: str = "", needs: str = "") -> Any:
    """``file_filter``: the file dialog's filter; ``needs``: the switch that must be on for
    this to apply (it is greyed out otherwise)."""
    meta = {
        "label": label,
        "kind": "path",
        "default": "",
        "group": group,
        "help": help_,
        "filter": file_filter,
        "needs": needs,
    }
    return field(default="", metadata=meta)


@dataclass(frozen=True)
class RemasterParams:
    # ----- Clean: what the restorer and repair take out -----
    cleanup_strength: float = _slider(
        "Clean-up strength", 100, 0, 100, "Clean",
        "Master amount for the learned clean-up; 0 = a plain upscale of the recording",
    )  # fmt: skip
    scan_lines: float = _slider(
        "Scan-line removal", 100, 0, 100, "Clean",
        "Horizontal lines and stripes running across the picture",
    )  # fmt: skip
    noise: float = _slider(
        "Noise & grain removal", 100, 0, 100, "Clean",
        "Grain, sparkle and flicker (also the restorer's detail recovery)",
    )  # fmt: skip
    static_repair: float = _slider(
        "Static & breakup repair", 100, 0, 100, "Clean",
        "Bands of static, colour snow and lost colour, rebuilt from the frames around them",
    )  # fmt: skip
    colour_cleanup: float = _slider(
        "Colour noise clean-up", 100, 0, 100, "Clean",
        "Blotchy colour and colour flicker, averaged over neighbouring frames "
        "(brightness untouched)",
    )  # fmt: skip
    stabilisation: float = _slider(
        "Smoothness (stabilisation)", 0, 0, 100, "Effects",
        "Takes out small shakes; flips, rolls and fast turns are never slowed down",
    )  # fmt: skip
    sharpen: float = _slider(
        "Sharpening", 0, 0, 100, "Effects", "Extra edge crispness on top of the clean-up",
    )  # fmt: skip
    # ----- Colour: correcting the camera -----
    lens_colour_fix: float = _slider(
        "Lens colour fix", 100, 0, 100, "Colour",
        "Removes the C03's blue edges and corners / warm centre",
    )  # fmt: skip
    warmth: float = _slider(
        "Warmth", 0, -100, 100, "Colour", "Cooler (blue) to warmer (orange)", unit="",
    )  # fmt: skip
    tint: float = _slider(
        "Tint", 0, -100, 100, "Colour", "Greener to more magenta", unit="",
    )  # fmt: skip
    saturation: float = _slider(
        "Saturation", 100, 0, 200, "Colour", "How colourful; 0 = black and white",
    )  # fmt: skip
    contrast: float = _slider(
        "Contrast", 100, 0, 200, "Colour", "Difference between darks and brights",
    )  # fmt: skip
    black_level: float = _slider(
        "Black level", 0, -100, 100, "Colour",
        "Lift (brighter shadows, more to see in them) to crush (deeper blacks); 0 = as is",
        unit="",
    )  # fmt: skip
    shadows: float = _slider(
        "Purple-shadow fix", 100, 0, 100, "Colour",
        "Takes the purple/blue cast out of dark areas",
    )  # fmt: skip
    # ----- Look: the film-style grade -----
    look_strength: float = _slider(
        "Look strength", 100, 0, 100, "Look",
        "How much of the C03 film look is applied; 0 = no grade",
        toggle={"label": "Look", "on": 100, "off": 0},
    )  # fmt: skip
    vignette: float = _slider(
        "Vignette", 100, 0, 200, "Look", "Darkened corners",
    )  # fmt: skip
    bloom: float = _slider(
        "Glow", 100, 0, 200, "Look", "Soft glow and red halation around bright lights",
    )  # fmt: skip
    # ----- Effects: off unless chosen -----
    motion_blur: float = _slider(
        "Motion blur", 0, 0, 100, "Effects",
        "Cinematic blur along the motion; 39 % matches the Resolve preset, 0 = off",
        toggle={"label": "Motion blur", "on": 39, "off": 0},
    )  # fmt: skip
    superview: str = _choice(
        "SuperView", "off", SUPERVIEW, "Effects",
        "Stretch 4:3 to 16:9, natural in the centre; osd_out also pushes the corners out",
        toggle={"label": "SuperView", "on": "stretch", "off": "off"},
        names={"off": "Off", "stretch": "Stretch", "osd_out": "Stretch, OSD out"},
    )  # fmt: skip
    # ----- Advanced -----
    osd_redraw: bool = _switch(
        "Redraw OSD from my font", False, "Advanced",
        "Re-draw the on-screen text crisply from a Betaflight font. Experimental: it can "
        "draw wrong characters",
        experimental=True,
    )  # fmt: skip
    osd_redraw_live: bool = _switch(
        "Also redraw moving numbers", False, "Advanced",
        "Redraw voltage, current, throttle and link values too. Off: they stay as recorded, "
        "cleaned of colour fringes, and only constant parts and verified timers are redrawn",
        experimental=True, needs="osd_redraw",
    )  # fmt: skip
    osd_font_path: str = _path(
        "OSD font (.mcm)", "Advanced",
        "Your Betaflight/MAX7456 font file; empty = redraw only the fixed labels, "
        "from the clip's own picture of them",
        file_filter="Betaflight OSD font (*.mcm)", needs="osd_redraw",
    )  # fmt: skip
    osd_cli_path: str = _path(
        "Betaflight CLI dump (optional)", "Advanced",
        "A 'dump' or 'diff' saved from the Betaflight CLI: the OSD is never redrawn "
        "outside the elements it lists",
        file_filter="Betaflight CLI dump (*.txt)", needs="osd_redraw",
    )  # fmt: skip

    def __post_init__(self) -> None:
        for f in fields(self):
            m = f.metadata
            v = getattr(self, f.name)
            if m["kind"] == "slider":
                if not isinstance(v, int | float) or not m["min"] <= v <= m["max"]:
                    raise ValueError(f"{f.name} = {v!r}: must be {m['min']}..{m['max']}")
            elif m["kind"] == "choice" and v not in m["choices"]:
                raise ValueError(f"{f.name} = {v!r}: must be one of {m['choices']}")

    # --- serialisation ---

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemasterParams:
        """Unknown keys are ignored and missing ones take their defaults, so settings saved
        by an older or newer version still load."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> RemasterParams:
        return cls.from_dict(json.loads(text))

    def with_(self, **changes: Any) -> RemasterParams:
        return replace(self, **changes)

    # --- what the pipeline needs to know ---

    @property
    def look_on(self) -> bool:
        """Anything in the colour/look stage that changes the picture."""
        return (
            self.colour_cleanup > 0
            or self.lens_colour_fix > 0
            or self.look_strength > 0
            or self.motion_blur > 0
            or self.superview != "off"
            or self.stabilisation > 0
            or self.sharpen > 0
            or self.black_level != 0
            or self.osd_redraw
        )


NO_LOOK = RemasterParams(colour_cleanup=0, lens_colour_fix=0, look_strength=0)
"""The cleaning alone, as before the C03 look (``--no-look``)."""


def schema() -> list[dict[str, Any]]:
    """Every setting with its metadata, in display order, for building the interface."""
    out = []
    for f in fields(RemasterParams):
        default = f.default if f.default is not MISSING else None
        out.append({"name": f.name, **dict(f.metadata), "default": default})
    return out
