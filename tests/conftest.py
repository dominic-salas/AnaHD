"""Measured numbers reported at the end of the run (``record_property("report", ...)``)
instead of pass/fail, for things that depend on the machine, such as speed."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


def pytest_terminal_summary(terminalreporter: Any) -> None:
    lines = [
        f"{report.nodeid.split('::')[-1]}: {value}"
        for reports in terminalreporter.stats.values()
        for report in reports
        if getattr(report, "when", None) == "call"
        for name, value in getattr(report, "user_properties", [])
        if name == "report"
    ]
    if lines:
        terminalreporter.section("measured")
        for line in lines:
            terminalreporter.write_line(line)


_REAL_DETECTOR = {"on": False}


@pytest.fixture(autouse=True, scope="session")
def _synthetic_recordings_are_analog() -> Iterator[None]:
    """The cleaning tests' synthetic recordings are analog by construction, but they are
    sharp (a 1280x720 60 fps picture with detail to the pixel grid) and the picture test in
    anahd.origin would take them for HDZero and route them down the digital path. For every
    test except those of the origin and digital-path modules, which are about that decision,
    the picture test says "not digital" (session scope, so module-scoped fixtures see it)."""
    from anahd import origin

    real = origin.looks_digital
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(origin, "looks_digital", lambda s: real(s) if _REAL_DETECTOR["on"] else False)
        yield


@pytest.fixture(autouse=True)
def _real_detector_where_it_is_tested(request: pytest.FixtureRequest) -> Iterator[None]:
    name = request.module.__name__.rsplit(".", 1)[-1]
    _REAL_DETECTOR["on"] = name in ("test_origin", "test_digital")
    yield
    _REAL_DETECTOR["on"] = False
