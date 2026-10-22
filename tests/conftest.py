"""Measured numbers reported at the end of the run (``record_property("report", ...)``)
instead of pass/fail, for things that depend on the machine, such as speed."""

from __future__ import annotations

from typing import Any


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
