"""Speed and time left for a clip being cleaned, steady enough to read.

The speed is measured over the last WINDOW seconds of the worker's own timestamps (not the
average since the start, which includes the slow start-up and drifts for minutes). No time
left is shown until the speed has settled: the last two HALF-second stretches agree
within STABLE. From then on the shown time counts down with the clock and moves towards
the measured one by at most CATCH_UP seconds per update, so it never jumps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

WINDOW = 10.0  # seconds of progress the speed is measured over
HALF = 5.0  # the two stretches compared to decide the speed has settled
STABLE = 0.05  # they agree within 5 %
CATCH_UP = 1.0  # seconds of correction per update at most


@dataclass
class Eta:
    samples: deque[tuple[float, int]] = field(default_factory=deque)  # (time, frames done)
    shown: float | None = None  # the time left last shown
    shown_at: float = 0.0

    def update(self, done: int, total: int, now: float) -> tuple[float, float | None]:
        """Frames per second, and seconds left (None while the speed is settling)."""
        if self.samples and done < self.samples[-1][1]:
            self.samples.clear()  # a new pass (a retry): start over
            self.shown = None
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[1][0] >= 2 * HALF:
            self.samples.popleft()
        fps = self._rate(now - WINDOW, now)
        if not total or fps <= 0:
            return fps, None
        raw = max(total - done, 0) / fps
        if self.shown is None:
            older, recent = self._rate(now - 2 * HALF, now - HALF), self._rate(now - HALF, now)
            span = now - self.samples[0][0]
            if span < 2 * HALF - 0.5 or older <= 0 or abs(recent - older) > STABLE * older:
                return fps, None
            self.shown = raw
        else:
            expected = self.shown - (now - self.shown_at)
            self.shown = expected + max(-CATCH_UP, min(CATCH_UP, raw - expected))
        self.shown = max(self.shown, 0.0)
        self.shown_at = now
        return fps, self.shown

    def _rate(self, start: float, end: float) -> float:
        """Frames per second between the samples nearest to ``start`` and ``end``."""
        inside = [(t, d) for t, d in self.samples if start - 0.5 <= t <= end + 0.5]
        if len(inside) < 2 or inside[-1][0] <= inside[0][0]:
            return 0.0
        (t0, d0), (t1, d1) = inside[0], inside[-1]
        return (d1 - d0) / (t1 - t0)
