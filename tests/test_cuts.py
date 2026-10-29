"""Cut markers: which frames are kept, and where an output moment comes from."""

from __future__ import annotations

import pytest

from anahd.cuts import Span, keep_spans, normalise, output_to_source, parse_cut, parse_time


def test_parse_times_and_cuts() -> None:
    assert parse_time("12.5") == 12.5
    assert parse_time("1:02.5") == 62.5
    assert parse_cut("0-4.5") == (0.0, 4.5)
    assert parse_cut("2:10-2:31") == (130.0, 151.0)
    for bad in ("5-2", "abc", "1-2-3", "-3"):
        with pytest.raises(ValueError):
            parse_cut(bad)


def test_overlapping_and_out_of_range_cuts_merge_and_clamp() -> None:
    assert normalise([(5, 8), (1, 2), (7, 9), (2, 3), (20, 30)], 10) == [(1, 3), (5, 9)]
    assert normalise([(-5, 1)], 10) == [(0, 1)]
    assert normalise([(10, 12)], 10) == []


def test_keep_spans_at_60_fps_are_frame_exact() -> None:
    fps, frames = 60.0, 300
    # a frame is cut when its time falls inside the cut: [0, 0.5) is frames 0..29
    spans = keep_spans([(0, 0.5), (2, 2.5), (4.75, 5)], frames, fps)
    assert spans == [Span(30, 120), Span(150, 285)]
    assert sum(s.frames for s in spans) == 225
    assert keep_spans([], frames, fps) == [Span(0, 300)]
    assert keep_spans([(0, 5)], frames, fps) == []  # everything cut
    # float noise must not move a boundary: 1/60 * 60 is frame 1, not 2
    assert keep_spans([(1 / 60, 2 / 60)], frames, fps)[0] == Span(0, 1)


def test_a_cut_shorter_than_a_frame_between_frames_cuts_nothing() -> None:
    assert keep_spans([(0.101, 0.105)], 300, 60.0) == [Span(0, 300)]


def test_output_time_maps_back_to_the_recording() -> None:
    fps = 60.0
    spans = [Span(30, 120), Span(150, 285)]
    assert output_to_source(0.0, spans, fps) == pytest.approx(0.5)
    assert output_to_source(89 / fps, spans, fps) == pytest.approx(119 / fps)
    assert output_to_source(90 / fps, spans, fps) == pytest.approx(150 / fps)  # across the cut
    assert output_to_source(10.0, spans, fps) == pytest.approx(284 / fps)  # past the end


@pytest.mark.parametrize("bad", ["abc", "5-2", "1-2-3"])
def test_a_bad_cut_on_the_command_line_names_the_format(
    bad: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from anahd import cli

    with pytest.raises(SystemExit):
        cli._parser().parse_args(["in.mp4", "--cut", bad])
    err = capsys.readouterr().err
    assert "START-END in seconds or m:ss, e.g. 0-4.5 or 1:05-1:20" in err, err
