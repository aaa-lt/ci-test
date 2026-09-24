from itertools import pairwise

import numpy as np
import pytest

from asr_service.chunking import FRAME, Window, plan_windows, speech_regions, stitch
from asr_service.types import Word


def energy(spans: list[tuple[float, float]], total: float, quiet: list[float] = ()) -> np.ndarray:
    """-80 dB silence, -20 dB speech in `spans`, -40 dB dips at `quiet` points."""
    e = np.full(int(total / FRAME), -80.0)
    for s, t in spans:
        e[int(s / FRAME) : int(t / FRAME)] = -20.0
    for q in quiet:
        e[int(q / FRAME) : int(q / FRAME) + 3] = -40.0
    return e


def test_speech_regions_merge_short_gaps_and_pad():
    e = energy([(1.0, 2.0), (2.2, 3.0), (6.0, 7.0)], 10.0)
    regions = speech_regions(e, min_gap=0.3, pad=0.2)
    assert regions == pytest.approx([(0.8, 3.2), (5.8, 7.2)])


def test_speech_regions_drop_clicks():
    e = energy([(1.0, 1.06), (4.0, 5.0)], 10.0)
    assert len(speech_regions(e)) == 1


def test_regions_are_packed_into_windows_cut_in_silence():
    regions = [(0.0, 5.0), (6.0, 12.0), (13.0, 20.0), (21.0, 30.0)]
    windows = plan_windows(regions, energy(regions, 31.0), max_len=15.0)
    assert [(w.start, w.end) for w in windows] == [(0.0, 12.0), (13.0, 20.0), (21.0, 30.0)]
    # cores meet halfway through each gap, so every instant belongs to exactly one window
    assert [(w.core_start, w.core_end) for w in windows] == [(0.0, 12.5), (12.5, 20.5), (20.5, 30.0)]
    assert all(w.end - w.start <= 15.0 for w in windows)


def test_long_region_is_split_with_overlap_at_the_quietest_point():
    region = [(0.0, 50.0)]
    e = energy(region, 51.0, quiet=[20.0, 41.0])
    windows = plan_windows(region, e, max_len=24.0, overlap=1.0, search=6.0)
    assert all(w.end - w.start <= 24.0 + 1e-9 for w in windows)
    cuts = [w.core_start for w in windows[1:]]
    assert cuts == pytest.approx([20.0, 41.0])
    for a, b in pairwise(windows):
        assert a.core_end == b.core_start
        assert a.end == pytest.approx(b.core_start + 1.0)  # overlap on both sides of the cut
        assert b.start == pytest.approx(a.core_end - 1.0)
    assert windows[0].start == 0.0 and windows[-1].end == 50.0


def test_split_rejects_overlap_that_leaves_no_step():
    with pytest.raises(ValueError):
        plan_windows([(0.0, 30.0)], energy([(0.0, 30.0)], 31.0), max_len=2.0, overlap=1.0)


def w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_stitch_takes_each_boundary_word_once():
    windows = [Window(0.0, 11.0, 0.0, 10.0), Window(9.0, 20.0, 10.0, 20.0)]
    left = [w("раз", 1, 2), w("два", 8.5, 9.2), w("три", 9.8, 10.4), w("четы", 10.6, 11.0)]
    right = [w("два", 9.0, 9.2), w("три", 9.82, 10.41), w("четыре", 10.6, 11.3), w("пять", 15, 16)]
    merged = stitch(windows, [left, right])
    assert [x.text for x in merged] == ["раз", "два", "три", "четыре", "пять"]


def test_stitch_drops_near_duplicate_across_cut():
    windows = [Window(0.0, 11.0, 0.0, 10.0), Window(9.0, 20.0, 10.0, 20.0)]
    # the same word, midpoint 9.99 in the left window and 10.01 in the right one
    merged = stitch(windows, [[w("да", 9.8, 10.18)], [w("да", 9.82, 10.2)]])
    assert [x.text for x in merged] == ["да"]


def test_stitch_keeps_words_outside_first_and_last_cores():
    windows = [Window(0.0, 5.0, 0.5, 4.0), Window(4.0, 9.0, 4.0, 8.5)]
    merged = stitch(windows, [[w("а", 0.1, 0.3)], [w("б", 8.6, 8.9)]])
    assert [x.text for x in merged] == ["а", "б"]
