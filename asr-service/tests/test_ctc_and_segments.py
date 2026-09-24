import pytest

from asr_service.ctc import ctc_words
from asr_service.segment import words_to_segments
from asr_service.textnorm import normalize
from asr_service.types import Word

LABELS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
SPACE = LABELS.index(" ")
BLANK = SPACE + 1
A, D, N = LABELS.index("а"), LABELS.index("д"), LABELS.index("н")


def test_ctc_words_collapse_repeats_and_split_at_space():
    # "да нна": repeated д collapses, н-blank-н stays double
    path = [BLANK, D, D, A, BLANK, SPACE, SPACE, N, BLANK, N, A, A, BLANK]
    words = ctc_words(path, LABELS, SPACE, frame_time=lambda f: f * 0.03)
    assert [x.text for x in words] == ["да", "нна"]
    assert words[0].start == pytest.approx(0.03) and words[0].end == pytest.approx(0.12)
    assert words[1].start == pytest.approx(0.21) and words[1].end == pytest.approx(0.36)


def test_ctc_words_empty_path():
    assert ctc_words([BLANK, SPACE, BLANK], LABELS, SPACE, frame_time=float) == []


def test_segments_split_on_long_pauses():
    words = [Word(text=t, start=s, end=e) for t, s, e in [("а", 0, 0.3), ("б", 0.5, 0.8), ("в", 2.0, 2.4)]]
    segments = words_to_segments(words, max_pause=0.8)
    assert [s.text for s in segments] == ["а б", "в"]
    assert (segments[1].start, segments[1].end) == (2.0, 2.4)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Всё, ЁЖИК!", "все ежик"),
        ("вернут 598 рублей", "вернут пятьсот девяносто восемь рублей"),
        ("  кино-пакет  плюс ", "кино пакет плюс"),
    ],
)
def test_normalize(raw: str, expected: str):
    assert normalize(raw) == expected
