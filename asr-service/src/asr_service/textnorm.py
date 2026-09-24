"""Text normalization for WER: one rule for every engine and every reference."""

from __future__ import annotations

import re

from num2words import num2words

_NUMBER = re.compile(r"\d+")
_NON_WORD = re.compile(r"[^\w\s]|_")


def normalize(text: str) -> str:
    """Lowercase, ё→е, digits spelled out, punctuation dropped, whitespace collapsed.

    Digits are spelled in the nominative case, so an engine that writes "598 рублей"
    still loses to a reference "пятьсот девяносто восемь рублей" only when the case
    differs; this bias against digit-writing engines is noted in the comparison.
    """
    text = text.lower().replace("ё", "е")
    text = _NUMBER.sub(lambda m: " " + num2words(int(m.group()), lang="ru") + " ", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())
