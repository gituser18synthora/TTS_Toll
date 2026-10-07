"""Word error rate between two transcripts (used as ASR agreement)."""
from __future__ import annotations

import unicodedata


def tokenize(text: str | None) -> list[str]:
    """Lower-case, drop punctuation (keeping combining marks of Indic scripts), split on spaces."""
    if not text:
        return []
    t = unicodedata.normalize("NFC", text).lower()
    t = "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in t)
    return t.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb))
        prev = cur
    return prev[-1]


def wer(reference: str | None, hypothesis: str | None) -> float:
    """Word-level edit distance / reference length. Both empty -> 0.0; empty reference -> 1.0."""
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
