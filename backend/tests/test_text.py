"""Transcript cleanup and WER."""
from __future__ import annotations

import pytest

from app.pipeline.text_norm import clean_text
from app.pipeline.wer import tokenize, wer
from app.schemas import check_speaker_name


@pytest.mark.parametrize("raw, expected", [
    ("  Hello   world  ", "Hello world"),
    ("He said “hi” — then left…", 'He said "hi" - then left...'),
    ("Wait ,what ?", "Wait, what?"),
    ("one,two", "one, two"),
    ("3:30 pm", "3:30 pm"),
    ("नमस्ते ।  आप कैसे हैं॥", "नमस्ते। आप कैसे हैं।"),
    ("zero​width", "zerowidth"),
    ("", ""),
    (None, ""),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_keeps_zwj_zwnj_for_indic_scripts():
    malayalam = "അവന്‍ വന്നു"  # chillu written with ZWJ
    hindi = "क्‍ष"
    assert "‍" in clean_text(malayalam)
    assert "‍" in clean_text(hindi)
    assert clean_text("a‌b") == "a‌b"


def test_clean_text_is_idempotent():
    s = "  “Hello” ,world …  नमस्ते ।"
    assert clean_text(clean_text(s)) == clean_text(s)


def test_wer_basic_cases():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert wer("the cat sat", "The cat, sat.") == 0.0  # case + punctuation ignored
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert wer("the cat sat", "the cat sat down") == pytest.approx(1 / 3)
    assert wer("the cat sat", "cat") == pytest.approx(2 / 3)
    assert wer("", "") == 0.0
    assert wer("", "something") == 1.0
    assert wer("a b", "") == 1.0


def test_wer_indic_tokens_keep_vowel_signs():
    assert tokenize("मैं घर जा रहा हूँ।") == ["मैं", "घर", "जा", "रहा", "हूँ"]
    assert wer("मैं घर जा रहा हूँ", "मै घर जा रहा हूँ") == pytest.approx(1 / 5)


@pytest.mark.parametrize("name, ok", [("ravi", True), ("Ravi Kumar", True), ("रवि", True), ("a:b", False), ("x|y", False), ("  ", False), ("--", False)])
def test_speaker_name_rules(name, ok):
    if ok:
        assert check_speaker_name(name)
    else:
        with pytest.raises(ValueError):
            check_speaker_name(name)
