"""Transcript normalization.

`normalize_transcript()` turns raw ASR output into the default editable
transcript and reports flags for the reviewer:

  1. language-independent cleanup (`clean_text`): NFC, quotes/dashes/ellipsis,
     spacing around punctuation, invisible characters (ZWJ/ZWNJ kept);
  2. non-speech annotations such as "[music]" / "(हँसी)" are removed
     -> flag `annotation_removed` (the audio probably contains them);
  3. numbers -> words in the clip's language -> flag `numbers_expanded`
     (a human should confirm how the number was actually read);
     digits that could not be expanded -> flag `digits_remaining`;
  4. characters outside the language's script(s) + basic punctuation are
     removed -> flag `chars_removed` (mostly foreign letters, e.g. Hindi
     transcribed in Urdu script -> also `wrong_script`);
  5. consistent sentence punctuation (danda for Hindi / Hinglish).

Canonical script: Hindi and Hinglish are stored in Devanagari; English words
stay in Latin script exactly as the ASR wrote them. Romanization happens only
at export time (see translit.py).

`comparison_form()` is a stricter normalization used only to compute the
WER between two ASR outputs (case, punctuation, nukta, candrabindu and
nasal-conjunct spelling variants are ignored).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .numbers import expand_numbers

# Zero-width joiner / non-joiner are meaningful in Indic scripts (e.g. Malayalam chillu,
# Devanagari half-forms) and must be preserved; other invisible characters are dropped.
_ZWNJ, _ZWJ = "‌", "‍"
_INVISIBLE = re.compile(r"[​‎‏⁠﻿­]")

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ",
    "，": ",", "．": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    "॥": "।",  # double danda -> danda
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_MAP))

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:।])")
_MISSING_SPACE_AFTER = re.compile(r"([,!?;:।])(?=[^\s\d,.!?;:\"')\]।])")
_REPEATED_PUNCT = re.compile(r"([,!?;:।])\1+")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Language-independent cleanup; idempotent."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _INVISIBLE.sub("", t)
    t = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], t)
    t = "".join(ch for ch in t if ch in (_ZWNJ, _ZWJ) or unicodedata.category(ch)[0] != "C" or ch in "\t\n")
    t = _MULTI_SPACE.sub(" ", t)
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _REPEATED_PUNCT.sub(r"\1", t)
    t = _MISSING_SPACE_AFTER.sub(r"\1 ", t)
    return t.strip()


# ------------------------------------------------------------------ per-language rules
SCRIPT_RANGES = {
    "deva": (0x0900, 0x097F),
    "taml": (0x0B80, 0x0BFF),
    "telu": (0x0C00, 0x0C7F),
    "mlym": (0x0D00, 0x0D7F),
}
LANG_SCRIPTS = {"hi": ("deva",), "hinglish": ("deva",), "en": (), "ta": ("taml",), "te": ("telu",), "ml": ("mlym",)}
NUMBER_LANG = {"hi": "hi", "hinglish": "hi", "en": "en", "ta": "ta", "te": "te", "ml": "ml"}
DANDA_LANGS = {"hi", "hinglish"}
ALLOWED_PUNCT = set(".,?!'\"-:; ")

_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>|♪[^♪]*♪|\*[^*]+\*|[♪♫🎵🎶]")
_ABBREV_DOT = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|St|Prof|Jr|Sr|vs|etc|Lt|Col|Capt|Gen|No)\.$", re.IGNORECASE)


def _char_allowed(ch: str, scripts: tuple[str, ...]) -> bool:
    if ch in ALLOWED_PUNCT or ch in (_ZWJ, _ZWNJ) or ch.isdigit():
        return True
    if ("a" <= ch.lower() <= "z") or ("À" <= ch <= "ɏ"):  # Latin (English words, names)
        return True
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in (SCRIPT_RANGES[s] for s in scripts))


def _sentence_end_danda(text: str) -> str:
    """'.' ending a Hindi sentence -> '।' (ellipses and abbreviations like 'Dr.' are kept)."""
    out = []
    tokens = re.split(r"(\s+)", text)
    for tok in tokens:
        if tok.endswith(".") and not tok.endswith("..") and not _ABBREV_DOT.search(tok) and len(tok) > 1:
            tok = tok[:-1] + "।"
        out.append(tok)
    return "".join(out)


def foreign_letter_ratio(text: str | None, language: str, roman: bool = False) -> float:
    """Share of letters outside the scripts allowed for `language` (Latin only for romanized rows)."""
    scripts = () if roman else LANG_SCRIPTS.get(language, ())
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if not _char_allowed(ch, scripts)) / len(letters)


@dataclass
class NormResult:
    text: str
    flags: set[str] = field(default_factory=set)


TEXT_FLAGS = {"annotation_removed", "numbers_expanded", "digits_remaining", "chars_removed", "wrong_script"}


def normalize_transcript(text: str | None, language: str, settings=None) -> NormResult:
    """Raw ASR text -> default editable transcript (+ flags for the reviewer)."""
    flags: set[str] = set()
    t = clean_text(text)
    if not t:
        return NormResult("", flags)

    stripped = _ANNOTATION.sub(" ", t)
    if stripped != t:
        flags.add("annotation_removed")
        t = clean_text(stripped)

    num_lang = NUMBER_LANG.get(language, language)
    if settings is not None:
        num_lang = settings.text.numbers_language.get(language, num_lang)
    t, expanded = expand_numbers(t, num_lang)
    if expanded:
        flags.add("numbers_expanded")
    if any(ch.isdigit() for ch in t):
        flags.add("digits_remaining")

    if language not in DANDA_LANGS:
        t = t.replace("।", ".")  # before the script filter, which would drop it
    scripts = LANG_SCRIPTS.get(language, ())
    letters = [ch for ch in t if ch.isalpha()]
    foreign = sum(1 for ch in letters if not _char_allowed(ch, scripts))
    if letters and foreign / len(letters) > 0.5:
        flags.add("wrong_script")  # e.g. Hindi audio transcribed in Urdu script
    kept = "".join(ch if _char_allowed(ch, scripts) else " " for ch in t)
    if kept.split() != t.split():
        flags.add("chars_removed")
    t = kept
    if language in DANDA_LANGS:
        t = _sentence_end_danda(t)
    return NormResult(clean_text(t), flags)


# ------------------------------------------------------------------ comparison form (for WER only)
_NUKTA_COMPOSED = str.maketrans({
    "क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज",
    "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य",
})
_MALAYALAM_CHILLU = str.maketrans({  # atomic chillu -> consonant + virama
    "ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്",
    "ൽ": "ല്", "ൾ": "ള്", "ൿ": "ക്",
})
_DEVA_NASAL_CONJUNCT = re.compile("[ङञणनम]्(?=[क-ह])")


def comparison_form(text: str | None, language: str) -> str:
    """Aggressive normalization so two ASR systems' spelling conventions don't count as errors."""
    t = normalize_transcript(text, language).text.lower()
    t = unicodedata.normalize("NFC", t).translate(_NUKTA_COMPOSED).replace("़", "")
    t = t.replace("ँ", "ं")                  # candrabindu -> anusvara
    t = _DEVA_NASAL_CONJUNCT.sub("ं", t)          # हिन्दी -> हिंदी
    t = t.translate(_MALAYALAM_CHILLU).replace(_ZWJ, "").replace(_ZWNJ, "")
    t = "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in t)
    return " ".join(t.split())
