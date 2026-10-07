"""Devanagari -> Roman for Hindi / Hinglish exports.

Stored transcripts stay Devanagari (with English words in Latin script as the
ASR wrote them); this module only produces the `roman` view used by the
exporter (`script: roman | both`).

1. Hindi schwa deletion on the Devanagari side: a virama is inserted where
   Hindi drops the inherent "a" (word-final, and medially in V C _ C V
   context, applied right to left): कमल -> kamal, समझना -> samajhnā,
   बचपन -> bachpan.
2. indic-transliteration turns the result into ISO 15919 / IAST / ITRANS.
3. The default "hinglish" scheme maps ISO to the casual ASCII spelling people
   type (aaj, achha, bahut, hindi, sambandh), with an exception list for the
   most frequent function words (में -> mein, हूँ -> hoon).
Latin-script words pass through untouched.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

VIRAMA, NUKTA = "्", "़"
MODS = {"ँ", "ं", "ः"}  # candrabindu, anusvara, visarga
SCHEMES = ("hinglish", "iso", "iast", "itrans")


def _is_consonant(ch: str) -> bool:
    return "क" <= ch <= "ह" or "क़" <= ch <= "य़" or "ॸ" <= ch <= "ॿ"


def _is_matra(ch: str) -> bool:
    return "ा" <= ch <= "ौ" or ch in "ॎॏॕॖॗॢॣ"


def _is_vowel(ch: str) -> bool:
    return "ऄ" <= ch <= "औ" or ch in "ॠॡॲॳॴॵॶॷ"


@dataclass
class _Unit:
    text: str
    kind: str          # C consonant, V independent vowel, O other
    vowel: str = ""    # "a" inherent, "-" virama, else the matra
    mods: bool = False


def _units(word: str) -> list[_Unit]:
    units, i = [], 0
    while i < len(word):
        ch = word[i]
        if _is_consonant(ch) or _is_vowel(ch):
            j = i + 1
            kind, vowel = ("C", "a") if _is_consonant(ch) else ("V", ch)
            if kind == "C":
                if j < len(word) and word[j] == NUKTA:
                    j += 1
                if j < len(word) and word[j] == VIRAMA:
                    vowel, j = "-", j + 1
                elif j < len(word) and _is_matra(word[j]):
                    vowel, j = word[j], j + 1
            mods = False
            while j < len(word) and word[j] in MODS:
                mods, j = True, j + 1
            units.append(_Unit(word[i:j], kind, vowel, mods))
            i = j
        else:
            units.append(_Unit(ch, "O"))
            i += 1
    return units


def delete_schwas(word: str) -> str:
    """Insert a virama wherever Hindi pronunciation drops the inherent vowel."""
    units = _units(word)
    n = len(units)
    if n < 2 or any(u.kind == "O" for u in units):
        return word
    deleted = [False] * n

    def voiced(k: int) -> bool:  # unit k ends in a pronounced vowel
        u = units[k]
        return u.kind == "V" or (u.kind == "C" and u.vowel != "-" and not deleted[k])

    last = units[-1]
    if last.kind == "C" and last.vowel == "a" and not last.mods:
        conjunct = units[-2].kind == "C" and units[-2].vowel == "-"
        if not (conjunct and last.text[0] in "रयव"):  # मित्र, सत्य, स्वास्थ्य keep their final a
            deleted[-1] = True
    for i in range(n - 2, 0, -1):  # medial: V C [a] C V, right to left
        u = units[i]
        if u.kind == "C" and u.vowel == "a" and not u.mods and voiced(i - 1) \
                and units[i + 1].kind == "C" and voiced(i + 1):
            deleted[i] = True
    return "".join(u.text + (VIRAMA if deleted[k] else "") for k, u in enumerate(units))


# ---------------------------------------------------------------- ISO -> casual Hinglish
_ISO_MAP = {
    "ā": "aa", "ī": "ee", "ū": "oo", "ē": "e", "ō": "o", "ô": "o", "ê": "e",
    "r̥̄": "ri", "r̥": "ri", "l̥": "li",
    "ṅ": "n", "ñ": "n", "ṇ": "n", "m̐": "n", "ḥ": "h",
    "ch": "chh", "c": "ch",
    "ṭh": "th", "ṭ": "t", "ḍh": "dh", "ḍ": "d", "ṛh": "rh", "ṛ": "r",
    "ś": "sh", "ṣ": "sh", "k͟h": "kh", "ġ": "g", "ẏ": "y",
}
_FINAL = {"ā": "a", "ī": "i", "ū": "u"}
_LABIAL = ("p", "b", "m")
_ISO_KEYS = sorted(_ISO_MAP, key=len, reverse=True)

# frequent function words whose everyday spelling differs from the regular mapping
HINGLISH_WORDS = {
    "में": "mein", "मैं": "main", "हैं": "hain", "है": "hai", "हूँ": "hoon", "हूं": "hoon", "नहीं": "nahin",
    "हाँ": "haan", "हां": "haan", "यह": "yeh", "वह": "woh", "यहाँ": "yahaan", "वहाँ": "wahaan", "कहाँ": "kahaan",
    "क्यों": "kyon", "क्या": "kya", "और": "aur", "भी": "bhi", "ही": "hi", "की": "ki", "के": "ke", "का": "ka",
    "को": "ko", "से": "se", "तो": "to", "जी": "ji", "हम": "hum", "तुम": "tum", "आप": "aap", "कुछ": "kuch",
    "अच्छा": "achha", "अच्छी": "achhi", "अच्छे": "achhe", "मुझे": "mujhe", "तुम्हें": "tumhein", "उन्हें": "unhein",
    "एक": "ek", "पर": "par", "लेकिन": "lekin", "क्योंकि": "kyonki", "बहुत": "bahut", "था": "tha", "थी": "thi",
    "थे": "the", "गया": "gaya", "गई": "gayi", "रहा": "raha", "रही": "rahi", "रहे": "rahe",
}


def _iso_word_to_hinglish(iso: str) -> str:
    s = unicodedata.normalize("NFC", iso)
    out, i = [], 0
    while i < len(s):
        if s[i] == "ṁ":  # anusvara: m before labials, else n
            nxt = s[i + 1:i + 2]
            out.append("m" if nxt and nxt in _LABIAL else "n")
            i += 1
            continue
        for key in _ISO_KEYS:
            if s.startswith(key, i):
                end = i + len(key)
                rest = s[end:]
                final = not rest or rest in ("ṁ", "m̐", "ḥ")
                if key in _FINAL and final:
                    # nasal-final long vowels keep their length: haan, hoon; but nahin
                    out.append(_FINAL[key] if not rest or key == "ī" else _ISO_MAP[key])
                else:
                    out.append(_ISO_MAP[key])
                i = end
                break
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


_SCHEME_IDS = {"iso": "ISO", "iast": "IAST", "itrans": "ITRANS"}
_DEV_RUN = re.compile(r"[ऀ-ॣ०-ॿ]+")


@lru_cache(maxsize=50_000)
def roman_word(word: str, scheme: str = "hinglish") -> str:
    from indic_transliteration import sanscript

    if scheme == "hinglish" and word in HINGLISH_WORDS:
        return HINGLISH_WORDS[word]
    dev = delete_schwas(word)
    target = getattr(sanscript, _SCHEME_IDS.get(scheme, "ISO"))
    roman = sanscript.transliterate(dev, sanscript.DEVANAGARI, target)
    return _iso_word_to_hinglish(roman) if scheme == "hinglish" else unicodedata.normalize("NFC", roman)


def to_roman(text: str, scheme: str = "hinglish") -> str:
    """Romanize every Devanagari run; Latin words, digits and punctuation pass through."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown roman scheme {scheme!r}; expected one of {SCHEMES}")
    text = text.replace("।", ".").replace("॥", ".")
    return _DEV_RUN.sub(lambda m: roman_word(m.group(0), scheme), text)


def has_devanagari(text: str | None) -> bool:
    return bool(text) and any("ऀ" <= ch <= "ॿ" for ch in text)
