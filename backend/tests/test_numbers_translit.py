"""Numbers -> words (en/hi/ta/te/ml), transcript normalization flags, romanization."""
from __future__ import annotations

import pytest

from app.pipeline.numbers import en_int, expand_numbers, hi_int, hi_year, ml_int, ta_int, te_int
from app.pipeline.text_norm import comparison_form, normalize_transcript
from app.pipeline.translit import delete_schwas, to_roman
from app.pipeline.transcripts import agreement_wer


@pytest.mark.parametrize("n, words", [
    (0, "शून्य"), (5, "पाँच"), (21, "इक्कीस"), (99, "निन्यानवे"), (100, "एक सौ"), (101, "एक सौ एक"),
    (1500, "एक हज़ार पाँच सौ"), (250000, "दो लाख पचास हज़ार"), (10**7, "एक करोड़"),
    (1234567, "बारह लाख चौंतीस हज़ार पाँच सौ सड़सठ"),
])
def test_hindi_numbers(n, words):
    assert hi_int(n) == words


def test_hindi_years():
    assert hi_year(1947) == "उन्नीस सौ सैंतालीस"
    assert hi_year(2024) == "दो हज़ार चौबीस"


@pytest.mark.parametrize("n, ta, te, ml", [
    (21, "இருபத்தொன்று", "ఇరవై ఒకటి", "ഇരുപത്തിയൊന്ന്"),
    (23, "இருபத்து மூன்று", "ఇరవై మూడు", "ഇരുപത്തിമൂന്ന്"),
    (100, "நூறு", "వంద", "നൂറ്"),
    (101, "நூற்றொன்று", "నూట ఒకటి", "നൂറ്റിയൊന്ന്"),
    (1000, "ஆயிரம்", "వెయ్యి", "ആയിരം"),
    (2000, "இரண்டாயிரம்", "రెండు వేలు", "രണ്ടായിരം"),
    (100000, "ஒரு லட்சம்", "ఒక లక్ష", "ഒരു ലക്ഷം"),
    (250000, "இரண்டு லட்சத்து ஐம்பதாயிரம்", "రెండు లక్షల యాభై వేలు", "രണ്ട് ലക്ഷത്തി അമ്പതിനായിരം"),
])
def test_dravidian_numbers(n, ta, te, ml):
    assert ta_int(n) == ta
    assert te_int(n) == te
    assert ml_int(n) == ml


def test_english_numbers_use_indian_grouping():
    assert en_int(150000) == "one lakh fifty thousand"
    assert expand_numbers("about 50% of Rs. 1,25,000", "en")[0] == "about fifty percent of one lakh twenty-five thousand rupees"
    assert expand_numbers("on 12th May 2024", "en")[0] == "on twelfth May twenty twenty-four"
    assert expand_numbers("call 9876543210", "en")[0] == "call nine eight seven six five four three two one zero"
    assert expand_numbers("it costs $1.87", "en")[0] == "it costs one dollar eighty-seven cents"
    assert expand_numbers("And $.60 of it", "en")[0] == "And sixty cents of it"   # whisper writes cents like this
    assert expand_numbers("₹.5 बचे", "hi")[0] == "पचास पैसे बचे"


def test_expand_numbers_in_running_indic_text():
    assert expand_numbers("मैंने ₹500 दिए", "hi")[0] == "मैंने पाँच सौ रुपये दिए"
    assert expand_numbers("२.५ किलो", "hi")[0] == "दो दशमलव पाँच किलो"   # Devanagari digits too
    assert expand_numbers("இது 21 ரூபாய்", "ta")[0] == "இது இருபத்தொன்று ரூபாய்"
    assert expand_numbers("₹500", "te")[0] == "ఐదు వందల రూపాయలు"      # attributive before a noun
    assert expand_numbers("₹2,000", "te")[0] == "రెండు వేల రూపాయలు"
    assert expand_numbers("500", "te")[0] == "ఐదు వందలు"                # standalone keeps the plural
    assert expand_numbers("no digits here", "en") == ("no digits here", False)


def test_normalize_transcript_flags_and_punctuation():
    r = normalize_transcript("[संगीत] मैं 5 बजे आऊँगा. 😀", "hi")
    assert r.text == "मैं पाँच बजे आऊँगा।"
    assert r.flags == {"annotation_removed", "numbers_expanded", "chars_removed"}
    # Hinglish keeps English words in Latin script; "Dr." is not turned into a danda
    r = normalize_transcript("Dr. Sharma ने meeting cancel कर दी.", "hinglish")
    assert r.text == "Dr. Sharma ने meeting cancel कर दी।"
    assert r.flags == set()
    # other languages use a plain full stop
    assert normalize_transcript("Hello there।", "en").text == "Hello there."
    assert normalize_transcript("ഞാൻ വരാം.", "ml").text == "ഞാൻ വരാം."


def test_wrong_script_is_flagged():
    r = normalize_transcript("کسی راجہ کے", "hi")  # Hindi audio, Urdu-script ASR output
    assert r.text == "" and {"wrong_script", "chars_removed"} <= r.flags


def test_comparison_form_ignores_spelling_variants():
    assert comparison_form("हिन्दी में बात!", "hi") == comparison_form("हिंदी में बात", "hi")
    assert comparison_form("ज़िंदगी", "hi") == comparison_form("जिंदगी", "hi")       # nukta
    assert comparison_form("चाँद", "hi") == comparison_form("चांद", "hi")           # candrabindu
    assert agreement_wer("मुझे 5 रुपये दो", "मुझे पाँच रुपये दो।", "hi") == 0.0       # digits vs words


@pytest.mark.parametrize("word, expected", [
    ("कमल", "kamal"), ("कमला", "kamla"), ("समझना", "samajhna"), ("बचपन", "bachpan"), ("धड़कन", "dharkan"),
    ("अपना", "apna"), ("समय", "samay"), ("मित्र", "mitra"), ("शब्द", "shabd"), ("हिंदी", "hindi"),
    ("संबंध", "sambandh"), ("चाँद", "chaand"), ("ज़िंदगी", "zindgi"), ("पानी", "paani"), ("नमस्ते", "namaste"),
])
def test_romanization_with_schwa_deletion(word, expected):
    assert to_roman(word) == expected


def test_romanization_keeps_latin_words_and_maps_danda():
    assert to_roman("आज मौसम बहुत अच्छा है। मैं office जा रहा हूँ।") == \
        "aaj mausam bahut achha hai. main office ja raha hoon."
    assert to_roman("मैंने स्वाद नहीं लिया", "iso") == "maiṁnē svād nahīṁ liyā"


def test_schwa_deletion_marks_with_virama():
    assert delete_schwas("कमल") == "कमल्"
    assert delete_schwas("समझना") == "समझ्ना"
    assert delete_schwas("क") == "क"  # single akshara keeps its vowel
