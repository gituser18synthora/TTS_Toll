"""Numbers -> words, per language, using Indian grouping (lakh / crore).

English uses num2words (en_IN). Hindi, Tamil, Telugu and Malayalam are
implemented here (num2words has no hi/ta/ml and its Telugu misses plural
forms like లక్షల / వేల).

`expand_numbers(text, lang)` rewrites digits inside running text: currency,
percentages, decimals, ordinals (en), times (h:mm), grouped numbers
(1,25,000), years and plain integers. Long digit strings (phone numbers,
IDs) are read digit by digit. Any expansion is a guess at how the speaker
read the number, so callers flag those clips for review.
"""
from __future__ import annotations

import re
from collections.abc import Callable

# ---------------------------------------------------------------- Hindi
HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(HI_0_99) == 100


def hi_int(n: int) -> str:
    if n < 100:
        return HI_0_99[n]
    parts = []
    for value, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        q, n = divmod(n, value)
        if q:
            parts.append(f"{hi_int(q)} {word}")
    if n:
        parts.append(HI_0_99[n])
    return " ".join(parts)


def hi_year(n: int) -> str:
    if 1100 <= n <= 1999:  # उन्नीस सौ सैंतालीस
        h, r = divmod(n, 100)
        return f"{HI_0_99[h]} सौ" + (f" {HI_0_99[r]}" if r else "")
    return hi_int(n)


# ---------------------------------------------------------------- Tamil
TA_UNITS = "பூஜ்ஜியம் ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது".split()
TA_TEENS = "பத்து பதினொன்று பன்னிரண்டு பதின்மூன்று பதினான்கு பதினைந்து பதினாறு பதினேழு பதினெட்டு பத்தொன்பது".split()
TA_TENS = dict(zip(range(2, 10), "இருபது முப்பது நாற்பது ஐம்பது அறுபது எழுபது எண்பது தொண்ணூறு".split()))
TA_TENS_C = dict(zip(range(2, 10), "இருபத்து முப்பத்து நாற்பத்து ஐம்பத்து அறுபத்து எழுபத்து எண்பத்து தொண்ணூற்று".split()))
TA_HUND = dict(zip(range(1, 10), "நூறு இருநூறு முந்நூறு நானூறு ஐந்நூறு அறுநூறு எழுநூறு எண்ணூறு தொள்ளாயிரம்".split()))
TA_HUND_C = dict(zip(range(1, 10),
                     "நூற்று இருநூற்று முந்நூற்று நானூற்று ஐந்நூற்று அறுநூற்று எழுநூற்று எண்ணூற்று தொள்ளாயிரத்து".split()))
TA_SIGN = {"அ": "", "ஆ": "ா", "இ": "ி", "ஈ": "ீ", "உ": "ு", "ஊ": "ூ", "எ": "ெ", "ஏ": "ே", "ஐ": "ை", "ஒ": "ொ", "ஓ": "ோ", "ஔ": "ௌ"}


def ta_join(prefix: str, word: str) -> str:
    """Sandhi: a prefix ending in -u merges with a vowel-initial word (இருபத்து + ஒன்று -> இருபத்தொன்று)."""
    if prefix.endswith("ு") and word[:1] in TA_SIGN:
        return prefix[:-1] + TA_SIGN[word[0]] + word[1:]
    return f"{prefix} {word}"


def _ta_lt100(n: int) -> str:
    if n < 10:
        return TA_UNITS[n]
    if n < 20:
        return TA_TEENS[n - 10]
    t, u = divmod(n, 10)
    return TA_TENS[t] if u == 0 else ta_join(TA_TENS_C[t], TA_UNITS[u])


def _ta_lt1000(n: int) -> str:
    h, r = divmod(n, 100)
    if not h:
        return _ta_lt100(r)
    return TA_HUND[h] if r == 0 else ta_join(TA_HUND_C[h], _ta_lt100(r))


def _ta_thousands(q: int, combining: bool) -> str:
    if q == 1:
        word = "ஆயிரம்"
    else:
        head = _ta_lt100(q)
        word = head[:-1] + "ாயிரம்" if head.endswith("ு") else f"{head} ஆயிரம்"  # இரண்டு -> இரண்டாயிரம்
    return word[:-2] + "த்து" if combining else word  # ஆயிரம் -> ஆயிரத்து


def ta_int(n: int) -> str:
    if n < 1000:
        return _ta_lt1000(n)
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, rest = divmod(n, 1000)
    pieces = []
    if crore:
        word = "ஒரு கோடி" if crore == 1 else f"{ta_int(crore)} கோடி"
        pieces.append(word + ("யே" if lakh or thousand or rest else ""))
    if lakh:
        head = "ஒரு" if lakh == 1 else _ta_lt100(lakh)
        pieces.append(f"{head} {'லட்சத்து' if thousand or rest else 'லட்சம்'}")
    if thousand:
        pieces.append(_ta_thousands(thousand, combining=bool(rest)))
    if rest:
        pieces.append(_ta_lt1000(rest))
    return " ".join(pieces)


# ---------------------------------------------------------------- Telugu
TE_0_19 = ("సున్నా ఒకటి రెండు మూడు నాలుగు ఐదు ఆరు ఏడు ఎనిమిది తొమ్మిది "
           "పది పదకొండు పన్నెండు పదమూడు పద్నాలుగు పదిహేను పదహారు పదిహేడు పద్దెనిమిది పందొమ్మిది").split()
TE_TENS = dict(zip(range(2, 10), "ఇరవై ముప్పై నలభై యాభై అరవై డెబ్బై ఎనభై తొంభై".split()))


def _te_lt100(n: int) -> str:
    if n < 20:
        return TE_0_19[n]
    t, u = divmod(n, 10)
    return TE_TENS[t] + (f" {TE_0_19[u]}" if u else "")


def _te_lt1000(n: int) -> str:
    h, r = divmod(n, 100)
    if not h:
        return _te_lt100(r)
    if h == 1:
        return "వంద" if not r else f"నూట {_te_lt100(r)}"
    return f"{TE_0_19[h]} వందలు" if not r else f"{TE_0_19[h]} వందల {_te_lt100(r)}"


def te_int(n: int) -> str:
    if n < 1000:
        return _te_lt1000(n)
    parts = []
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, rest = divmod(n, 1000)
    tail = [lakh, thousand, rest]
    if crore:
        more = any(tail)
        parts.append("ఒక కోటి" if crore == 1 else f"{te_int(crore)} {'కోట్ల' if more else 'కోట్లు'}")
    if lakh:
        more = bool(thousand or rest)
        parts.append("ఒక లక్ష" if lakh == 1 else f"{_te_lt100(lakh)} {'లక్షల' if more else 'లక్షలు'}")
    if thousand:
        parts.append("వెయ్యి" if thousand == 1 else f"{_te_lt100(thousand)} {'వేల' if rest else 'వేలు'}")
    if rest:
        parts.append(_te_lt1000(rest))
    return " ".join(parts)


_TE_ATTRIBUTIVE = (("వందలు", "వందల"), ("వేలు", "వేల"), ("లక్షలు", "లక్షల"), ("కోట్లు", "కోట్ల"))


def te_attributive(words: str) -> str:
    """Before a noun Telugu uses the oblique form: ఐదు వందలు -> ఐదు వందల రూపాయలు."""
    for plain, oblique in _TE_ATTRIBUTIVE:
        if words.endswith(plain):
            return words[: -len(plain)] + oblique
    return words


# ---------------------------------------------------------------- Malayalam
ML_0_19 = ("പൂജ്യം ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച് ആറ് ഏഴ് എട്ട് ഒമ്പത് "
           "പത്ത് പതിനൊന്ന് പന്ത്രണ്ട് പതിമൂന്ന് പതിനാല് പതിനഞ്ച് പതിനാറ് പതിനേഴ് പതിനെട്ട് പത്തൊമ്പത്").split()
ML_TENS = dict(zip(range(2, 10), "ഇരുപത് മുപ്പത് നാല്പത് അമ്പത് അറുപത് എഴുപത് എൺപത് തൊണ്ണൂറ്".split()))
ML_TENS_C = dict(zip(range(2, 10), "ഇരുപത്തി മുപ്പത്തി നാല്പത്തി അമ്പത്തി അറുപത്തി എഴുപത്തി എൺപത്തി തൊണ്ണൂറ്റി".split()))
ML_HUND = dict(zip(range(1, 10), "നൂറ് ഇരുനൂറ് മുന്നൂറ് നാനൂറ് അഞ്ഞൂറ് അറുനൂറ് എഴുനൂറ് എണ്ണൂറ് തൊള്ളായിരം".split()))
ML_HUND_C = dict(zip(range(1, 10),
                     "നൂറ്റി ഇരുനൂറ്റി മുന്നൂറ്റി നാനൂറ്റി അഞ്ഞൂറ്റി അറുനൂറ്റി എഴുനൂറ്റി എണ്ണൂറ്റി തൊള്ളായിരത്തി".split()))
ML_THOUSANDS = {1: "ആയിരം", 2: "രണ്ടായിരം", 3: "മൂവായിരം", 4: "നാലായിരം", 5: "അയ്യായിരം", 6: "ആറായിരം",
                7: "ഏഴായിരം", 8: "എട്ടായിരം", 9: "ഒമ്പതിനായിരം", 10: "പതിനായിരം"}
ML_SIGN = {"അ": "", "ആ": "ാ", "ഇ": "ി", "ഈ": "ീ", "ഉ": "ു", "ഊ": "ൂ", "എ": "െ", "ഏ": "േ", "ഐ": "ൈ", "ഒ": "ൊ", "ഓ": "ോ", "ഔ": "ൌ"}


def ml_join(prefix: str, word: str) -> str:
    """ഇരുപത്തി + ഒന്ന് -> ഇരുപത്തിയൊന്ന് (glide യ before vowels); consonant-initial words attach directly."""
    if word[:1] in ML_SIGN:
        return prefix + "യ" + ML_SIGN[word[0]] + word[1:]
    return prefix + word


def _ml_lt100(n: int) -> str:
    if n < 20:
        return ML_0_19[n]
    t, u = divmod(n, 10)
    return ML_TENS[t] if u == 0 else ml_join(ML_TENS_C[t], ML_0_19[u])


def _ml_lt1000(n: int) -> str:
    h, r = divmod(n, 100)
    if not h:
        return _ml_lt100(r)
    if r == 0:
        return ML_HUND[h]
    # single units fuse (നൂറ്റിയൊന്ന്); tens stay a separate word (തൊള്ളായിരത്തി തൊണ്ണൂറ്റിയൊമ്പത്)
    return ml_join(ML_HUND_C[h], ML_0_19[r]) if r < 10 else f"{ML_HUND_C[h]} {_ml_lt100(r)}"


def _ml_thousands(q: int) -> str:
    if q in ML_THOUSANDS:
        return ML_THOUSANDS[q]
    word = _ml_lt100(q)
    if word.endswith("ത്"):
        return word[:-1] + "ിനായിരം"
    return (word[:-1] if word.endswith("്") else word) + "ായിരം"


def ml_int(n: int) -> str:
    if n < 1000:
        return _ml_lt1000(n)
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, rest = divmod(n, 1000)
    pieces = []
    if crore:
        pieces.append("ഒരു കോടി" if crore == 1 else f"{ml_int(crore)} കോടി")
    if lakh:
        word = "ഒരു ലക്ഷം" if lakh == 1 else f"{_ml_lt100(lakh)} ലക്ഷം"
        pieces.append(word[:-1] + "ത്തി" if thousand or rest else word)  # ലക്ഷം -> ലക്ഷത്തി
    if thousand:
        word = _ml_thousands(thousand)
        pieces.append(word[:-1] + "ത്തി" if rest else word)  # ആയിരം -> ആയിരത്തി
    if rest:
        pieces.append(_ml_lt1000(rest))
    return " ".join(pieces)


# ---------------------------------------------------------------- English
def en_int(n: int) -> str:
    from num2words import num2words

    return num2words(n, lang="en_IN").replace(",", "")


def en_year(n: int) -> str:
    from num2words import num2words

    return num2words(n, lang="en", to="year").replace(",", "")


def en_ordinal(n: int) -> str:
    from num2words import num2words

    return num2words(n, lang="en", to="ordinal").replace(",", "")


# ---------------------------------------------------------------- per-language table
LANG = {
    "en": {"int": en_int, "year": en_year, "point": "point", "percent": "percent", "minus": "minus",
           "rupees": "rupees", "paise": "paise", "dollars": "dollars", "cents": "cents"},
    "hi": {"int": hi_int, "year": hi_year, "point": "दशमलव", "percent": "प्रतिशत", "minus": "माइनस",
           "rupees": "रुपये", "paise": "पैसे", "dollars": "डॉलर", "cents": "सेंट"},
    "ta": {"int": ta_int, "year": ta_int, "point": "புள்ளி", "percent": "சதவீதம்", "minus": "மைனஸ்",
           "rupees": "ரூபாய்", "paise": "பைசா", "dollars": "டாலர்", "cents": "சென்ட்"},
    "te": {"int": te_int, "year": te_int, "point": "పాయింట్", "percent": "శాతం", "minus": "మైనస్",
           "rupees": "రూపాయలు", "paise": "పైసలు", "dollars": "డాలర్లు", "cents": "సెంట్లు"},
    "ml": {"int": ml_int, "year": ml_int, "point": "പോയിന്റ്", "percent": "ശതമാനം", "minus": "മൈനസ്",
           "rupees": "രൂപ", "paise": "പൈസ", "dollars": "ഡോളർ", "cents": "സെന്റ്"},
}

# native digits -> ASCII (Devanagari, Tamil, Telugu, Malayalam)
_NATIVE_DIGITS = {}
for base in (0x0966, 0x0BE6, 0x0C66, 0x0D66):
    for d in range(10):
        _NATIVE_DIGITS[chr(base + d)] = str(d)
_NATIVE_RE = re.compile("[" + "".join(_NATIVE_DIGITS) + "]")

_CURRENCY = r"(?P<cur>₹|Rs\.?|INR|रु\.?|रू\.?|\$)"
_NUM = r"(?P<num>\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(?P<frac>\d+))?"
PATTERNS = [
    ("cents", re.compile(_CURRENCY + r"\s?\.(?P<cents>\d{1,2})(?!\d)")),   # $.60 -> sixty cents
    ("currency", re.compile(_CURRENCY + r"\s?" + _NUM)),
    ("percent", re.compile(_NUM + r"\s?%")),
    ("time", re.compile(r"(?<![\d:])(?P<h>\d{1,2}):(?P<m>\d{2})(?![\d:])")),
    ("ordinal", re.compile(r"(?P<num>\d+)(?P<suf>st|nd|rd|th)\b", re.IGNORECASE)),
    ("number", re.compile(r"(?P<neg>(?<![\w-])-)?" + _NUM)),
]


def _digits(s: str, words: Callable[[int], str]) -> str:
    return " ".join(words(int(d)) for d in s)


def _read_number(num: str, frac: str | None, lang: str, *, allow_year: bool = True) -> str:
    t = LANG[lang]
    plain = num.replace(",", "")
    grouped = "," in num
    if not grouped and frac is None and (len(plain) > 9 or (len(plain) > 1 and plain.startswith("0"))):
        return _digits(plain, t["int"])  # phone numbers, codes, IDs
    n = int(plain)
    if allow_year and not grouped and frac is None and len(plain) == 4 and 1100 <= n <= 2099:
        out = t["year"](n)
    else:
        out = t["int"](n)
    if frac is not None:
        out += f" {t['point']} " + _digits(frac, t["int"])
    return out


def expand_numbers(text: str, lang: str) -> tuple[str, bool]:
    """Return (text with numbers spelled out, whether anything was expanded)."""
    if lang not in LANG:
        return text, False
    text = _NATIVE_RE.sub(lambda m: _NATIVE_DIGITS[m.group(0)], text)
    if not any(ch.isdigit() for ch in text):
        return text, False
    t = LANG[lang]

    noun_form = te_attributive if lang == "te" else (lambda w: w)  # number followed by a noun

    def currency(m: re.Match) -> str:
        cur = m.group("cur")
        dollars = cur == "$"
        major = noun_form(_read_number(m.group("num"), None, lang, allow_year=False))
        unit = t["dollars"] if dollars else t["rupees"]
        if lang == "en" and m.group("num") == "1":
            unit = unit[:-1]  # one dollar / one rupee
        out = f"{major} {unit}"
        frac = m.group("frac")
        if frac and int(frac):
            minor = t["int"](int((frac + "0")[:2]))
            out += f" {minor} {t['cents'] if dollars else t['paise']}"
        return out

    def cents(m: re.Match) -> str:
        minor = t["int"](int((m.group("cents") + "0")[:2]))
        return f"{minor} {t['cents'] if m.group('cur') == '$' else t['paise']}"

    def percent(m: re.Match) -> str:
        return f"{noun_form(_read_number(m.group('num'), m.group('frac'), lang, allow_year=False))} {t['percent']}"

    def time(m: re.Match) -> str:
        h, mm = int(m.group("h")), int(m.group("m"))
        return t["int"](h) if mm == 0 and lang == "en" else f"{t['int'](h)} {t['int'](mm)}"

    def ordinal(m: re.Match) -> str:
        n = int(m.group("num"))
        return en_ordinal(n) if lang == "en" else t["int"](n)

    def number(m: re.Match) -> str:
        out = _read_number(m.group("num"), m.group("frac"), lang)
        return f"{t['minus']} {out}" if m.group("neg") else out

    handlers = {"cents": cents, "currency": currency, "percent": percent, "time": time, "ordinal": ordinal,
                "number": number}
    for name, rx in PATTERNS:
        text = rx.sub(lambda m, h=handlers[name]: f" {h(m)} ", text)
    return re.sub(r"\s+", " ", text).strip(), True
