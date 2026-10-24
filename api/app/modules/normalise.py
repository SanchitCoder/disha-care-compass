"""Text normalisation: lowercase, repeated-letter collapse, punctuation, Devanagari romanisation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from indic_transliteration import sanscript

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_CLAUSE_PUNCT = re.compile(r"[,;.!?\n:()\[\]|]+")
_OTHER_PUNCT = re.compile(r"[^\w\s'|ऀ-ॿ]", re.UNICODE)
_HI_WORDS = {"ko", "hai", "hain", "ka", "ki", "ke", "se", "mein", "me", "nahi", "nhi", "bhi", "aur", "ho", "raha", "rahi", "rahe",
             "papa", "mummy", "bukhar", "khansi", "dard", "kal", "din", "ghar", "hua", "gaya", "gayi", "mujhe", "mera", "meri", "thoda", "bahut"}


def _squash(s: str, keep: int) -> str:
    return re.sub(r"(.)\1{2,}", r"\1" * keep, s)


def romanise(text: str) -> str:
    """Devanagari -> simple lower-case Roman (ITRANS, then simplified)."""
    r = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    r = r.replace("M", "n").replace("~N", "n").replace(".N", "n").replace(".h", "")
    r = r.lower().replace("aa", "a").replace("ii", "i").replace("uu", "u")
    return " ".join(re.sub(r"(?<=[^aeiou\s])a\b", "", w) for w in r.split())


def clean_variants(text: str) -> list[str]:
    """Match-ready copies: punctuation -> clause marker '|', repeats collapsed two ways."""
    base = text.lower()
    out = []
    for keep in (1, 2):
        t = _squash(base, keep)
        t = _CLAUSE_PUNCT.sub(" | ", t)
        t = _OTHER_PUNCT.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t not in out:
            out.append(t)
    return out


def detect_language(text: str) -> str:
    """'hi' (Devanagari) | 'hinglish' (Roman Hindi) | 'en'."""
    if DEVANAGARI.search(text):
        return "hi"
    toks = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(toks & _HI_WORDS) >= 1 else "en"


@dataclass
class Normalised:
    current: str                    # current turn, original
    history_user: list[str]         # earlier user turns
    full: str                       # history user turns + current, original
    variants: list[str] = field(default_factory=list)   # match-ready copies (incl. romanised)
    clean: str = ""                 # primary cleaned copy of `full`
    clean_current: str = ""         # primary cleaned copy of the current turn only
    has_devanagari: bool = False
    language: str = "en"            # detected
    key: str = ""                   # cache key text


def normalise(current: str, history: list[dict] | None = None, language: str = "auto") -> Normalised:
    history = history or []
    hist_user = [h["content"] for h in history if h.get("role") == "user"]
    full = " . ".join(hist_user + [current])
    has_dev = bool(DEVANAGARI.search(full))
    variants = clean_variants(full)
    if has_dev:
        variants += [v for v in clean_variants(romanise(full)) if v not in variants]
    lang = language if language not in ("auto", "", None) else detect_language(current if len(current) > 6 else full)
    return Normalised(current=current, history_user=hist_user, full=full, variants=variants, clean=variants[0], clean_current=clean_variants(current)[0],
                      has_devanagari=has_dev, language=lang, key=variants[0])
