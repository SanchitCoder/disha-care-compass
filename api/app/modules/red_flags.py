"""Deterministic red-flag gate. NO LLM. Runs first on every turn.

Pattern source: config/red_flags.yaml + the taxonomy's red_flag_examples (exact phrases).
Negation: a negator within 4 tokens (same clause, outside the match) suppresses a hit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import yaml

from .. import config
from ..data_loader import load_taxonomy
from .normalise import Normalised, normalise

CLAUSE_STOPS = {"|", "aur", "lekin", "par", "magar", "but", "and", "however", "though", "जबकि", "लेकिन", "और", "पर"}
NEG_TOKENS = {"nahi", "nahin", "nhi", "nahee", "na", "no", "not", "never", "without", "nai", "नहीं", "नही", "ना", "न", "bina", "बिना"}
NEG_PHRASES = re.compile(r"(theek|thik|normal|sahi|fine|ok|okay|ठीक|सामान्य)\s*(hai|hain|he|है|हैं)|koi\s*(dikkat|problem|takleef)|कोई\s*(दिक्कत|समस्या|तकलीफ)")
WINDOW = 4        # tokens before a match
WINDOW_AFTER = 2  # tokens after (stricter: 'behosh hain, bukhar nahi' must still fire)

_INFANT_AGE = re.compile(r"(\d+)\s*(din|dino|day|days|hafte|hafta|week|weeks|mahine|mahina|mahine|month|months|महीने|महीना|दिन)\s*(ka|ki|ke|old|के|का|की)?\s*(bachcha|bacha|bachchi|baby|beta|beti|shishu|बच्चा|बच्ची|बेटा|बेटी)")
_INFANT_NEW = re.compile(r"newborn|naya\s*bachcha|nav\s*jaat|नवजात|नया\s*बच्चा")
_INFANT_SYMPTOM = re.compile(r"bukhar|fever|tez\s*garam|doodh\s*(nahi|nhi)|dudh\s*(nahi|nhi)|बुखार|दूध\s*नहीं")


@dataclass
class Flag:
    id: str
    label_en: str
    label_hi: str
    any_patterns: list[re.Pattern]
    all_groups: list[list[re.Pattern]]
    special: str | None = None


@lru_cache(maxsize=1)
def load_flags() -> list[Flag]:
    raw = yaml.safe_load((config.CONFIG_DIR / "red_flags.yaml").read_text(encoding="utf-8"))
    comp = lambda p: re.compile(p, re.I | re.U)
    flags = [Flag(f["id"], f["label_en"], f.get("label_hi", f["label_en"]),
                  [comp(p) for p in f.get("patterns", [])],
                  [[comp(p) for p in grp] for grp in f.get("requires_all", [])],
                  f.get("special")) for f in raw["flags"]]
    # taxonomy red-flag examples -> exact-phrase flags
    for i, ex in enumerate(load_taxonomy().red_flag_examples):
        phrase = re.escape(ex.lower().strip())
        flags.append(Flag(f"taxonomy_{i}", ex, ex, [comp(phrase)], []))
    return flags


def _negated(text: str, start: int, end: int) -> bool:
    before = text[:start].split()[-WINDOW:]
    after = text[end:].split()[:WINDOW_AFTER]
    kept_before = []
    for t in reversed(before):
        if t in CLAUSE_STOPS:
            break
        kept_before.append(t)
    kept_after = []
    for t in after:
        if t in CLAUSE_STOPS:
            break
        kept_after.append(t)
    window = kept_before + kept_after
    return bool(set(window) & NEG_TOKENS) or bool(NEG_PHRASES.search(" ".join(reversed(kept_before)) + " | " + " ".join(kept_after)))


def _first_valid(patterns: list[re.Pattern], text: str) -> bool:
    for p in patterns:
        for m in p.finditer(text):
            if not _negated(text, m.start(), m.end()):
                return True
    return False


def _infant_under_3_months(text: str) -> bool:
    if _INFANT_NEW.search(text) and _INFANT_SYMPTOM.search(text):
        return True
    for m in _INFANT_AGE.finditer(text):
        n, unit = int(m.group(1)), m.group(2)
        days = n * (1 if unit in ("din", "dino", "day", "days", "दिन") else 7 if unit in ("hafte", "hafta", "week", "weeks") else 30)
        if days < 90 and _INFANT_SYMPTOM.search(text):
            return True
    return False


def check_text(norm: Normalised) -> list[str]:
    """Flag ids that fire on any variant of the text."""
    hits: list[str] = []
    for f in load_flags():
        for text in norm.variants:
            fired = False
            if f.special == "infant_fever":
                fired = _infant_under_3_months(text)
            else:
                fired = _first_valid(f.any_patterns, text) or (
                    bool(f.all_groups) and all(_first_valid(g, text) for g in f.all_groups))
            if fired:
                hits.append(f.id)
                break
    return hits


def check(text: str, history: list[dict] | None = None) -> tuple[bool, list[str]]:
    flags = check_text(normalise(text, history))
    return bool(flags), flags


def check_slots(probes: dict) -> list[str]:
    """Red flags surfaced by extracted slots (LLM or answered probes)."""
    out = []
    if probes.get("breathing_difficulty") == "yes":
        out.append("breathing")
    if probes.get("chest_pain") == "yes":
        out.append("chest_pain")
    if probes.get("confusion") == "yes":
        out.append("confusion")
    return out


# internal flag id -> the organisers' controlled vocabulary (nearest term where the vocabulary has none; the reasoning text names the real trigger)
VOCAB = {
    "unresponsive": "unconsciousness", "breathing": "breathlessness", "blue_lips": "breathlessness", "choking": "breathlessness",
    "chest_pain": "chest_pain", "chest_pain_combo": "chest_pain", "stroke": "stroke_symptoms", "seizure": "seizure",
    "severe_bleeding": "severe_bleeding", "head_injury": "unconsciousness", "poisoning": "unconsciousness",
    "pregnancy_emergency": "severe_bleeding", "infant_fever": "high_fever_infant", "severe_burn": "severe_bleeding",
    "suicidal": "suicidal_ideation", "severe_dehydration": "severe_dehydration", "confusion": "unconsciousness",
    "model_triage_emergency": "unconsciousness", "nurse_or_followup_escalation": "unconsciousness", "patient_worsening": "breathlessness",
}


def to_vocab(ids: list[str]) -> list[str]:
    """Internal ids -> organiser vocabulary, de-duplicated, order kept."""
    out: list[str] = []
    for i in ids:
        v = VOCAB.get(i, i)
        if v not in out:
            out.append(v)
    return out


def label(flag_id: str, lang: str = "en") -> str:
    for f in load_flags():
        if f.id == flag_id:
            return f.label_hi if lang == "hi" else f.label_en
    return {"confusion": "confusion / altered alertness"}.get(flag_id, flag_id)
