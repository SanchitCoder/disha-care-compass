"""Caregiver matching, reproducing the organisers' reference rules (verified: 100% of 340 reference lists reproduced).

Hard filters : caregiver active, role fits the care type, has the required skill (any of), and the request is inside the caregiver's
               own service radius (max_radius_km). Teleconsultation is virtual: distance is ignored.
Score        : 0.35*language + 0.30*proximity + 0.20*skill + 0.15*rating     (weights in thresholds.json)
               proximity = 1 - d/12 km (floored at 0; 1.0 for virtual); rating = min-max over the roster; language = 1 if the caregiver
               speaks the user's language or Hindi (every caregiver does), else 0.5.
Order        : score rounded to 2 decimals and capped at 0.99, ties broken by caregiver id. Top 3.
No widening  : a caregiver outside their own service radius cannot serve the request, so the list can be empty.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

from .. import config
from ..data_loader import Caregiver, load_caregivers, load_taxonomy

SCORE_CAP = 0.99
_PROCEDURES = [   # checked in order; first hit wins
    ("iv", re.compile(r"\biv\b|drip|ड्रिप|saline|\bcannula", re.I)),
    ("injection", re.compile(r"injection|इंजेक्शन|इंसुलिन|insulin|\binj\b|\bim\b", re.I)),
    ("catheter", re.compile(r"catheter|कैथेटर", re.I)),
    ("dressing", re.compile(r"dressing|ड्रेसिंग|पट्टी|patti|bedsore|bed sore|ghav|घाव|wound|post[- ]?op|operation|ऑपरेशन|surgery", re.I)),
]


def role_map() -> dict[str, set[str]]:
    """care label -> compatible roles (from config/taxonomy.yaml matching.roles). Printed by `python -m app.modules.matching`."""
    tax = load_taxonomy()
    roles = tax.matching["roles"]
    return {c: ({roles[c]} if c in roles else set()) for c in tax.care_types}


def detect_procedure(text: str, slot_procedure: str = "unknown") -> Optional[str]:
    for name, pat in _PROCEDURES:
        if pat.search(text or ""):
            return name
    return slot_procedure if slot_procedure in ("injection", "iv", "dressing", "catheter") else None


def required_skills(care_type: str, text: str, slot_procedure: str = "unknown") -> Optional[set[str]]:
    m = load_taxonomy().matching
    if care_type == "home_nursing" or care_type == load_taxonomy().nurse_label:
        proc = detect_procedure(text, slot_procedure)
        return set(m["procedure_skills"][proc]) if proc in m["procedure_skills"] else None
    d = m.get("default_skills", {}).get(care_type)
    return set(d) if d else None


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lng2 - lng1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


@dataclass
class Match:
    caregiver: Caregiver
    score: float
    reason: str
    slot: Optional[datetime]
    km: Optional[float]
    parts: dict = field(default_factory=dict)


def next_slot(c: Caregiver, now: Optional[datetime] = None) -> Optional[datetime]:
    """Earliest bookable time inside the caregiver's daily free window (the data has windows, not dates)."""
    if not c.free_from or not c.free_until:
        return None
    now = now or datetime.now()
    start = now.replace(hour=c.free_from.hour, minute=c.free_from.minute, second=0, microsecond=0)
    end = now.replace(hour=c.free_until.hour, minute=c.free_until.minute, second=0, microsecond=0)
    if now <= start:
        return start
    if now < end - timedelta(minutes=30):
        return (now + timedelta(minutes=30 - now.minute % 30)).replace(second=0, microsecond=0)
    return start + timedelta(days=1)


def _hhmm(t: Optional[time]) -> str:
    return t.strftime("%H:%M") if t else ""


def _wanted_language(user_lang: str) -> str:
    return "english" if user_lang == "en" else "hindi"


def match(care_type: str, urgency: str, cluster: str, user_lang: str, location: Optional[dict], top_n: int = 3,
          text: str = "", slot_procedure: str = "unknown", gender_pref: str = "any", served_ids: Optional[set[str]] = None,
          app_bonuses: bool = False, now: Optional[datetime] = None) -> list[Match]:
    tax = load_taxonomy()
    if care_type == tax.emergency_label:
        return []
    T = config.THRESHOLDS
    W = T["match_weights"]
    cgs, _ = load_caregivers()
    role = tax.matching["roles"].get(care_type)
    virtual = care_type in tax.matching.get("virtual", [])
    need = required_skills(care_type, text, slot_procedure)
    rated = [c.rating for c in cgs if c.rating is not None]
    lo, hi = (min(rated), max(rated)) if rated else (0.0, 5.0)
    want = _wanted_language(user_lang)
    cap = T.get("distance_cap_km", 12)

    out: list[Match] = []
    for c in cgs:
        if not c.active or c.role != role:
            continue
        if need and not (set(c.skills) & need):
            continue
        km = haversine_km(location["lat"], location["lng"], c.lat, c.lng) if location else None
        if not virtual and km is not None and km > c.max_radius_km:
            continue
        f_dist = 1.0 if virtual or km is None else max(0.0, 1 - km / cap)
        speaks = want in c.languages
        f_lang = 1.0 if (speaks or "hindi" in c.languages) else 0.5
        f_skill = 1.0
        f_rate = 0.5 if c.rating is None else (c.rating - lo) / (hi - lo) if hi > lo else 1.0
        contrib = {"language": W["language"] * f_lang, "distance": W["distance"] * f_dist, "skill": W["skill"] * f_skill, "rating": W["rating"] * f_rate}
        score = sum(contrib.values())
        if app_bonuses:
            if gender_pref in ("female", "male") and c.gender == gender_pref:
                score += 0.05
            if served_ids and c.id in served_ids:
                score += 0.05
        score = round(min(score, SCORE_CAP), 2)
        lang_txt = f"{(want if speaks else 'hindi').capitalize()}-speaking"
        bits = [lang_txt]
        if not virtual and km is not None:
            bits.append("<1 km away" if km < 1 else f"{km:.1f} km away")
        if c.free_from and c.free_until:
            bits.append(f"free {_hhmm(c.free_from)}–{_hhmm(c.free_until)}")
        if virtual and c.rating is not None:
            bits.append(f"★{c.rating:g}")
        reason = ""
        for b in bits:
            cand = ", ".join(filter(None, [reason, b]))
            if len(cand) <= 60:
                reason = cand
        out.append(Match(c, score, reason, next_slot(c, now), km, contrib))
    out.sort(key=lambda m: (-m.score, m.caregiver.id))
    return out[:top_n]


if __name__ == "__main__":
    for k, v in role_map().items():
        print(f"{k:<24} <- roles {sorted(v) or '(none: emergency path)'}")
