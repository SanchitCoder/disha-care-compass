"""Safety Net: check-ins at +6 h / +24 h and re-triage of the family's reply."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from .. import db, pipeline
from ..data_loader import load_taxonomy
from ..schemas import ReplyResponse, TriageRequest
from . import explain
from .intake import Slots

BETTER = re.compile(r"\b(behtar|better|theek|thik|improved|accha|achha)\b|बेहतर|ठीक", re.I)
WORSE = re.compile(r"zyada\s*kharab|jyada\s*kharab|worse|bigad|badh\s*(gaya|gayi|raha)|ज़्यादा\s*खराब|बिगड़", re.I)
SAME = re.compile(r"waisa\s*hi|same|no change|vaisa hi|वैसा\s*ही", re.I)


def classify(text: str) -> Optional[str]:
    """Chip answers / short replies -> better | same | worse | None (free text, needs the pipeline)."""
    t = text.strip()
    if len(t) > 40:
        return None
    if WORSE.search(t):
        return "worse"
    if SAME.search(t):
        return "same"
    if BETTER.search(t):
        return "better"
    return None


def schedule(case_id: str, lang: str) -> list[dict]:
    existing = db.select("checkins", case_id=case_id)
    if existing:
        return existing
    q, _ = explain.checkin_question(lang)
    now = datetime.now(timezone.utc)
    return [db.insert("checkins", {"case_id": case_id, "due_at": (now + timedelta(hours=h)).isoformat(), "question": q,
                                   "answered_at": None, "answer_text": None, "outcome": None, "hours": h}) for h in (6, 24)]


def simulate(case_id: str, hours: int) -> Optional[dict]:
    rows = [c for c in db.select("checkins", case_id=case_id) if c.get("answered_at") is None]
    if not rows:
        return None
    pick = min(rows, key=lambda c: abs((c.get("hours") or 6) - hours))
    case = db.get("cases", case_id) or {}
    lang = (case.get("slots") or {}).get("language", "hinglish")
    q, opts = explain.checkin_question(lang)
    db.update("checkins", pick["id"], {"due_at": datetime.now(timezone.utc).isoformat()})
    return {"checkin_id": pick["id"], "hours": pick.get("hours"), "question": q, "options": opts, "language": lang}


def _close_checkins(case_id: str, text: str, outcome: str):
    now = datetime.now(timezone.utc).isoformat()
    for c in db.select("checkins", case_id=case_id):
        if c.get("answered_at") is None:
            db.update("checkins", c["id"], {"answered_at": now, "answer_text": text, "outcome": outcome})


def reply(case_id: str, text: str) -> Optional[ReplyResponse]:
    case = db.get("cases", case_id)
    if case is None:
        return None
    tax = load_taxonomy()
    old_urg = case.get("urgency")
    chip = classify(text)
    history = (case.get("history") or []) + [{"role": "assistant", "content": explain.checkin_question((case.get("slots") or {}).get("language", "hinglish"))[0]}]
    if chip == "better":
        _close_checkins(case_id, text, "better")
        db.update("cases", case_id, {"status": "closed"})
        return ReplyResponse(escalated=False, closed=True, message="Good to hear. Check-ins closed.")
    if chip == "same":
        first = next((c for c in db.select("checkins", case_id=case_id) if c.get("answered_at") is None), None)
        if first:
            db.update("checkins", first["id"], {"answered_at": datetime.now(timezone.utc).isoformat(), "answer_text": text, "outcome": "same"})
        return ReplyResponse(escalated=False, message="Noted. We will check again later.")

    loc = case.get("location")
    stored = case.get("slots") or {}
    req = TriageRequest(user_input=text, language="auto", location=loc, conversation_history=[t for t in history if t["role"] in ("user", "assistant")])
    out = pipeline.run(req, force_decision=True, app_mode=True, case_id=case_id)
    new_urg = out.contract.urgency
    rank = tax.urgency_rank
    more_severe = out.is_emergency or rank.get(new_urg, 0) > rank.get(old_urg, 0)
    if chip == "worse" and not more_severe:
        slots = Slots.model_validate(stored.get("slots", {})) if stored.get("slots") else out.slots
        lv = tax.urgency_levels
        bumped = lv[min(lv.index(old_urg or lv[0]) + 1, len(lv) - 1)]
        out = pipeline.rebuild_decision(slots, stored.get("cluster", out.cluster), out.result, case.get("care_type") or tax.care_types[0],
                                        bumped, stored.get("language", out.app.language), loc, case_id)
        more_severe = True
    _close_checkins(case_id, text, "escalated" if more_severe else "stable")
    patch = {"history": history + [{"role": "user", "content": text}]}
    if more_severe:
        patch.update(status="escalated", care_type=out.contract.care_type, urgency=out.contract.urgency, red_flags=out.contract.red_flags,
                     reasoning=out.contract.reasoning, triage=out.contract.model_dump(), app=out.app.model_dump())
        db.insert("notifications", {"case_id": case_id, "family_member_id": None, "channel": "sms (simulated)", "sent_at": datetime.now(timezone.utc).isoformat(),
                                    "message": f"Disha: the patient's condition looks worse. New advice: {out.contract.care_type} / {out.contract.urgency}."})
    db.update("cases", case_id, patch)
    out.app.status = "escalated" if more_severe else "open"
    return ReplyResponse(escalated=more_severe, message="Escalated" if more_severe else "No change in advice", triage=out.contract, app=out.app)
