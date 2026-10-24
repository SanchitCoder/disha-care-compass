from fastapi import APIRouter, HTTPException

from .. import db, pipeline
from ..modules import safety_net
from ..modules.intake import Slots
from ..schemas import NurseApproveRequest, SessionTriageResponse, TriageResponse
from ..data_loader import load_taxonomy

router = APIRouter(tags=["nurse"])


@router.get("/v1/cases/{case_id}", summary="Current state of a case (used for polling by the user screen)")
def get_case(case_id: str):
    c = db.get("cases", case_id)
    if c is None:
        raise HTTPException(404, "case not found")
    return {"case_id": case_id, "status": c["status"], "triage": c.get("triage"), "app": c.get("app")}


@router.get("/v1/nurse/queue", summary="Cases awaiting nurse review")
def queue():
    rows = [c for c in db.select("cases", status="awaiting_review")]
    return [{"case_id": c["id"], "created_at": c.get("created_at"), "care_type": c.get("care_type"), "urgency": c.get("urgency"),
             "confidence": c.get("confidence"), "reasoning": c.get("reasoning"),
             "key_factors": (c.get("app") or {}).get("key_factors", []), "history": c.get("history") or []} for c in rows]


@router.post("/v1/nurse/{case_id}/approve", response_model=SessionTriageResponse, summary="Approve (optionally with edits) a low-confidence case")
def approve(case_id: str, body: NurseApproveRequest):
    c = db.get("cases", case_id)
    if c is None:
        raise HTTPException(404, "case not found")
    tax = load_taxonomy()
    care = body.care_type or c["care_type"]
    urg = body.urgency or c["urgency"]
    stored = c.get("slots") or {}
    slots = Slots.model_validate(stored.get("slots", {}))
    edited = (care, urg) != (c["care_type"], c["urgency"])
    out = pipeline.rebuild_decision(slots, stored.get("cluster", "general"), None, care, urg, stored.get("language", "hinglish"), c.get("location"), case_id, confidence=1.0)
    if not edited and c.get("triage"):          # approved as-is: keep the AI's own text + matches
        out.contract = TriageResponse.model_validate(c["triage"])
    db.insert("nurse_reviews", {"case_id": case_id, "reviewer_name": body.reviewer_name, "decision": "edited" if edited else "approved",
                                "edited_care_type": body.care_type, "edited_urgency": body.urgency, "note": body.note})
    out.app.status = "open"
    db.update("cases", case_id, {"status": "open", "care_type": care, "urgency": urg, "needs_nurse_review": False, "red_flags": out.contract.red_flags,
                                 "reasoning": out.contract.reasoning, "triage": out.contract.model_dump(), "app": out.app.model_dump()})
    if care != tax.emergency_label:
        safety_net.schedule(case_id, out.app.language)
    return SessionTriageResponse(triage=out.contract, app=out.app)


@router.get("/v1/taxonomy", summary="Care types / urgency levels with display names (for the UI)")
def taxonomy():
    from ..modules.explain import CARE_NAMES, WHEN
    tax = load_taxonomy()
    return {"care_types": [{"id": c, "names": CARE_NAMES.get(c, {"en": c.replace("_", " "), "hinglish": c.replace("_", " "), "hi": c.replace("_", " ")})} for c in tax.care_types],
            "urgency_levels": [{"id": u, "names": WHEN.get(u, {"en": u})} for u in tax.urgency_levels],
            "emergency_label": tax.emergency_label}
