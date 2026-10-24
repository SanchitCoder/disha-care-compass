from typing import Optional

from fastapi import APIRouter, Query

from .. import db, pipeline
from ..modules import safety_net
from ..schemas import SessionTriageRequest, SessionTriageResponse, TriageRequest, TriageResponse

router = APIRouter()


@router.post("/triage", response_model=TriageResponse, tags=["contract"], summary="Route a request (scoring contract)")
def triage(req: TriageRequest, force_decision: bool = Query(False, description="Never ask a follow-up; always decide (batch prediction).")):
    """Exact contract endpoint. Works with no database and no Supabase."""
    return pipeline.run(req, force_decision=force_decision).contract


def _persist(req: SessionTriageRequest, out: pipeline.PipelineOut, history: list[dict]) -> str:
    """Store/update the case (+ patient profile) and auto-schedule the Safety Net. Returns patient_id."""
    app, tc = out.app, out.contract
    patient_id = req.patient_id
    sl = out.slots
    if sl.patient.relation != "unknown" or sl.patient.age_band != "unknown":
        fields = {"relation_to_caller": sl.patient.relation, "age_band": sl.patient.age_band, "chronic_conditions": sl.chronic_conditions}
        if patient_id and db.get("patients", patient_id):
            db.update("patients", patient_id, {k: v for k, v in fields.items() if v not in ("unknown", [])})
        else:
            patient_id = db.insert("patients", {"family_id": req.family_id, **fields, "notes": None})["id"]
    turns = history + [{"role": "user", "content": req.user_input}]
    if tc.follow_up_question:
        turns.append({"role": "assistant", "content": tc.follow_up_question})
    decided = tc.care_type is not None
    status = "awaiting_review" if app.needs_nurse_review else ("open" if decided else "open")
    row = {"patient_id": patient_id, "family_id": req.family_id, "status": status, "care_type": tc.care_type, "urgency": tc.urgency,
           "red_flags": tc.red_flags, "reasoning": tc.reasoning, "confidence": app.confidence, "needs_nurse_review": app.needs_nurse_review,
           "history": turns, "slots": {"slots": out.slots.model_dump(), "cluster": out.cluster, "language": app.language},
           "location": req.location.model_dump() if req.location else None, "triage": tc.model_dump(), "app": app.model_dump()}
    if db.get("cases", app.case_id):
        db.update("cases", app.case_id, row)
    else:
        db.insert("cases", {"id": app.case_id, **row})
    if decided and not out.is_emergency and not app.needs_nurse_review:
        safety_net.schedule(app.case_id, app.language)
    app.status, app.patient_id, app.family_id = status, patient_id, req.family_id
    return patient_id


@router.post("/v1/session/triage", response_model=SessionTriageResponse, tags=["app"], summary="Triage + app metadata (used by the React app)")
def session_triage(req: SessionTriageRequest, force_decision: bool = Query(False)):
    profile = db.get("patients", req.patient_id) if req.patient_id else None
    existing = db.get("cases", req.case_id) if req.case_id else None
    history = [t.model_dump() for t in req.conversation_history]
    served = {b["caregiver_id"] for b in db.select("bookings")
              if (db.get("cases", b["case_id"]) or {}).get("patient_id") == req.patient_id} if req.patient_id else None
    out = pipeline.run(req, force_decision=force_decision, app_mode=True, profile=profile, case_id=req.case_id, served_ids=served)
    _persist(req, out, history)
    return SessionTriageResponse(triage=out.contract, app=out.app)
