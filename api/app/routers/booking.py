from fastapi import APIRouter, HTTPException

from .. import db
from ..data_loader import load_caregivers, load_taxonomy
from ..modules import explain, safety_net
from ..modules.intake import Slots
from ..schemas import BookRequest, BookResponse, VisitBrief

router = APIRouter()


def caregiver_dict(cid: str) -> dict:
    for c in load_caregivers()[0]:
        if c.id == cid:
            return {"caregiver_id": c.id, "name": c.name, "role": c.role, "skills": c.skills, "languages": c.languages,
                    "rating": c.rating, "review_count": c.review_count, "gender": c.gender, "lat": c.lat, "lng": c.lng,
                    "max_radius_km": c.max_radius_km, "active": c.active, "tier": c.tier, "qualification": c.qualification,
                    "experience_years": c.experience_years, "base_area": c.base_area,
                    "free_from": c.free_from.strftime("%H:%M") if c.free_from else None,
                    "free_until": c.free_until.strftime("%H:%M") if c.free_until else None}
    raise HTTPException(404, f"caregiver {cid} not found")


@router.get("/v1/caregivers/{caregiver_id}", tags=["app"])
def get_caregiver(caregiver_id: str):
    return caregiver_dict(caregiver_id)


@router.post("/v1/book", response_model=BookResponse, tags=["app"], summary="Book a caregiver and generate the Visit Brief")
def book(req: BookRequest):
    case = db.get("cases", req.case_id)
    if case is None:
        raise HTTPException(404, "case not found")
    if case.get("care_type") in (None, load_taxonomy().emergency_label):
        raise HTTPException(400, "this case has no bookable decision (emergency or still asking questions)")
    cg = caregiver_dict(req.caregiver_id)
    stored = case.get("slots") or {}
    slots = Slots.model_validate(stored.get("slots", {}))
    brief = explain.visit_brief(slots, case["care_type"], stored.get("cluster", "general"), case.get("reasoning") or "")
    b = db.insert("bookings", {"case_id": req.case_id, "caregiver_id": req.caregiver_id, "slot": req.slot, "visit_brief": brief, "status": "confirmed"})
    db.update("cases", req.case_id, {"status": "booked"})
    safety_net.schedule(req.case_id, stored.get("language", "hinglish"))
    return BookResponse(booking_id=b["id"], visit_brief=VisitBrief(**brief), caregiver=cg)
