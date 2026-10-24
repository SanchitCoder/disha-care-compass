from fastapi import APIRouter, HTTPException, Query

from .. import db
from ..modules import safety_net
from ..schemas import ReplyRequest, ReplyResponse

router = APIRouter(prefix="/v1/safety-net", tags=["safety-net"])


def _case(case_id: str) -> dict:
    c = db.get("cases", case_id)
    if c is None:
        raise HTTPException(404, "case not found")
    return c


@router.post("/{case_id}/schedule", summary="Create check-ins at +6 h and +24 h")
def schedule(case_id: str):
    c = _case(case_id)
    return {"checkins": safety_net.schedule(case_id, (c.get("slots") or {}).get("language", "hinglish"))}


@router.post("/{case_id}/simulate", summary="DEMO ONLY: return the due check-in question now")
def simulate(case_id: str, hours: int = Query(6, ge=1, le=48)):
    _case(case_id)
    safety_net.schedule(case_id, (db.get("cases", case_id).get("slots") or {}).get("language", "hinglish"))
    r = safety_net.simulate(case_id, hours)
    if r is None:
        raise HTTPException(409, "no open check-ins for this case")
    return r


@router.post("/{case_id}/reply", response_model=ReplyResponse, summary="Re-triage the family's reply; escalates if worse")
def reply(case_id: str, body: ReplyRequest):
    _case(case_id)
    return safety_net.reply(case_id, body.text)
