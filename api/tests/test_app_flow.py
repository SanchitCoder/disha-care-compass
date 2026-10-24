"""Demo flow end to end (in-memory store): fever -> question -> plan -> book -> Safety Net -> escalation."""
from fastapi.testclient import TestClient

from app.data_loader import load_taxonomy
from app.main import app

c = TestClient(app)
TAX = load_taxonomy()
LOC = {"lat": 28.54, "lng": 77.27}


def triage(text, history=None, case_id=None, **kw):
    r = c.post("/v1/session/triage", json={"user_input": text, "location": LOC, "conversation_history": history or [], "case_id": case_id, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def decided_case():
    first = "papa ko do din se bukhar hai aur khansi bhi"
    r = triage(first)
    cid = r["app"]["case_id"]
    if r["triage"]["follow_up_question"]:
        assert r["app"]["follow_up_options"]
        hist = [{"role": "user", "content": first}, {"role": "assistant", "content": r["triage"]["follow_up_question"]}]
        r = triage("Nahi", hist, cid)
    assert r["triage"]["care_type"] and r["app"]["watch_list"] and r["app"]["caregivers"]
    return r


def test_full_demo_flow():
    r = decided_case()
    cid = r["app"]["case_id"]
    if r["app"]["needs_nurse_review"]:
        return
    top = r["app"]["caregivers"][0]
    b = c.post("/v1/book", json={"case_id": cid, "caregiver_id": top["caregiver_id"], "slot": top["slot"]})
    assert b.status_code == 200, b.text
    brief = b.json()["visit_brief"]
    assert len(brief["summary_lines"]) == 5 and brief["bring_list"]
    q = c.post(f"/v1/safety-net/{cid}/simulate", params={"hours": 6}).json()
    assert q["question"] and len(q["options"]) == 3
    rep = c.post(f"/v1/safety-net/{cid}/reply", json={"text": "saans phool rahi hai"}).json()
    assert rep["escalated"] is True
    assert rep["triage"]["care_type"] == TAX.emergency_label and "112" in rep["triage"]["reasoning"]
    assert c.get(f"/v1/cases/{cid}").json()["status"] == "escalated"


def test_better_closes_checkins():
    r = decided_case()
    cid = r["app"]["case_id"]
    c.post(f"/v1/safety-net/{cid}/schedule")
    rep = c.post(f"/v1/safety-net/{cid}/reply", json={"text": "Behtar"}).json()
    assert rep["closed"] and not rep["escalated"]


def test_worse_chip_escalates():
    r = decided_case()
    cid = r["app"]["case_id"]
    c.post(f"/v1/safety-net/{cid}/schedule")
    rep = c.post(f"/v1/safety-net/{cid}/reply", json={"text": "Zyada kharab"}).json()
    assert rep["escalated"] is True


def test_emergency_not_bookable_and_no_matches():
    r = triage("papa behosh ho gaye")
    assert r["app"]["is_emergency"] and r["app"]["emergency_steps"] and r["triage"]["matched_caregivers"] == []
    b = c.post("/v1/book", json={"case_id": r["app"]["case_id"], "caregiver_id": "CG_001", "slot": "2025-01-15T10:00:00"})
    assert b.status_code == 400


def test_nurse_queue_approve():
    from app import db
    r = triage("kuch samajh nahi aa raha kya karun, thoda ajeeb sa lag raha hai")
    cid = r["app"]["case_id"]
    db.update("cases", cid, {"status": "awaiting_review", "needs_nurse_review": True})   # force the review path
    q = c.get("/v1/nurse/queue").json()
    assert any(x["case_id"] == cid for x in q)
    if r["triage"]["care_type"]:
        a = c.post(f"/v1/nurse/{cid}/approve", json={"care_type": TAX.care_types[0], "urgency": TAX.urgency_levels[1]})
        assert a.status_code == 200, a.text
        assert c.get(f"/v1/cases/{cid}").json()["status"] == "open"


def test_patient_profile_reused():
    r = triage("papa ko bukhar hai", force_decision="true") if False else triage("meri mummy 65 saal ki hain, unhe bukhar hai")
    pid = r["app"]["patient_id"]
    assert pid
    r2 = triage("mummy ko bukhar hai", patient_id=pid)
    assert r2["app"]["patient_summary"]["age_band"] == "60+"
