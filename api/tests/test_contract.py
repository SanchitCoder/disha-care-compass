import pytest
from fastapi.testclient import TestClient

from app.data_loader import FOLLOW_UP, load_cases, load_multi_turn, load_taxonomy
from app.main import app
from app.schemas import TriageResponse

client = TestClient(app)
TAX = load_taxonomy()
KEYS = {"follow_up_question", "care_type", "urgency", "red_flags", "reasoning", "matched_caregivers"}
LOC = {"lat": 28.54, "lng": 77.27}
SAMPLE = {"user_input": "papa ko do din se bukhar hai aur khansi bhi", "language": "auto", "location": LOC, "conversation_history": []}


def post(body, **params):
    r = client.post("/triage", json=body, params=params)
    assert r.status_code == 200, r.text
    assert set(r.json()) == KEYS          # no extra top-level fields
    return TriageResponse.model_validate(r.json())


def test_taxonomy_matches_organiser_labels():
    assert set(TAX.care_types) == {"paramedic_home_visit", "home_nursing", "lab_test_home", "teleconsultation", "emergency_112"}
    assert TAX.urgency_levels == ["scheduled", "same_day", "emergency"]
    assert TAX.emergency_label == "emergency_112" and TAX.immediate_label == "emergency"


def test_sample_request_decides_like_the_brief():
    """The brief's own example: a decision (no follow-up) with a Medic visit, same day, and matched caregivers."""
    r = post(SAMPLE)
    assert r.follow_up_question is None
    assert (r.care_type, r.urgency) == ("paramedic_home_visit", "same_day")
    assert 1 <= len(r.matched_caregivers) <= 3 and not r.red_flags
    for m in r.matched_caregivers:
        assert 0 <= m.score <= 1 and round(m.score, 2) == m.score and m.reason and len(m.reason) <= 60


def test_force_decision_never_follows_up():
    r = post({"user_input": "I need help", "location": LOC}, force_decision="true")
    assert r.follow_up_question is None and r.care_type in TAX.care_types


@pytest.mark.parametrize("text", ["I need help", "doctor chahiye", "mummy ke liye kuch book karna hai", "मदद चाहिए"])
def test_vague_request_asks_one_follow_up(text):
    r = post({"user_input": text, "location": LOC})
    assert r.follow_up_question and r.care_type is None and r.urgency is None and not r.matched_caregivers


def test_multi_turn_examples_from_organisers():
    for ex in load_multi_turn():
        hist, last = [], None
        for t in ex["turns"]:
            if t["role"] == "user":
                if last is not None:      # a reply to the previous assistant turn
                    pass
                cur, last = t["text"], t
                expected = next((x for x in ex["turns"][ex["turns"].index(t) + 1:] if x["role"] == "assistant"), None)
                r = post({"user_input": cur, "location": LOC, "conversation_history": list(hist)})
                if expected is not None:
                    assert (r.care_type, r.urgency) == (expected.get("care_type"), expected.get("urgency")), (ex["id"], cur, r)
                    assert bool(r.follow_up_question) == bool(expected.get("follow_up_question"))
                hist += [{"role": "user", "content": cur}, {"role": "assistant", "content": r.follow_up_question or r.reasoning or ""}]


def test_emergency_uses_organiser_vocabulary():
    r = post({"user_input": "papa behosh ho gaye, saans nahi le rahe"})
    assert r.care_type == TAX.emergency_label and r.urgency == TAX.immediate_label
    assert set(r.red_flags) == {"unconsciousness", "breathlessness"}
    assert r.matched_caregivers == [] and "112" in r.reasoning and "108" in r.reasoning


def test_emergency_even_when_forced_or_in_history():
    r = post({"user_input": "haan", "conversation_history": [{"role": "user", "content": "mummy ko daura pad raha hai"}]}, force_decision="true")
    assert r.care_type == TAX.emergency_label and r.red_flags == ["seizure"]


def test_negated_red_flag_is_not_emergency():
    r = post({"user_input": "bukhar hai par saans mein koi dikkat nahi", "location": LOC}, force_decision="true")
    assert r.care_type != TAX.emergency_label


def test_organiser_dataset_regression():
    """Pipeline wiring check: every distinct organiser request gets its labelled care type and urgency (offline kNN + rules)."""
    seen, bad = set(), []
    for c in load_cases():
        if c.text in seen:
            continue
        seen.add(c.text)
        r = post({"user_input": c.text, "location": c.location or LOC})
        if (r.care_type, r.urgency) != (c.care_type, c.urgency):
            bad.append((c.text, (c.care_type, c.urgency), (r.care_type, r.urgency)))
    assert not bad, bad


def test_no_clinical_content_in_reasoning():
    from app.modules.explain import is_safe
    assert is_safe(post(SAMPLE).reasoning)


def test_docs_and_health():
    assert client.get("/docs").status_code == 200
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["supabase"] in ("ok", "off", "down")
