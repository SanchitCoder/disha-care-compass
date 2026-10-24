from app import groq_client
from app.modules import explain
from app.modules.intake import Slots, Symptom


def test_is_safe():
    assert not explain.is_safe("give paracetamol 500mg twice")
    assert not explain.is_safe("aapko pneumonia hai")
    assert not explain.is_safe("take one tablet")
    assert explain.is_safe("Fever and cough for 2 days; no breathing difficulty reported.")
    assert explain.is_safe("Risk factors: reported: diabetes", allow=("diabetes",))


def test_unsafe_llm_brief_falls_back_to_template(monkeypatch):
    slots = Slots(symptoms=[Symptom(name_en="fever", duration_days=2)])
    bad = {"summary_lines": ["Give paracetamol 500mg", "b", "c", "d", "e"], "bring_list": ["thermometer"]}
    monkeypatch.setattr(groq_client, "chat_json", lambda *a, **k: bad)
    brief = explain.visit_brief(slots, "paramedic_home_visit", "fever", "x")
    assert "paracetamol" not in " ".join(brief["summary_lines"]).lower()
    assert len(brief["summary_lines"]) == 5 and brief["bring_list"]


def test_unsafe_smoothing_falls_back(monkeypatch):
    from app import config
    from app.modules.triage import TriageResult
    monkeypatch.setitem(config.THRESHOLDS, "smooth_reasoning_with_llm", True)
    monkeypatch.setattr(groq_client, "chat_json", lambda *a, **k: {"text": "Take paracetamol 500mg"})
    res = TriageResult("paramedic_home_visit", "same_day", {}, {}, [], 0.9, "llm")
    assert "paracetamol" not in explain.reasoning(res, Slots(symptoms=[Symptom(name_en="fever")])).lower()


def test_good_llm_brief_is_used(monkeypatch):
    slots = Slots()
    good = {"summary_lines": ["Father, 60+", "Fever 2 days", "No breathing difficulty", "None", "Call son"], "bring_list": ["thermometer", "mystery_item"]}
    monkeypatch.setattr(groq_client, "chat_json", lambda *a, **k: good)
    brief = explain.visit_brief(slots, "paramedic_home_visit", "fever", "x")
    assert brief["summary_lines"][0] == "Father, 60+" and brief["bring_list"] == ["thermometer"]
