from datetime import datetime

import pytest

from app.data_loader import Caregiver, load_cases, load_taxonomy
from app.modules import matching

NOW = datetime(2025, 1, 15, 9, 0)
LOC = {"lat": 28.54, "lng": 77.27}


def cg(id, role="medic", lat=28.54, lng=77.27, skills=("vitals_monitoring",), langs=("hindi",), rating=4.5, radius=10, active=True, frm=(9, 0), to=(17, 0)):
    from datetime import time
    return Caregiver(id, id, role, list(skills), lat, lng, list(langs), rating, max_radius_km=radius, active=active,
                     free_from=time(*frm), free_until=time(*to))


def patch(monkeypatch, cgs):
    monkeypatch.setattr(matching, "load_caregivers", lambda: (cgs, None))


def ids(ms):
    return [m.caregiver.id for m in ms]


def test_role_map_from_taxonomy():
    rm = matching.role_map()
    assert rm["paramedic_home_visit"] == {"medic"} and rm["home_nursing"] == {"medic"}
    assert rm["lab_test_home"] == {"phlebotomist"} and rm["teleconsultation"] == {"doctor"} and rm["emergency_112"] == set()


def test_filters_role_active_skill_radius(monkeypatch):
    patch(monkeypatch, [cg("ok"), cg("inactive", active=False), cg("doc", role="doctor"), cg("far", lat=29.0), cg("noskill", skills=("injection_im",)),
                        cg("tight", lat=28.58, radius=2)])
    ms = matching.match("paramedic_home_visit", "same_day", "fever", "hinglish", LOC, text="bukhar hai")
    assert ids(ms) == ["ok"]
    assert matching.match("emergency_112", "emergency", "fever", "hinglish", LOC) == []


def test_procedure_picks_required_skill(monkeypatch):
    patch(monkeypatch, [cg("inj", skills=("injection_im",)), cg("dress", skills=("wound_dressing",))])
    assert ids(matching.match("home_nursing", "same_day", "x", "en", LOC, text="Need an IM injection given at home")) == ["inj"]
    assert ids(matching.match("home_nursing", "same_day", "x", "en", LOC, text="पत्नी की पट्टी बदलवानी है")) == ["dress"]
    assert ids(matching.match("home_nursing", "same_day", "x", "en", LOC, text="wound dressing after surgery")) == ["dress"]


def test_teleconsult_ignores_distance(monkeypatch):
    patch(monkeypatch, [cg("far_doc", role="doctor", lat=29.5, skills=("teleconsultation",), radius=5)])
    ms = matching.match("teleconsultation", "same_day", "x", "en", LOC)
    assert ids(ms) == ["far_doc"] and "km away" not in ms[0].reason and "★" in ms[0].reason


def test_closer_equal_caregiver_ranks_higher(monkeypatch):
    patch(monkeypatch, [cg("far", lat=28.60), cg("near", lat=28.545), cg("mid", lat=28.57)])
    assert ids(matching.match("paramedic_home_visit", "same_day", "x", "hinglish", LOC)) == ["near", "mid", "far"]


def test_score_formula_and_cap(monkeypatch):
    patch(monkeypatch, [cg("a", rating=5.0), cg("b", rating=3.8)])
    a, b = matching.match("paramedic_home_visit", "same_day", "x", "hinglish", LOC)
    assert a.score == 0.99 and a.caregiver.id == "a"                       # 0.35 + 0.30 + 0.20 + 0.15 = 1.0, capped at 0.99
    assert b.score == 0.85                                                  # worst rating: 0.35 + 0.30 + 0.20 + 0


def test_tie_breaks_by_id(monkeypatch):
    patch(monkeypatch, [cg("CG_030"), cg("CG_021"), cg("CG_012")])
    assert ids(matching.match("paramedic_home_visit", "same_day", "x", "hinglish", LOC)) == ["CG_012", "CG_021", "CG_030"]


def test_no_widening_beyond_service_radius(monkeypatch):
    patch(monkeypatch, [cg("far1", lat=28.9), cg("far2", lat=28.95)])
    assert matching.match("paramedic_home_visit", "same_day", "x", "hinglish", LOC) == []


def test_reason_strings(monkeypatch):
    patch(monkeypatch, [cg("a", langs=("hindi", "english"))])
    en = matching.match("paramedic_home_visit", "same_day", "x", "en", LOC)[0].reason
    hi = matching.match("paramedic_home_visit", "same_day", "x", "hinglish", LOC)[0].reason
    assert en.startswith("English-speaking") and hi.startswith("Hindi-speaking")
    assert "free 09:00–17:00" in en and len(en) <= 60


def test_next_slot(monkeypatch):
    c = cg("a", frm=(13, 0), to=(16, 0))
    assert matching.next_slot(c, datetime(2025, 1, 15, 9, 0)) == datetime(2025, 1, 15, 13, 0)
    assert matching.next_slot(c, datetime(2025, 1, 15, 14, 10)) == datetime(2025, 1, 15, 14, 30)
    assert matching.next_slot(c, datetime(2025, 1, 15, 17, 0)) == datetime(2025, 1, 16, 13, 0)


def test_reproduces_every_organiser_reference_list():
    """Verified against the organisers' reference matcher output: all non-emergency, non-follow-up rows (including empty lists)."""
    em = load_taxonomy().emergency_label
    checked = bad = 0
    for c in load_cases():
        if c.care_type in (None, em):
            continue
        lang = "en" if c.extra["detected_language"] == "English" else "hinglish"
        got = ids(matching.match(c.care_type, c.urgency, "general", lang, c.location, text=c.text))
        ref = c.reference_matches
        checked += 1
        bad += got[:len(ref)] != ref or (not ref and bool(got))
    assert checked >= 240 and bad == 0, (checked, bad)
