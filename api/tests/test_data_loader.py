from app.data_loader import FOLLOW_UP, label_distribution, load_caregivers, load_cases, load_split, load_taxonomy


def test_caregivers_parse():
    cgs, now = load_caregivers()
    assert len(cgs) == 60 and now is None
    c = {x.id: x for x in cgs}["CG_002"]
    assert c.role == "medic" and "injection_im" in c.skills and c.max_radius_km == 5 and c.active and c.gender == "male"
    assert "english" in c.languages and c.free_from.hour == 9 and c.free_until.hour == 15
    assert {x.role for x in cgs} == {"medic", "doctor", "phlebotomist"} and sum(not x.active for x in cgs) == 8


def test_cases_and_follow_up_label():
    cases = load_cases()
    assert len(cases) == 340
    assert sum(c.care_label == FOLLOW_UP for c in cases) == 39
    assert all((c.urgency is None) == (c.care_type is None) for c in cases)


def test_splits_are_stratified_and_disjoint():
    tr, va = load_split("train"), load_split("val")
    assert len(tr) + len(va) == 340 and not ({c.id for c in tr} & {c.id for c in va})
    assert {k for k in label_distribution(va)} == {k for k in label_distribution(tr)}


def test_unseen_split_holds_out_whole_templates():
    tr, va = load_split("train_unseen"), load_split("val_unseen")
    assert not ({c.text for c in tr} & {c.text for c in va}) and len(va) > 20
