# Sanocare Care-Triage & Caregiver-Matching — Hackathon Dataset (Synthetic)

**For:** IIIT-Delhi hackathon · **Provided by:** Sanocare Tech Innovations Pvt Ltd
**Version:** v1 · 24 Oct 2026

> ⚠️ **This is 100% SYNTHETIC data.** Every message, caregiver, location and label was
> machine-generated for this hackathon. It contains **no real patients, no real staff, and no PII.**
> It is safe to share, modify, and redistribute for the event. It is **not** clinical ground truth
> and must **not** be used to make real medical-triage decisions.

---

## What the task is

Build the "brain" behind Sanocare's WhatsApp assistant: take a free-text health request (often in
Hinglish/Hindi/English), decide **what kind of care** is needed and **how urgent** it is, flag any
**medical red flags**, and **match the best caregivers** nearby — or ask **one follow-up question**
when there isn't enough information yet.

This mirrors the problem statement's API contract exactly:

```
// Request
{ "user_input": "papa ko do din se bukhar hai aur khansi bhi",
  "language": "auto",
  "location": { "lat": 28.54, "lng": 77.27 },
  "conversation_history": [] }

// Response
{ "follow_up_question": null,
  "care_type": "paramedic_home_visit",
  "urgency": "same_day",
  "red_flags": [],
  "reasoning": "Fever with cough for 2 days, no breathing difficulty reported",
  "matched_caregivers": [ { "caregiver_id": "CG_014", "score": 0.92,
                            "reason": "Hindi-speaking, 3 km away, free at 4 pm" } ] }
```

When information is insufficient, the system returns a `follow_up_question` and leaves
`care_type`/`urgency` null.

---

## Files

| File | Rows | What it is |
|---|---|---|
| `triage_dataset.jsonl` | 340 | Labelled examples, one JSON object per line, in the full nested request/response shape above. **Primary training file.** |
| `triage_dataset_flat.csv` | 340 | The same data flattened to one row each (for quick EDA / pandas). |
| `caregivers.csv` | 60 | The caregiver roster to match against (the "supply" side). |
| `multi_turn_examples.jsonl` | 2 | Worked multi-turn conversations showing a follow-up → resolution, and an emergency. |

---

## `triage_dataset.jsonl` schema

Each line:
```
id                 : string, e.g. "S0001"
detected_language  : "English" | "Hindi" | "Hinglish"   (what the input is written in)
request            : { user_input, language:"auto", location:{lat,lng}, conversation_history:[] }
response:
  follow_up_question : string | null   (non-null ONLY when more info is needed)
  care_type          : enum | null
  urgency            : enum | null
  red_flags          : string[]        (controlled vocab; non-empty ⇒ emergency)
  reasoning          : string          (short human-readable justification)
  matched_caregivers : [ { caregiver_id, score (0–1), reason } ]
```

### Enums
- **`care_type`**: `paramedic_home_visit` (a Sanocare *Medic* visits the home — the ops term is "Medic", the contract field keeps "paramedic") · `home_nursing` (dressing, injections, IV, catheter, vitals) · `lab_test_home` (home sample collection) · `teleconsultation` (virtual doctor) · `emergency_112` (red-flag → emergency services) · `null` (follow-up needed).
- **`urgency`**: `emergency` · `same_day` · `scheduled` · `null`.
- **`red_flags`** (controlled vocab): `chest_pain`, `breathlessness`, `unconsciousness`, `severe_bleeding`, `stroke_symptoms`, `seizure`, `high_fever_infant`, `suicidal_ideation`, `severe_dehydration`. **Any red flag ⇒ `care_type=emergency_112`, `urgency=emergency`, and `matched_caregivers=[]`** (route to 112, never home care).

Class balance: `home_nursing` 103 · `emergency_112` 56 · `lab_test_home` 51 · `teleconsultation` 48 · `paramedic_home_visit` 43 · follow-up (null) 39.

---

## `caregivers.csv` schema

`caregiver_id, name, role, qualification, gender, languages, skills, base_area, base_lat, base_lng, max_radius_km, tier, rating, experience_years, free_from, free_until, active`

- **role**: `medic` (GNM / B.Sc Nursing — does the physical home visit), `doctor` (MBBS — virtual only), `phlebotomist` (home sample collection).
- **languages / skills**: `;`-separated (e.g. `Hindi;English`, `injection_im;wound_dressing;vitals_monitoring`).
- **tier** (medics only): `base/standard/advanced/expert` — higher tiers unlock more procedures.
- **base_lat/lng + max_radius_km**: the caregiver serves requests within that radius.
- **free_from/free_until**: availability window today (24h IST).

---

## How `matched_caregivers.score` was generated (so you have a target to beat)

The reference labels score each eligible caregiver as a weighted blend — **language match (0.35)**,
**proximity (0.30, closer = higher, capped at the service radius)**, **skill match (0.20)**, and
**rating (0.15)** — then return the top 1–3. Teleconsultation ignores distance (doctors are virtual).
Emergencies return **no** caregivers. You are free to design a better matcher; this is a reasonable baseline.

---

## Suggested tasks (pick any / all)
1. **Triage classifier** — predict `care_type` + `urgency` from `user_input` (+ language robustness across Hinglish/Hindi/English).
2. **Red-flag / safety detector** — high-recall emergency detection → `emergency_112` (missing one is worse than a false alarm).
3. **Follow-up decision** — decide when to ask a `follow_up_question` vs commit to a `care_type`, and generate a good question.
4. **Caregiver matcher** — rank `caregivers.csv` for a request (language + distance + skill + availability); beat the baseline score.
5. **End-to-end** — assemble 1–4 into the full request→response contract.

---

## Notes & licence
- Fully synthetic, no PII — free to use, modify and redistribute for this hackathon. If you cite it: *"Sanocare synthetic care-triage dataset v1 (2026), provided for the IIIT-Delhi hackathon."*
- Geography is Delhi NCR (lat ≈ 28.3–28.75, lng ≈ 77.0–77.5).
- Public external datasets are allowed **if cited**; this synthetic set is provided so teams can start immediately without hunting for data.
- **Not medical advice / not clinical ground truth.** The red-flag and triage labels are illustrative, not validated for real-world use.
