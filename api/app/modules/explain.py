"""Explanations: reasoning, watch-list, emergency copy, visit brief. Templates first; LLM text is guard-railed."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from .. import config, groq_client
from ..data_loader import load_taxonomy
from .intake import Slots, probes_cfg
from .triage import TriageResult
from . import red_flags

# ---------------- output guardrail ----------------
_MED = re.compile(r"\b(\d+\s*(mg|ml|mcg|gm|g)\b|mg\b|tablets?|tab\b|capsules?|syrup|goli|paracetamol|crocin|dolo|ibuprofen|aspirin|antibiotics?|"
                  r"amoxicillin|azithromycin|cetirizine|dose|dosage|prescri\w+|medicine|dawai|dawa)\b", re.I)
_DISEASE = re.compile(r"\b(pneumonia|malaria|dengue|covid\w*|typhoid|tuberculosis|tb|asthma|bronchitis|influenza|flu|cancer|jaundice|"
                      r"hepatitis|appendicitis|gastroenteritis|viral|infection|diabetes|hypertension|sepsis|meningitis)\b", re.I)
_DIAG_CUE = re.compile(r"\b(you have|you are suffering|suffering from|diagnos\w+|aapko|unhe|inhe|likely|probably|looks like|seems to be|hai\b.*\bbimari)\b", re.I)


def is_safe(text: str, allow: tuple[str, ...] = ()) -> bool:
    """False if text contains medicine/dose words or diagnosis phrasing. `allow` = caller-reported terms to ignore."""
    t = text
    for a in allow:
        t = re.sub(re.escape(a), " ", t, flags=re.I)
    if _MED.search(t):
        return False
    if _DISEASE.search(t) and _DIAG_CUE.search(t):
        return False
    return True


# ---------------- vocab ----------------
CARE_NAMES = {
    "paramedic_home_visit": {"en": "Medic home visit", "hinglish": "Medic ka ghar par visit", "hi": "मेडिक का घर पर विज़िट"},
    "home_nursing": {"en": "home nursing visit", "hinglish": "ghar par nursing visit", "hi": "घर पर नर्सिंग विज़िट"},
    "lab_test_home": {"en": "lab test at home", "hinglish": "ghar par lab test (sample collection)", "hi": "घर पर लैब टेस्ट (सैंपल कलेक्शन)"},
    "teleconsultation": {"en": "doctor teleconsult", "hinglish": "doctor se phone/video salah", "hi": "डॉक्टर से फ़ोन/वीडियो सलाह"},
    "emergency_112": {"en": "emergency help", "hinglish": "emergency madad", "hi": "आपातकालीन मदद"},
}
WHEN = {
    "scheduled": {"en": "scheduled (book a time)", "hinglish": "tay samay par", "hi": "तय समय पर"},
    "same_day": {"en": "today", "hinglish": "aaj hi", "hi": "आज ही"},
    "emergency": {"en": "right now", "hinglish": "abhi", "hi": "अभी"},
}
URG_NAMES = {"scheduled": "scheduled", "same_day": "same-day", "emergency": "emergency"}
NEED_TEXT = {   # what the request is about, when no symptom was extracted
    "paramedic_home_visit": "General health check requested at home",
    "home_nursing": "Nursing procedure requested at home",
    "lab_test_home": "Diagnostic test with home sample collection requested",
    "teleconsultation": "Non-urgent doctor consultation requested",
}
DISCLAIMER = {"en": "This is guidance, not a diagnosis.", "hinglish": "Yeh salah hai, diagnosis nahi.", "hi": "यह सलाह है, निदान नहीं।"}
EMERGENCY_SPEAK = {
    "en": "This may be an emergency. Call 112 or 108 now. Stay with the patient.",
    "hinglish": "Yeh emergency ho sakti hai. Abhi 112 ya 108 par call karein. Mareez ke saath rahein.",
    "hi": "यह आपातकाल हो सकता है। अभी 112 या 108 पर कॉल करें। मरीज़ के साथ रहें।",
}
EMERGENCY_STEPS = {
    "en": ["Stay with the patient and stay calm.", "Lay them down safely; do not give food or drink.", "Keep your phone free so the ambulance can call you."],
    "hinglish": ["Mareez ke saath rahein, ghabraiye nahi.", "Unhe surakshit lita dein; khana ya paani na dein.", "Phone khali rakhein taaki ambulance aapko call kar sake."],
    "hi": ["मरीज़ के साथ रहें, घबराएँ नहीं।", "उन्हें सुरक्षित लिटा दें; खाना या पानी न दें।", "फ़ोन खाली रखें ताकि एम्बुलेंस आपको कॉल कर सके।"],
}


def _l(lang: str) -> str:
    return lang if lang in ("en", "hinglish", "hi") else "en"


def care_name(label: str, lang: str) -> str:
    return CARE_NAMES.get(label, {}).get(_l(lang)) or label.replace("_", " ")


# ---------------- reasoning (contract field; English) ----------------
def reasoning(res: TriageResult, slots: Slots) -> str:
    sx = [s for s in slots.symptoms]
    if sx:
        dur = max((s.duration_days or 0) for s in sx)
        dur_txt = f" for {dur:g} day{'s' if dur != 1 else ''}" if dur >= 1 else ""
        complaint = " and ".join(s.name_en for s in sx[:3]) + dur_txt
    elif res.key_factors:
        complaint = ", ".join(res.key_factors[:3])
    else:
        complaint = NEED_TEXT.get(res.care_type, "The request described")
    p = slots.red_flag_probes
    absent = [txt for k, txt in (("breathing_difficulty", "breathing difficulty"), ("chest_pain", "chest pain"), ("confusion", "confusion")) if getattr(p, k) == "no"]
    absent_txt = f"; no {' or '.join(absent)} reported" if absent else ""
    risk = []
    if slots.patient.age_band in ("<1", "60+"):
        risk.append(f"age {slots.patient.age_band}")
    if slots.patient.pregnant == "yes":
        risk.append("pregnancy")
    if slots.chronic_conditions:
        risk.append("existing condition reported")
    risk_txt = f" Extra caution because of {', '.join(risk)}." if risk else ""
    out = f"{complaint[:1].upper()}{complaint[1:]}{absent_txt}. Routed to {care_name(res.care_type, 'en')} ({URG_NAMES.get(res.urgency, res.urgency)}).{risk_txt}"
    allow = tuple(slots.chronic_conditions)
    if config.THRESHOLDS.get("smooth_reasoning_with_llm"):
        d = groq_client.chat_json('Rewrite the sentence in plain simple English, one sentence, same facts, no new facts, no disease names, no medicines. JSON {"text": "..."}', out, name="smooth")
        if d and isinstance(d.get("text"), str) and is_safe(d["text"], allow):
            return d["text"]
    return out


def emergency_reasoning(flags: list[str], lang: str = "en") -> str:
    labels = ", ".join(dict.fromkeys(red_flags.label(f, "en") for f in flags)) or "possible emergency signs"
    extra = " If someone may harm themselves, also call Tele-MANAS 14416." if "suicidal" in flags else ""
    return f"Possible emergency ({labels}). Call 112 (national emergency) or 108 (ambulance) now and stay with the patient.{extra}"


# ---------------- watch list ----------------
def watch_list(cluster: str, lang: str) -> list[str]:
    cfg = probes_cfg()
    lang = _l(lang)
    suffix = cfg["watch_suffix"][lang]
    return [f"{w[lang]} {suffix}" for w in cfg["clusters"][cluster]["watch"]][:4]


def checkin_question(lang: str) -> tuple[str, list[str]]:
    cfg = probes_cfg()
    lang = _l(lang)
    return cfg["checkin_questions"][lang], cfg["checkin_options"][lang]


def speak_decision(res: TriageResult, lang: str) -> str:
    lang = _l(lang)
    care = care_name(res.care_type, lang)
    when = WHEN.get(res.urgency, {}).get(lang, res.urgency)
    lead = {"en": f"We suggest: {care}, {when}.", "hinglish": f"Humari salah: {care}, {when}.", "hi": f"हमारी सलाह: {care}, {when}।"}[lang]
    return f"{lead} {DISCLAIMER[lang]}"


# ---------------- visit brief ----------------
class _BriefOut(BaseModel):
    summary_lines: list[str]
    bring_list: list[str]


def _yn(v: str) -> str:
    return {"yes": "yes", "no": "no"}.get(v, "not asked")


def brief_template(slots: Slots, res_care: str, cluster: str, reasoning_txt: str, contact: str = "") -> dict:
    p = slots.red_flag_probes
    sx = ", ".join(f"{s.name_en}{f' ({s.duration_days:g} d)' if s.duration_days else ''}" for s in slots.symptoms) or "see case notes"
    risks = []
    if slots.patient.age_band in ("<1", "60+"):
        risks.append(f"age {slots.patient.age_band}")
    if slots.patient.pregnant == "yes":
        risks.append("pregnant")
    risks += [f"reported: {c}" for c in slots.chronic_conditions]
    bring = list(dict.fromkeys(probes_cfg()["clusters"][cluster]["bring"]))
    return {
        "summary_lines": [
            f"Patient: {slots.patient.relation} of caller, age band {slots.patient.age_band}.",
            f"Main complaint: {sx}.",
            f"Red flags checked - breathing difficulty: {_yn(p.breathing_difficulty)}; chest pain: {_yn(p.chest_pain)}; confusion: {_yn(p.confusion)}; unable to eat/drink: {_yn(p.unable_to_eat_drink)}.",
            f"Risk factors: {'; '.join(risks) if risks else 'none reported'}.",
            f"Family contact: {contact or 'via caller'}; language: {slots.language}; routed as {res_care.replace('_', ' ')}.",
        ],
        "bring_list": bring,
    }


def visit_brief(slots: Slots, res_care: str, cluster: str, reasoning_txt: str, contact: str = "") -> dict:
    base = brief_template(slots, res_care, cluster, reasoning_txt, contact)
    allowed = probes_cfg()["allowed_bring"]
    system = ('You write a short handover for a home-visit caregiver. You never diagnose or mention medicines/doses. '
              f'Return JSON {{"summary_lines": [exactly 5 short lines: who, main complaint+duration, red flags checked, risk factors, family contact preference], '
              f'"bring_list": [items chosen ONLY from: {", ".join(allowed)}]}}')
    d = groq_client.chat_json(system, f"Slots: {slots.model_dump_json()}\nRouting: {res_care}\nReasoning: {reasoning_txt}", _BriefOut, name="brief")
    if d and len(d["summary_lines"]) == 5:
        bring = [b for b in d["bring_list"] if b in allowed] or base["bring_list"]
        if all(is_safe(l, tuple(slots.chronic_conditions)) for l in d["summary_lines"]):
            return {"summary_lines": [l.strip() for l in d["summary_lines"]], "bring_list": bring}
    return base
