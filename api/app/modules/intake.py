"""Slot extraction (Groq JSON, with a regex fallback) + deterministic follow-up policy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal, Optional

import yaml
from pydantic import BaseModel, BeforeValidator, Field

from .. import config, groq_client
from .normalise import Normalised

YNU = ("yes", "no", "unknown")


def _enum(*allowed: str, default: str = "unknown"):
    def fix(v):
        v = str(v).strip().lower() if v is not None else default
        return v if v in allowed else default
    return BeforeValidator(fix)


class Patient(BaseModel):
    relation: Annotated[str, _enum("self", "father", "mother", "child", "spouse", "other", "unknown")] = "unknown"
    age_band: Annotated[str, _enum("<1", "1-5", "6-17", "18-40", "41-60", "60+", "unknown")] = "unknown"
    pregnant: Annotated[str, _enum(*YNU)] = "unknown"


class Symptom(BaseModel):
    name_en: str
    duration_days: Optional[float] = None
    severity: Annotated[str, _enum("mild", "moderate", "severe", "unknown")] = "unknown"


class Probes(BaseModel):
    breathing_difficulty: Annotated[str, _enum(*YNU)] = "unknown"
    chest_pain: Annotated[str, _enum(*YNU)] = "unknown"
    confusion: Annotated[str, _enum(*YNU)] = "unknown"
    unable_to_eat_drink: Annotated[str, _enum(*YNU)] = "unknown"


class Needs(BaseModel):
    procedure: Annotated[str, _enum("injection", "dressing", "catheter", "iv", "none", "unknown")] = "unknown"


class Prefs(BaseModel):
    caregiver_gender: Annotated[str, _enum("female", "male", "any", default="any")] = "any"
    time: Optional[str] = None


class Slots(BaseModel):
    patient: Patient = Field(default_factory=Patient)
    symptoms: list[Symptom] = Field(default_factory=list)
    red_flag_probes: Probes = Field(default_factory=Probes)
    chronic_conditions: list[str] = Field(default_factory=list)
    needs: Needs = Field(default_factory=Needs)
    language: Annotated[str, _enum("hi", "en", "hinglish", default="en")] = "en"
    preferences: Prefs = Field(default_factory=Prefs)


@lru_cache(maxsize=1)
def probes_cfg() -> dict:
    return yaml.safe_load((config.CONFIG_DIR / "probes.yaml").read_text(encoding="utf-8"))


# ---------------- LLM extraction ----------------
SYSTEM = """You extract structured slots from a home-healthcare request written in English, Hindi or Hinglish (often by a relative, not the patient).
Rules: extract ONLY what is stated or clearly implied; use "unknown" otherwise; NEVER diagnose, never name diseases as conclusions, never suggest medicines.
Return one JSON object with exactly this shape:
{"patient": {"relation": "self|father|mother|child|spouse|other|unknown", "age_band": "<1|1-5|6-17|18-40|41-60|60+|unknown", "pregnant": "yes|no|unknown"},
 "symptoms": [{"name_en": "fever", "duration_days": 2, "severity": "mild|moderate|severe|unknown"}],
 "red_flag_probes": {"breathing_difficulty": "yes|no|unknown", "chest_pain": "yes|no|unknown", "confusion": "yes|no|unknown", "unable_to_eat_drink": "yes|no|unknown"},
 "chronic_conditions": ["as stated by the caller, e.g. diabetes"],
 "needs": {"procedure": "injection|dressing|catheter|iv|none|unknown"},
 "language": "hi|en|hinglish",
 "preferences": {"caregiver_gender": "female|male|any", "time": "string or null"}}
A red_flag_probe is "no" only if the text says it is absent (e.g. "saans theek hai"); "yes" only if present; otherwise "unknown".
The transcript may include earlier assistant questions and the user's short answers (Haan = yes, Nahi = no) - apply the answers to the question asked."""


def _transcript(history: list[dict], current: str) -> str:
    lines = [f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}" for h in history]
    lines.append(f"User: {current}")
    return "\n".join(lines)


def extract_llm(history: list[dict], current: str) -> Optional[Slots]:
    data = groq_client.chat_json(SYSTEM, _transcript(history, current), name="intake")
    if data is None:
        return None
    try:
        return Slots.model_validate(data)
    except Exception:
        return None


# ---------------- regex fallback ----------------
_NUM = {"ek": 1, "one": 1, "a": 1, "do": 2, "two": 2, "teen": 3, "three": 3, "char": 4, "chaar": 4, "four": 4, "paanch": 5, "panch": 5,
        "five": 5, "chhe": 6, "six": 6, "saat": 7, "seven": 7, "aath": 8, "nau": 9, "das": 10}
_DUR = re.compile(r"(\d+|ek|one|a|do|two|teen|three|char|chaar|four|paanch|panch|five|chhe|six|saat|seven|aath|nau|das)\s*(din|dino|days?|hafte|hafta|weeks?|ghante|hours?)")
_SYMPTOMS = {
    "fever": r"bukhar|bukhaar|fever|tez garam|बुखार", "cough": r"khansi|khasi|cough|खांसी|खाँसी", "headache": r"(sar|sir)\s*dard|headache|सिर दर्द",
    "vomiting": r"ulti|vomit|उल्टी", "loose motions": r"dast|loose motion|diarr?hoea|दस्त", "abdominal pain": r"pet\s*(me|mein)?\s*dard|stomach\s*(pain|ache)|पेट",
    "cold": r"zukam|zukaam|cold|runny nose|ज़ुकाम", "sore throat": r"gale\s*(me|mein)?\s*(dard|kharash)|sore throat", "weakness": r"kamzori|weakness|कमज़ोरी",
    "dizziness": r"chakkar|dizz", "wound": r"ghav|wound|cut|chot|injury", "body pain": r"body\s*pain|badan\s*dard|jodon", "rash": r"daane|rash|khujli|allergy",
    "breathing trouble": r"saa?ns\s*(phool|lene)|breathless",
}
_CHRONIC = {"diabetes": r"sugar|diabet|madhumeh", "high blood pressure": r"\bbp\b|blood pressure|hypertension", "asthma": r"asthma|dama\b", "heart condition": r"heart|dil ki"}
_REL = [("father", r"papa|pita|father|dad|abbu|dada|nana|pitaji|daddy|chacha"), ("mother", r"mummy|mamma|maa\b|mother|mom|ammi|dadi|nani|maji"),
        ("child", r"bachch|beta|beti|baby|son\b|daughter|bete|bacha|kid"), ("spouse", r"patni|pati\b|wife|husband|biwi|shauhar"),
        ("self", r"\bmujhe\b|\bmera\b|\bmeri\b|\bmain\b|\bi have\b|\bmy (fever|cough|head)")]


def _num(s: str) -> float:
    return float(s) if s.isdigit() else float(_NUM.get(s, 1))


def extract_regex(norm: Normalised) -> Slots:
    t = norm.clean
    s = Slots(language=norm.language)
    for rel, pat in _REL:
        if re.search(pat, t):
            s.patient.relation = rel
            break
    m = re.search(r"(\d+)\s*(saal|sal|years?|yrs?|varsh|वर्ष|साल)", t)
    if m:
        a = int(m.group(1))
        s.patient.age_band = "<1" if a < 1 else "1-5" if a <= 5 else "6-17" if a <= 17 else "18-40" if a <= 40 else "41-60" if a <= 60 else "60+"
    elif re.search(r"\d+\s*(mahine|month|din ka|days? old)", t):
        s.patient.age_band = "<1"
    elif re.search(r"buzurg|boodha|bujurg|dada|dadi|nana|nani|elderly", t):
        s.patient.age_band = "60+"
    if re.search(r"pregnan|garbh|गर्भ", t):
        s.patient.pregnant = "yes"
    dur = None
    d = _DUR.search(t)
    if d:
        n = _num(d.group(1))
        unit = d.group(2)
        dur = n * (7 if unit.startswith(("haft", "week")) else 1 / 24 if unit.startswith(("ghant", "hour")) else 1)
    elif re.search(r"kal se|since yesterday|kal raat", t):
        dur = 1
    for name, pat in _SYMPTOMS.items():
        if re.search(pat, t):
            s.symptoms.append(Symptom(name_en=name, duration_days=dur))
    for name, pat in _CHRONIC.items():
        if re.search(pat, t):
            s.chronic_conditions.append(name)
    for proc, pat in (("injection", r"injection|इंजेक्शन|inj\b"), ("dressing", r"dressing|patti|पट्टी|ghav"), ("catheter", r"catheter"), ("iv", r"\biv\b|drip")):
        if re.search(pat, t):
            s.needs.procedure = proc
            break
    p = s.red_flag_probes
    if re.search(r"saa?ns[a-z\s]{0,14}(theek|normal|sahi)|saa?ns[a-z\s]{0,20}(dikkat|takleef|problem)\s*(nahi|nhi)|no (difficulty|trouble) breathing|breathing (is )?(normal|fine)", t):
        p.breathing_difficulty = "no"
    if re.search(r"(seene|chest|chhati)\s*(me|mein)?\s*(dard|pain)", t):
        p.chest_pain = "yes"
    if re.search(r"confus|ghabra|behosh|बेहोश", t):
        p.confusion = "yes"
    if re.search(r"(kuch|khana|paani|pani)\s*(nahi|nhi)\s*(kha|pee|pi)|not eating|unable to (eat|drink)", t):
        p.unable_to_eat_drink = "yes"
    if re.search(r"female|mahila|lady|ladki", t):
        s.preferences.caregiver_gender = "female"
    return s


# ---------------- deterministic answers to our own probe questions ----------------
_YES = {"haan", "han", "ha", "haa", "yes", "y", "yeah", "ji haan", "ji", "हाँ", "हां", "हा", "जी", "ok"}
_NO = {"nahi", "nahin", "nhi", "no", "n", "na", "nope", "नहीं", "नही", "ना"}


def _all_texts(key: str) -> dict[str, str]:
    q = probes_cfg()["probe_questions"][key]
    return {q[l]: l for l in ("en", "hinglish", "hi")}


def asked_probe(assistant_text: str) -> Optional[str]:
    for pid, q in probes_cfg()["probe_questions"].items():
        if assistant_text.strip() in q.values():
            return pid
    return None


def _yn(answer: str) -> Optional[str]:
    first = re.split(r"[\s,.!?]+", answer.strip().lower())
    for tok in (answer.strip().lower(), first[0] if first else ""):
        if tok in _YES:
            return "yes"
        if tok in _NO:
            return "no"
    return None


def resolve_answers(history: list[dict], current: str) -> tuple[dict, Optional[str], list[str]]:
    """Walk assistant-question / user-answer pairs. Returns (probe values, age_band, asked probe ids)."""
    cfg = probes_cfg()
    turns = [*history, {"role": "user", "content": current}]
    probes: dict[str, str] = {}
    age = None
    asked: list[str] = []
    for i, h in enumerate(turns[:-1]):
        if h["role"] != "assistant":
            continue
        pid = asked_probe(h["content"])
        if not pid:
            continue
        asked.append(pid)
        ans = turns[i + 1]["content"] if turns[i + 1]["role"] == "user" else ""
        if pid == "intent":
            continue                      # free-text answer to the generic question; the triage reads it from the history
        if pid == "age_band":
            for opts in (cfg["age_options"][l] for l in ("en", "hinglish", "hi")):
                if ans.strip() in opts:
                    age = cfg["age_option_bands"][opts.index(ans.strip())]
        else:
            v = _yn(ans)
            if v:
                probes[pid] = v
    return probes, age, asked


def extract(norm: Normalised, history: list[dict], current: str, profile: Optional[dict] = None) -> tuple[Slots, list[str], str]:
    """Slots for the whole conversation. Returns (slots, asked probe ids, source 'llm'|'regex')."""
    slots = extract_llm(history, current)
    source = "llm"
    if slots is None:
        slots, source = extract_regex(norm), "regex"
    answers, age, asked = resolve_answers(history, current)
    for k, v in answers.items():
        setattr(slots.red_flag_probes, k, v)
    if age:
        slots.patient.age_band = age
    if profile:   # repeat requests: stored patient profile fills gaps
        if slots.patient.age_band == "unknown" and profile.get("age_band"):
            slots.patient.age_band = profile["age_band"]
        if slots.patient.relation == "unknown" and profile.get("relation_to_caller"):
            slots.patient.relation = profile["relation_to_caller"]
        for c in profile.get("chronic_conditions") or []:
            if c not in slots.chronic_conditions:
                slots.chronic_conditions.append(c)
    slots.language = norm.language if norm.language in ("hi", "en", "hinglish") else slots.language
    return slots, asked, source


# ---------------- clusters + follow-up policy ----------------
def detect_cluster(norm: Normalised, slots: Slots) -> str:
    cfg = probes_cfg()
    text = norm.clean + " " + " ".join(s.name_en for s in slots.symptoms).lower()
    best, best_hits = "general", 0
    for name in cfg["cluster_order"]:
        kws = cfg["clusters"][name]["keywords"]
        hits = sum(1 for k in kws if re.search(r"(?<![a-z])" + re.escape(k.lower()), text))
        if name == "child" and slots.patient.relation == "child" or (name == "child" and slots.patient.age_band in ("<1", "1-5")):
            hits += 1
        if name == "pregnancy" and slots.patient.pregnant == "yes":
            hits += 1
        if name == "chronic_procedure" and slots.needs.procedure not in ("none", "unknown"):
            hits += 2
        if hits > best_hits:
            best, best_hits = name, hits
    return best


@dataclass
class Question:
    probe_id: str
    text: str
    options: list[str]


def _lang(norm_lang: str) -> str:
    return norm_lang if norm_lang in ("en", "hinglish", "hi") else "en"


def generic_question(lang: str) -> Question:
    """The one clarifying question for a request too vague to route."""
    cfg = probes_cfg()
    lang = _lang(lang)
    return Question("intent", cfg["probe_questions"]["intent"][lang], cfg["intent_options"][lang])


def next_question(cluster: str, slots: Slots, asked: list[str], lang: str, include_optional: bool) -> Optional[Question]:
    cfg = probes_cfg()
    lang = _lang(lang)
    for p in cfg["clusters"][cluster]["probes"]:
        pid = p["id"]
        if pid in asked or (not p.get("critical") and not include_optional):
            continue
        if pid == "age_band":
            if slots.patient.age_band != "unknown" or slots.patient.relation in cfg["parent_relations"]:
                continue
            return Question(pid, cfg["probe_questions"][pid][lang], cfg["age_options"][lang])
        if getattr(slots.red_flag_probes, pid) == "unknown":
            return Question(pid, cfg["probe_questions"][pid][lang], cfg["yes_no_options"][lang])
    return None
