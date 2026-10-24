"""Triage: one Groq JSON call -> score distributions -> cost-aware decision -> consistency rules.

Care labels come from the dataset taxonomy plus a pseudo-label `follow_up` (request too vague: ask one question, decide nothing).
In the organisers' data, urgency is a function of care type (lab -> scheduled, every other home/teleconsult -> same_day,
emergency -> emergency), so after the decision urgency is set from the learned prior (see `care_urgency_prior`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from .. import config, groq_client
from .. import data_loader
from ..data_loader import FOLLOW_UP, load_taxonomy
from .intake import Slots
from .retrieval import Hit


@dataclass
class TriageResult:
    care_type: Optional[str]          # None when follow_up is True
    urgency: Optional[str]
    care_scores: dict[str, float]
    urgency_scores: dict[str, float]
    key_factors: list[str]
    confidence: float
    source: str                      # llm | knn | manual
    emergency_by_model: bool = False
    follow_up: bool = False
    notes: list[str] = field(default_factory=list)


def care_labels() -> list[str]:
    return load_taxonomy().care_types + [FOLLOW_UP]


def _norm(d: dict, labels: list[str]) -> dict[str, float]:
    v = {l: max(float(d.get(l, 0) or 0), 0.0) for l in labels}
    s = sum(v.values())
    return {l: x / s for l, x in v.items()} if s > 0 else {l: 1 / len(labels) for l in labels}


@lru_cache(maxsize=1)
def care_urgency_prior() -> dict[str, tuple[str, float]]:
    """care type -> (majority urgency, purity) learned from the training split."""
    counts: dict[str, dict[str, int]] = {}
    for c in data_loader.load_split(data_loader.TRAIN_SPLIT):
        if c.care_type and c.urgency:
            counts.setdefault(c.care_type, {}).setdefault(c.urgency, 0)
            counts[c.care_type][c.urgency] += 1
    return {ct: (max(u, key=u.get), max(u.values()) / sum(u.values())) for ct, u in counts.items()}


def build_prompt(text: str, slots: Slots, hits: list[Hit]) -> tuple[str, str]:
    tax = load_taxonomy()
    defs = "\n".join(f"- {l}: {tax.definitions.get(l, '')}" for l in tax.care_types + tax.urgency_levels)
    prior = care_urgency_prior()
    rules = "; ".join(f"{c} -> {u}" for c, (u, p) in prior.items() if p >= 0.95)
    care_keys = ", ".join(f'"{l}": 0.0' for l in care_labels())
    urg_keys = ", ".join(f'"{l}": 0.0' for l in tax.urgency_levels)
    system = f"""You are a triage ROUTER for a home-healthcare service in India. You route; you NEVER diagnose, name diseases, or mention medicines.
Definitions (from the service taxonomy):
{defs}
- follow_up: {tax.follow_up_definition}
Urgency usually follows the care type: {rules}. Emergency/emergency_112 only for red flags (chest pain, trouble breathing, unconsciousness, severe bleeding, stroke signs, seizure, infant fever, suicidal thoughts, severe dehydration).
When unsure between care types prefer the more capable option; when the request is really too vague to act on choose follow_up.
Return JSON: {{"care_type_scores": {{{care_keys}}}, "urgency_scores": {{{urg_keys}}},
"key_factors": ["short phrases from the request, e.g. fever 2 days, injection at home"]}}
Each score dict is a probability distribution that sums to 1."""
    shots = "\n".join(f'- "{h.case.text}" -> {h.case.care_label} / {h.case.urgency_label}' for h in hits)
    user = f"Request:\n{text}\n\nExtracted slots:\n{slots.model_dump_json()}\n\nSimilar labelled cases:\n{shots}"
    return system, user


def score_llm(text: str, slots: Slots, hits: list[Hit]) -> Optional[tuple[dict, dict, list[str]]]:
    tax = load_taxonomy()
    system, user = build_prompt(text, slots, hits)
    d = groq_client.chat_json(system, user, name="triage")
    if not d or "care_type_scores" not in d or "urgency_scores" not in d:
        return None
    kf = [str(x) for x in d.get("key_factors", [])][:6]
    return _norm(d["care_type_scores"], care_labels()), _norm(d["urgency_scores"], tax.urgency_levels), kf


def score_knn(hits: list[Hit]) -> tuple[dict, dict]:
    """Similarity-weighted vote over retrieved cases (Groq fallback)."""
    tax = load_taxonomy()
    c = {l: 1e-3 for l in care_labels()}
    u = {l: 1e-3 for l in tax.urgency_levels}
    for h in hits:
        w = h.sim ** 2 + 1e-6
        c[h.case.care_label] = c.get(h.case.care_label, 0) + w
        if h.case.urgency in u:
            u[h.case.urgency] += w
    return _norm(c, care_labels()), _norm(u, tax.urgency_levels)


def score(text: str, slots: Slots, hits: list[Hit]) -> tuple[dict, dict, list[str], str]:
    r = score_llm(text, slots, hits)
    if r:
        return (*r, "llm")
    c, u = score_knn(hits)
    return c, u, [], "knn"


def _bump(urg: str) -> str:
    lv = load_taxonomy().urgency_levels
    return lv[min(lv.index(urg) + 1, len(lv) - 1)]


def decide(care_scores: dict, urg_scores: dict, slots: Slots, key_factors: list[str], source: str,
           T: Optional[dict] = None, allow_follow_up: bool = True) -> TriageResult:
    T = T or config.THRESHOLDS
    tax = load_taxonomy()
    lv = tax.urgency_levels
    routine, same_day, immediate = lv[0], lv[1], lv[-1]
    em = tax.emergency_label
    notes: list[str] = []
    conf = max(care_scores.values()) * max(urg_scores.values())

    emergency_by_model = care_scores.get(em, 0) >= T["t_emergency"]
    # kNN retrieval is negation-blind: if the caller explicitly ruled a red flag out, don't let lexical neighbours alone raise an emergency
    veto = source == "knn" and "no" in slots.red_flag_probes.model_dump().values()
    if veto:
        emergency_by_model = False
        notes.append("kNN emergency vote vetoed by explicit negative probe")
    if emergency_by_model:
        return TriageResult(em, immediate, care_scores, urg_scores, key_factors, round(conf, 3), source, True, False, notes)

    # request too vague -> ask (only when allowed)
    if allow_follow_up and care_scores.get(FOLLOW_UP, 0) >= T.get("t_follow_up", 0.5):
        return TriageResult(None, None, care_scores, urg_scores, key_factors, round(care_scores[FOLLOW_UP], 3), source, False, True, notes)

    # care type: argmax of the non-emergency labels. Ties within tie_margin (default 0: this taxonomy has no capability order) go to the later label
    non_em = [l for l in tax.care_types if l != em]
    top = max(care_scores[l] for l in non_em)
    care = max((l for l in non_em if top - care_scores[l] <= T["tie_margin"]), key=lambda l: (care_scores[l], tax.care_types.index(l)))

    # urgency from cost-aware thresholds
    if urg_scores[immediate] >= T["t_immediate"]:
        urgency = immediate
    elif urg_scores[same_day] + urg_scores[immediate] >= T["t_same_day"]:
        urgency = same_day
    else:
        urgency = routine
    if veto and urgency == immediate:      # the veto also covers an 'emergency' urgency vote
        urgency = same_day
    if source == "knn" and T.get("fallback_round_up", True) and urgency == routine:
        urgency = _bump(urgency)
        notes.append("fallback: urgency rounded up")

    # procedure need -> nurse label
    if tax.nurse_label and slots.needs.procedure not in ("none", "unknown") and urgency != immediate:
        care = tax.nurse_label
        notes.append("procedure need -> nursing")

    # emergency <=> immediate
    if urgency == immediate:
        return TriageResult(em, immediate, care_scores, urg_scores, key_factors, round(conf, 3), source, False, False, notes)

    # urgency is a function of care type in the organisers' data
    prior = care_urgency_prior().get(care)
    if prior and prior[1] >= T.get("urgency_prior_purity", 0.95) and urgency != prior[0]:
        notes.append(f"urgency set from care-type prior ({prior[0]})")
        urgency = prior[0]
    return TriageResult(care, urgency, care_scores, urg_scores, key_factors, round(conf, 3), source, False, False, notes)
