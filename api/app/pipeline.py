"""normalise -> red flags -> intake -> triage -> matching -> explain. Pure: no persistence."""
from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from . import config
from .data_loader import load_caregivers, load_taxonomy
from .modules import explain, intake, matching, red_flags
from .modules.normalise import normalise
from .modules.retrieval import get_retriever
from .modules.triage import TriageResult, decide, score
from .schemas import AppMeta, MatchedCaregiver, PatientSummary, TriageRequest, TriageResponse


@dataclass
class PipelineOut:
    contract: TriageResponse
    app: AppMeta
    slots: intake.Slots
    cluster: str
    result: Optional[TriageResult]
    is_emergency: bool


_cache: "OrderedDict[str, PipelineOut]" = OrderedDict()


def clear_cache():
    _cache.clear()


def _caregiver_cards(ms: list[matching.Match]) -> list[dict]:
    return [{"caregiver_id": m.caregiver.id, "name": m.caregiver.name, "role": m.caregiver.role, "skills": m.caregiver.skills,
             "languages": m.caregiver.languages, "rating": m.caregiver.rating, "review_count": m.caregiver.review_count,
             "gender": m.caregiver.gender, "lat": m.caregiver.lat, "lng": m.caregiver.lng,
             "km": round(m.km, 1) if m.km is not None else None, "slot": m.slot.isoformat() if m.slot else None,
             "score": m.score, "reason": m.reason} for m in ms]


def _summary(slots: intake.Slots) -> PatientSummary:
    return PatientSummary(relation=slots.patient.relation, age_band=slots.patient.age_band)


def _emergency(flags: list[str], slots, cluster, lang, case_id, conf=1.0) -> PipelineOut:
    tax = load_taxonomy()
    contract = TriageResponse(care_type=tax.emergency_label, urgency=tax.immediate_label, red_flags=red_flags.to_vocab(flags),
                              reasoning=explain.emergency_reasoning(flags), matched_caregivers=[])
    app = AppMeta(case_id=case_id, confidence=conf, needs_nurse_review=False, is_emergency=True, language=lang,
                  patient_summary=_summary(slots), speak_text=explain.EMERGENCY_SPEAK[explain._l(lang)],
                  emergency_steps=explain.EMERGENCY_STEPS[explain._l(lang)])
    return PipelineOut(contract, app, slots, cluster, None, True)


def _follow_up(q: intake.Question, slots, cluster, lang, case_id, n_asked) -> PipelineOut:
    contract = TriageResponse(follow_up_question=q.text)
    app = AppMeta(case_id=case_id, language=lang, follow_up_options=q.options, question_no=n_asked + 1,
                  patient_summary=_summary(slots), speak_text=q.text)
    return PipelineOut(contract, app, slots, cluster, None, False)


def _vague(q: intake.Question, slots, cluster, lang, case_id, n_asked) -> PipelineOut:
    return _follow_up(q, slots, cluster, lang, case_id, n_asked)


def run(req: TriageRequest, force_decision: bool = False, app_mode: bool = False, profile: Optional[dict] = None,
        case_id: Optional[str] = None, served_ids: Optional[set[str]] = None) -> PipelineOut:
    history = [t.model_dump() for t in req.conversation_history]
    loc = req.location.model_dump() if req.location else None
    key = None
    if not app_mode:
        key = hashlib.sha1(json.dumps([req.user_input, history, loc and [round(loc["lat"], 2), round(loc["lng"], 2)], force_decision, req.language],
                                      ensure_ascii=False).encode()).hexdigest()
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    out = _run(req, history, loc, force_decision, app_mode, profile, case_id or str(uuid.uuid4()), served_ids)
    if key:
        _cache[key] = out
        if len(_cache) > 256:
            _cache.popitem(last=False)
    return out


def _run(req, history, loc, force, app_mode, profile, case_id, served_ids) -> PipelineOut:
    T = config.THRESHOLDS
    tax = load_taxonomy()
    norm = normalise(req.user_input, history, req.language)
    lang = norm.language

    # 1) deterministic gate - first, every turn, no LLM
    flags = red_flags.check_text(norm)
    if flags:
        return _emergency(flags, intake.extract_regex(norm), "general", lang, case_id)

    # 2) intake (LLM, regex fallback) + gate on extracted slots
    slots, asked, _src = intake.extract(norm, history, req.user_input, profile)
    slot_flags = red_flags.check_slots(slots.red_flag_probes.model_dump())
    slot_flags += red_flags.check_text(normalise(" . ".join(s.name_en for s in slots.symptoms), None)) if slots.symptoms else []
    if slot_flags:
        return _emergency(list(dict.fromkeys(slot_flags)), slots, "general", lang, case_id)
    cluster = intake.detect_cluster(norm, slots)
    can_ask = (not force) and len(asked) < T["max_questions"]

    # 3) app only: templated safety probes (breathing / chest pain / confusion ...), one question at a time.
    #    The scoring endpoint (/triage) does not ask these: the organisers' data decides such requests directly.
    if can_ask and app_mode:
        q = intake.next_question(cluster, slots, asked, lang, include_optional=False)
        if q:
            return _follow_up(q, slots, cluster, lang, case_id, len(asked))

    # 4) triage. With history, retrieval looks at the newest turn first (an earlier vague turn must not drag the neighbours)
    hits = get_retriever().top_k(norm.clean_current if history else norm.clean, 6)
    care_s, urg_s, kf, src = score(norm.full, slots, hits)
    res = decide(care_s, urg_s, slots, kf, src, allow_follow_up=can_ask)

    if res.follow_up:                                   # too vague to route: ask ONE generic question (also applies to the app)
        return _vague(intake.generic_question(lang), slots, cluster, lang, case_id, len(asked))

    if app_mode and can_ask and res.confidence < T["ask_if_conf_below"] and res.care_type != tax.emergency_label:
        q = intake.next_question(cluster, slots, asked, lang, include_optional=True)
        if q:
            return _follow_up(q, slots, cluster, lang, case_id, len(asked))

    if res.care_type == tax.emergency_label:
        return _emergency(["model_triage_emergency"], slots, cluster, lang, case_id, res.confidence)

    # 5) matching + explanation
    ms = matching.match(res.care_type, res.urgency, cluster, lang, loc, top_n=3, text=norm.full, slot_procedure=slots.needs.procedure,
                        gender_pref=slots.preferences.caregiver_gender, served_ids=served_ids, app_bonuses=app_mode)
    contract = TriageResponse(
        care_type=res.care_type, urgency=res.urgency, red_flags=[], reasoning=explain.reasoning(res, slots),
        matched_caregivers=[MatchedCaregiver(caregiver_id=m.caregiver.id, score=m.score, reason=m.reason) for m in ms])
    app = AppMeta(case_id=case_id, confidence=res.confidence, needs_nurse_review=res.confidence < T["t_nurse_review"],
                  language=lang, watch_list=explain.watch_list(cluster, lang), patient_summary=_summary(slots),
                  speak_text=explain.speak_decision(res, lang), key_factors=res.key_factors, caregivers=_caregiver_cards(ms))
    return PipelineOut(contract, app, slots, cluster, res, False)


def rebuild_decision(slots: intake.Slots, cluster: str, base: Optional[TriageResult], care_type: str, urgency: str,
                     lang: str, loc: Optional[dict], case_id: str, confidence: float = 1.0) -> PipelineOut:
    """Re-issue a decision with a given care_type/urgency (nurse edit, Safety Net escalation)."""
    tax = load_taxonomy()
    if care_type == tax.emergency_label or urgency == tax.immediate_label:
        return _emergency(["nurse_or_followup_escalation"], slots, cluster, lang, case_id, confidence)
    res = TriageResult(care_type, urgency, base.care_scores if base else {}, base.urgency_scores if base else {},
                       base.key_factors if base else [], confidence, base.source if base else "manual")
    ms = matching.match(care_type, urgency, cluster, lang, loc, top_n=3, slot_procedure=slots.needs.procedure,
                        gender_pref=slots.preferences.caregiver_gender, app_bonuses=True)
    contract = TriageResponse(care_type=care_type, urgency=urgency, red_flags=[], reasoning=explain.reasoning(res, slots),
                              matched_caregivers=[MatchedCaregiver(caregiver_id=m.caregiver.id, score=m.score, reason=m.reason) for m in ms])
    app = AppMeta(case_id=case_id, confidence=confidence, needs_nurse_review=False, language=lang,
                  watch_list=explain.watch_list(cluster, lang), patient_summary=_summary(slots),
                  speak_text=explain.speak_decision(res, lang), key_factors=res.key_factors, caregivers=_caregiver_cards(ms))
    return PipelineOut(contract, app, slots, cluster, res, False)
