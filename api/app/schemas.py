"""Pydantic models.

CONTRACT models (Location .. TriageResponse) follow section 5.1 of the brief. NOTE: no organiser
API-schema file was available at build time; when it arrives, edit ONLY this block (organisers' schema wins).
Label sets (CareType / Urgency) are built from the taxonomy file, never hard-coded.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .data_loader import load_taxonomy

_tax = load_taxonomy()
CareType = Literal[tuple(_tax.care_types)]       # type: ignore[valid-type]
Urgency = Literal[tuple(_tax.urgency_levels)]    # type: ignore[valid-type]


# ---------------- CONTRACT ----------------
class Location(BaseModel):
    lat: float
    lng: float


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TriageRequest(BaseModel):
    user_input: str = Field(..., examples=["papa ko do din se bukhar hai aur khansi bhi"])
    language: str = "auto"
    location: Optional[Location] = None
    conversation_history: list[Turn] = Field(default_factory=list)


class MatchedCaregiver(BaseModel):
    model_config = ConfigDict(extra="forbid")
    caregiver_id: str
    score: float = Field(..., ge=0, le=1)
    reason: str


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    follow_up_question: Optional[str] = None
    care_type: Optional[CareType] = None            # type: ignore[valid-type]
    urgency: Optional[Urgency] = None               # type: ignore[valid-type]
    red_flags: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    matched_caregivers: list[MatchedCaregiver] = Field(default_factory=list)


# ---------------- APP (not part of the scoring contract) ----------------
class SessionTriageRequest(TriageRequest):
    patient_id: Optional[str] = None
    family_id: Optional[str] = None
    case_id: Optional[str] = None


class PatientSummary(BaseModel):
    relation: str = "unknown"
    age_band: str = "unknown"


class AppMeta(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    family_id: Optional[str] = None
    status: str = "open"
    confidence: float = 0.0
    needs_nurse_review: bool = False
    is_emergency: bool = False
    language: str = "en"
    watch_list: list[str] = Field(default_factory=list)
    follow_up_options: list[str] = Field(default_factory=list)
    question_no: int = 0
    patient_summary: PatientSummary = Field(default_factory=PatientSummary)
    speak_text: str = ""
    key_factors: list[str] = Field(default_factory=list)
    caregivers: list[dict] = Field(default_factory=list)     # matched caregivers with profile + eta (for the UI)
    emergency_steps: list[str] = Field(default_factory=list)


class SessionTriageResponse(BaseModel):
    triage: TriageResponse
    app: AppMeta


class BookRequest(BaseModel):
    case_id: str
    caregiver_id: str
    slot: str


class VisitBrief(BaseModel):
    summary_lines: list[str]
    bring_list: list[str]


class BookResponse(BaseModel):
    booking_id: str
    visit_brief: VisitBrief
    caregiver: dict


class TranscribeResponse(BaseModel):
    text: str
    language: str


class ReplyRequest(BaseModel):
    text: str


class NurseApproveRequest(BaseModel):
    care_type: Optional[CareType] = None            # type: ignore[valid-type]
    urgency: Optional[Urgency] = None               # type: ignore[valid-type]
    reviewer_name: str = "Nurse"
    note: str = ""


class ReplyResponse(BaseModel):
    escalated: bool
    closed: bool = False
    message: str = ""
    triage: Optional[TriageResponse] = None
    app: Optional[AppMeta] = None
