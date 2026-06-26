from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime


class SubScores(BaseModel):
    skillsMatch: float
    semanticRelevance: float
    behavioralSignal: float
    careerTrajectory: float


class DebateTranscript(BaseModel):
    pro: str
    skeptic: str


class VerificationCheck(BaseModel):
    id: str
    label: str
    result: str  # pass | pending | review
    detail: str
    badge: Optional[str] = None


class VerificationResult(BaseModel):
    candidate_id: int
    checks: list[VerificationCheck]
    overall_status: str  # verified | review | pending


class CandidateCreate(BaseModel):
    name: str
    email: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        return v


class CandidateOut(BaseModel):
    id: int
    name: str
    email: str
    title: Optional[str] = None
    location: Optional[str] = None
    verification_status: str
    review_note: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RankedCandidate(BaseModel):
    id: int
    name: str
    email: str
    title: Optional[str] = None
    location: Optional[str] = None
    overallScore: float
    scores: SubScores
    verificationStatus: str
    borderline: bool
    highlights: list[str]
    whyRank: str
    evidence: list[str]
    debate: Optional[DebateTranscript] = None
    resumeSnippet: Optional[str] = None
    reviewNote: Optional[str] = None
    computePath: Optional[str] = None

    model_config = {"from_attributes": True}
