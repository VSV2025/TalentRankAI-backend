from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(255))
    location = Column(String(255))
    resume_path = Column(String(512))
    resume_text = Column(Text)
    resume_hash = Column(String(64))
    skills = Column(JSON)
    experience_years = Column(Float, default=0)
    highlights = Column(JSON)
    phone = Column(String(50), nullable=True)
    resume_snippet = Column(String(500))
    verification_status = Column(String(50), default="pending")
    review_note = Column(String(500))
    consistency_score = Column(Float, default=1.0)
    email_confirmed = Column(Boolean, default=False)
    verification_token = Column(String(128))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scores = relationship("CandidateScore", back_populates="candidate")


class CandidateScore(Base):
    __tablename__ = "candidate_scores"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    overall_score = Column(Float, default=0)
    skills_match = Column(Float, default=0)
    semantic_relevance = Column(Float, default=0)
    behavioral_signal = Column(Float, default=0)
    career_trajectory = Column(Float, default=0)
    production_evidence = Column(Float, default=0.0)
    why_rank = Column(Text)
    evidence = Column(JSON)
    debate = Column(JSON)
    rank = Column(Integer)
    borderline = Column(Boolean, default=False)
    compute_path = Column(String(20), default="heuristic")
    graph_fit_score = Column(Float, default=50.0)
    skill_breadth_score = Column(Float, default=50.0)
    pipeline_timings = Column(JSON)
    career_trajectory_detail = Column(Text, nullable=True)
    gaps = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    candidate = relationship("Candidate", back_populates="scores")
    job = relationship("Job", back_populates="scores")
