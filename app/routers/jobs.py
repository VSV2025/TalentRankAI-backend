"""Job creation and ranking endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.candidate import Candidate, CandidateScore
from ..models.job import Job
from ..schemas.candidate import RankedCandidate, SubScores, DebateTranscript
from ..schemas.job import JobCreate, JobOut, FunnelStage, RankRequest
from ..services.pipeline import run_pipeline

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def create_job(body: JobCreate, db: Session = Depends(get_db)):
    job = Job(title=body.title, description=body.description)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/{job_id}/rank", response_model=list[RankedCandidate])
def rank_candidates(
    job_id: int,
    body: Optional[RankRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Run the full 7-layer pipeline for a job.
    If body.description is provided, update the job description first.
    Returns the ranked shortlist.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if body and body.description:
        job.description = body.description
        db.commit()

    # Fetch all candidates
    candidates_db = db.query(Candidate).all()
    if not candidates_db:
        raise HTTPException(status_code=400, detail="No candidates in the database. Run seed first.")

    # Build pipeline input dicts
    candidate_dicts = []
    for c in candidates_db:
        candidate_dicts.append({
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "title": c.title or "",
            "location": c.location or "",
            "skills": c.skills or [],
            "resume_text": c.resume_text or "",
            "experience_years": c.experience_years or 3.0,
            "highlights": c.highlights or [],
            "resume_snippet": c.resume_snippet or "",
            "verification_status": c.verification_status,
            "review_note": c.review_note,
        })

    # Run pipeline
    try:
        result = run_pipeline(job.description, candidate_dicts)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    ranked = result["ranked"]
    funnel_counts = result["funnel_counts"]

    # Persist scores to DB
    db.query(CandidateScore).filter(CandidateScore.job_id == job_id).delete()
    for cand in ranked:
        score_row = CandidateScore(
            candidate_id=cand["id"],
            job_id=job_id,
            overall_score=cand.get("overall_score", 0),
            skills_match=cand.get("skills_match", 0),
            semantic_relevance=cand.get("semantic_relevance", 0),
            behavioral_signal=cand.get("behavioral_signal", 0),
            career_trajectory=cand.get("career_trajectory", 0),
            why_rank=cand.get("why_rank", ""),
            evidence=cand.get("evidence", []),
            debate=cand.get("debate"),
            rank=cand.get("rank", 0),
            borderline=cand.get("borderline", False),
            compute_path=cand.get("compute_path", "fast"),
            pipeline_timings=result.get("timings"),
        )
        db.add(score_row)

    # Store funnel counts on job
    job.funnel_counts = funnel_counts
    job.requirements = result.get("requirements")
    db.commit()

    logger.info(f"Job {job_id} ranked {len(ranked)} candidates")

    return _build_ranked_response(ranked, candidates_db)


@router.get("/{job_id}/rank", response_model=list[RankedCandidate])
def get_ranked_candidates(job_id: int, db: Session = Depends(get_db)):
    """Return the latest cached rankings for a job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    scores = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .order_by(CandidateScore.rank)
        .all()
    )
    if not scores:
        raise HTTPException(
            status_code=404,
            detail="No rankings found. POST /jobs/{id}/rank to run the pipeline.",
        )

    candidates_db = {c.id: c for c in db.query(Candidate).all()}

    result = []
    for s in scores:
        c = candidates_db.get(s.candidate_id)
        if not c:
            continue
        result.append({
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "title": c.title or "",
            "location": c.location or "",
            "overall_score": s.overall_score,
            "skills_match": s.skills_match,
            "semantic_relevance": s.semantic_relevance,
            "behavioral_signal": s.behavioral_signal,
            "career_trajectory": s.career_trajectory,
            "why_rank": s.why_rank or "",
            "evidence": s.evidence or [],
            "debate": s.debate,
            "rank": s.rank,
            "borderline": s.borderline,
            "compute_path": s.compute_path,
            "verification_status": c.verification_status,
            "review_note": c.review_note,
            "skills": c.skills or [],
            "highlights": c.highlights or [],
            "resume_snippet": c.resume_snippet or "",
        })

    return _build_ranked_response(result, list(candidates_db.values()))


@router.get("/{job_id}/funnel", response_model=list[FunnelStage])
def get_funnel(job_id: int, db: Session = Depends(get_db)):
    """Return funnel stage counts for this job's last pipeline run."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.funnel_counts:
        return job.funnel_counts

    # Default funnel if no pipeline run yet
    return [
        {"label": "Fast Retrieval", "count": 10000, "description": "Keyword + embedding pre-filter"},
        {"label": "Enrichment", "count": 200, "description": "Profile enrichment + deduplication"},
        {"label": "Deep Reasoning", "count": 30, "description": "LLM semantic scoring + sub-scores"},
        {"label": "Ranked & Fairness-Checked", "count": 10, "description": "Final shortlist with bias audit"},
    ]


def _build_ranked_response(ranked: list[dict], candidates_db) -> list[RankedCandidate]:
    """Convert raw dicts to RankedCandidate Pydantic models."""
    result = []
    for cand in ranked:
        debate = None
        if cand.get("debate"):
            d = cand["debate"]
            if isinstance(d, dict) and d.get("pro"):
                debate = DebateTranscript(pro=d["pro"], skeptic=d.get("skeptic", ""))

        result.append(RankedCandidate(
            id=cand["id"],
            name=cand.get("name", ""),
            email=cand.get("email", ""),
            title=cand.get("title") or None,
            location=cand.get("location") or None,
            overallScore=round(float(cand.get("overall_score", 0)), 1),
            scores=SubScores(
                skillsMatch=round(float(cand.get("skills_match", 0)), 1),
                semanticRelevance=round(float(cand.get("semantic_relevance", 0)), 1),
                behavioralSignal=round(float(cand.get("behavioral_signal", 0)), 1),
                careerTrajectory=round(float(cand.get("career_trajectory", 0)), 1),
            ),
            verificationStatus=cand.get("verification_status", "pending"),
            borderline=bool(cand.get("borderline", False)),
            highlights=cand.get("highlights") or [],
            whyRank=cand.get("why_rank") or "",
            evidence=cand.get("evidence") or [],
            debate=debate,
            resumeSnippet=cand.get("resume_snippet") or None,
            reviewNote=cand.get("review_note") or None,
            computePath=cand.get("compute_path") or None,
        ))
    return result
