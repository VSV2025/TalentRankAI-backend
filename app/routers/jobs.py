"""Job creation and ranking endpoints."""
import logging
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models.candidate import Candidate, CandidateScore
from ..models.job import Job
from ..schemas.candidate import RankedCandidate, SubScores, DebateTranscript
from ..schemas.job import JobCreate, JobOut, FunnelStage, RankRequest
from ..services.pipeline import run_pipeline

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)

# In-memory job store: task_id -> {status, layer, progress, error}
_ranking_jobs: dict = {}


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


@router.post("/{job_id}/rank")
def rank_candidates(
    job_id: int,
    body: Optional[RankRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Launch the 7-layer pipeline in the background.
    Returns {"task_id": "..."} immediately — poll /rank/status/{task_id} for progress.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if body and body.description:
        job.description = body.description
        db.commit()

    candidates_db = db.query(Candidate).all()
    if not candidates_db:
        raise HTTPException(status_code=400, detail="No candidates in the database. Run seed first.")

    jd = job.description
    candidate_dicts = [
        {
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "title": c.title or "",
            "location": c.location or "",
            "skills": c.skills or [],
            "resume_text": c.resume_text or "",
            "experience_years": c.experience_years if c.experience_years is not None else 0.0,
            "highlights": c.highlights or [],
            "resume_snippet": c.resume_snippet or "",
            "verification_status": c.verification_status,
            "review_note": c.review_note,
        }
        for c in candidates_db
    ]

    task_id = str(uuid.uuid4())
    _ranking_jobs[task_id] = {"status": "running", "layer": "Starting pipeline…", "progress": 0, "error": None}

    def _run_pipeline_bg():
        db2 = SessionLocal()
        try:
            def progress_cb(layer: str, pct: int) -> None:
                _ranking_jobs[task_id]["layer"] = layer
                _ranking_jobs[task_id]["progress"] = pct

            result = run_pipeline(jd, candidate_dicts, progress_cb=progress_cb)

            ranked = result["ranked"]
            funnel_counts = result["funnel_counts"]

            db2.query(CandidateScore).filter(CandidateScore.job_id == job_id).delete()
            for cand in ranked:
                score_row = CandidateScore(
                    candidate_id=cand["id"],
                    job_id=job_id,
                    overall_score=cand.get("overall_score", 0),
                    skills_match=cand.get("skills_match", 0),
                    semantic_relevance=cand.get("semantic_relevance", 0),
                    behavioral_signal=cand.get("behavioral_signal", 0),
                    career_trajectory=cand.get("career_trajectory", 0),
                    production_evidence=cand.get("production_evidence", 0.0),
                    why_rank=cand.get("why_rank", ""),
                    evidence=cand.get("evidence", []),
                    debate=cand.get("debate"),
                    rank=cand.get("rank", 0),
                    borderline=cand.get("borderline", False),
                    compute_path=cand.get("compute_path", "heuristic"),
                    pipeline_timings=result.get("timings"),
                    graph_fit_score=cand.get("graph_fit_score", 50.0),
                    skill_breadth_score=cand.get("skill_breadth_score", 50.0),
                    career_trajectory_detail=cand.get("career_trajectory_detail"),
                    gaps=cand.get("gaps"),
                )
                db2.add(score_row)

                # Persist LLM-generated highlights back to Candidate so the card displays them
                if cand.get("highlights"):
                    cand_row = db2.query(Candidate).filter(Candidate.id == cand["id"]).first()
                    if cand_row:
                        cand_row.highlights = cand["highlights"]

            job2 = db2.query(Job).filter(Job.id == job_id).first()
            if job2:
                job2.funnel_counts = funnel_counts
                job2.requirements = result.get("requirements")
            db2.commit()

            _ranking_jobs[task_id]["status"] = "done"
            _ranking_jobs[task_id]["progress"] = 100
            logger.info(f"[bg] Job {job_id} ranked {len(ranked)} candidates (task={task_id[:8]})")

        except Exception as e:
            logger.error(f"[bg] Pipeline failed (task={task_id[:8]}): {e}", exc_info=True)
            _ranking_jobs[task_id]["status"] = "error"
            _ranking_jobs[task_id]["error"] = str(e)
        finally:
            db2.close()

    threading.Thread(target=_run_pipeline_bg, daemon=True, name=f"rank-{task_id[:8]}").start()
    return {"task_id": task_id}


@router.get("/{job_id}/rank/status/{task_id}")
def get_rank_status(job_id: int, task_id: str):
    """Poll ranking progress. status: 'running' | 'done' | 'error'"""
    job_state = _ranking_jobs.get(task_id)
    if not job_state:
        raise HTTPException(status_code=404, detail="Task not found")
    return job_state


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
        return []

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
            "production_evidence": s.production_evidence,
            "why_rank": s.why_rank or "",
            "evidence": s.evidence or [],
            "debate": s.debate,
            "rank": s.rank,
            "borderline": s.borderline,
            "compute_path": s.compute_path,
            "graph_fit_score": s.graph_fit_score,
            "skill_breadth_score": s.skill_breadth_score,
            "career_trajectory_detail": s.career_trajectory_detail,
            "gaps": s.gaps,
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

    return [
        {"label": "Fast Retrieval", "count": 10000, "description": "Keyword + embedding pre-filter"},
        {"label": "Enrichment", "count": 200, "description": "Profile enrichment + deduplication"},
        {"label": "Deep Reasoning", "count": 30, "description": "LLM semantic scoring + sub-scores"},
        {"label": "Ranked & Fairness-Checked", "count": 10, "description": "Final shortlist with bias audit"},
    ]


def _build_ranked_response(ranked: list[dict], candidates_db) -> list[RankedCandidate]:
    result = []
    for cand in ranked:
        debate = None
        if cand.get("debate"):
            d = cand["debate"]
            if isinstance(d, dict) and d.get("pro"):
                debate = DebateTranscript(pro=d["pro"], skeptic=d.get("skeptic", ""))

        result.append(RankedCandidate(
            id=cand["id"],
            rank=cand.get("rank"),
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
                productionEvidence=round(float(cand.get("production_evidence", 0)), 1),
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
            graphFitScore=round(float(cand.get("graph_fit_score", 50)), 1) if cand.get("graph_fit_score") is not None else None,
            skillBreadthScore=round(float(cand.get("skill_breadth_score", 50)), 1) if cand.get("skill_breadth_score") is not None else None,
            careerTrajectoryDetail=cand.get("career_trajectory_detail") or None,
            gaps=cand.get("gaps") or None,
        ))
    return result
