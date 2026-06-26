"""Hackathon submission API — offline 7-layer pipeline with no LLM calls."""
import io
import json
import logging
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..services.offline_pipeline import (
    HACKATHON_JD,
    run_pipeline,
    results_to_csv,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/hackathon", tags=["hackathon"])

# In-memory job store  { job_id -> job_dict }
_jobs: dict = {}


def _make_job() -> dict:
    return {
        "status": "pending",
        "progress": {
            "current_layer": "L1 JD Parse",
            "layer_index": 0,
            "processed": 0,
            "total": 0,
            "message": "Starting pipeline…",
        },
        "results": None,
        "error": None,
    }


def _run_job(job_id: str, candidates: list) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def progress_cb(p: dict):
        job["progress"] = p

    try:
        output = run_pipeline(candidates, top_k=min(2000, len(candidates)), progress_cb=progress_cb)
        job["status"] = "complete"
        job["results"] = output
    except Exception as exc:
        logger.exception("Pipeline job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jd")
def get_jd():
    """Return the pre-parsed hackathon job description."""
    return HACKATHON_JD


@router.post("/rank")
async def start_ranking(
    file: Optional[UploadFile] = File(default=None),
    use_sample: bool = Form(default=False),
    sample_path: Optional[str] = Form(default=None),
):
    """
    Start the 7-layer offline ranking pipeline.

    Accepts one of:
    - file upload (.jsonl or .json array of candidate objects)
    - use_sample=true  → uses the 50-candidate sample from the challenge bundle
    - sample_path      → absolute path to a local candidates.jsonl on the server
    """
    candidates: list = []

    if file is not None:
        raw = await file.read()
        text = raw.decode("utf-8")
        # Support both JSONL and JSON array
        if text.strip().startswith("["):
            candidates = json.loads(text)
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    elif sample_path:
        import os
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=400, detail=f"File not found: {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                candidates = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))

    elif use_sample:
        # Serve from the bundled sample in the challenge directory
        import os
        default_paths = [
            r"C:\Users\drgsr\Downloads\[PUB] India_runs_data_and_ai_challenge"
            r"\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\sample_candidates.json",
        ]
        loaded = False
        for p in default_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    candidates = json.load(f)
                loaded = True
                break
        if not loaded:
            raise HTTPException(
                status_code=404,
                detail="Sample file not found. Please upload a candidates file instead.",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a file upload, use_sample=true, or sample_path.",
        )

    if not candidates:
        raise HTTPException(status_code=400, detail="No valid candidates found in input.")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = _make_job()
    _jobs[job_id]["progress"]["total"] = len(candidates)

    t = threading.Thread(target=_run_job, args=(job_id, candidates), daemon=True)
    t.start()

    return {"job_id": job_id, "total_candidates": len(candidates), "status": "running"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll pipeline status and progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "has_results": job["results"] is not None,
        "error": job["error"],
    }


@router.get("/results/{job_id}")
def get_results(job_id: str):
    """Return full results (top-100 ranked candidates + funnel counts)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] == "running":
        raise HTTPException(status_code=202, detail="Pipeline still running.")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    return job["results"]


@router.get("/download/{job_id}")
def download_csv(job_id: str):
    """Download the submission.csv for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete yet.")

    csv_text = results_to_csv(job["results"]["results"])
    return StreamingResponse(
        io.BytesIO(csv_text.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=submission.csv"},
    )
