"""Candidate intake endpoints: submit, verify email token, list."""
import os
import uuid
import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..config import get_settings
from ..models.candidate import Candidate, CandidateScore
from ..schemas.candidate import CandidateOut, RankedCandidate, VerificationResult
from ..services.resume_parser import parse_resume, file_hash
from ..services.verification import run_verification

router = APIRouter(prefix="/candidates", tags=["candidates"])
logger = logging.getLogger(__name__)
settings = get_settings()


@router.get("/", response_model=list[CandidateOut])
def list_candidates(db: Session = Depends(get_db)):
    return db.query(Candidate).order_by(Candidate.created_at.desc()).all()


@router.get("/{candidate_id}", response_model=CandidateOut)
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    c = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return c


@router.post("/", status_code=status.HTTP_201_CREATED)
async def submit_candidate(
    name: str = Form(...),
    email: str = Form(...),
    resume: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Accept candidate submission with resume upload.
    Runs verification and returns structured check results.
    """
    # Validate file type
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if resume.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail="Only PDF or DOCX files are accepted.",
        )

    # Save file
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(resume.filename or "resume.pdf").suffix.lower()
    file_name = f"{uuid.uuid4().hex}{ext}"
    file_path = upload_dir / file_name

    content = await resume.read()
    file_path.write_bytes(content)

    # Duplicate detection by email
    existing = db.query(Candidate).filter(
        Candidate.email == email.strip().lower()
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail="A candidate with this email already exists.",
        )

    # Parse resume
    parsed = parse_resume(str(file_path))
    resume_hash = file_hash(str(file_path))

    # Duplicate by resume hash
    hash_existing = db.query(Candidate).filter(
        Candidate.resume_hash == resume_hash
    ).first()
    if hash_existing:
        raise HTTPException(
            status_code=409,
            detail="This resume has already been submitted.",
        )

    # Run verification
    verification = run_verification(
        form_name=name,
        form_email=email,
        resume_path=str(file_path),
        resume_parsed=parsed,
    )

    # Derive verification status
    overall_status = verification["overall_status"]

    # Create DB record
    token = secrets.token_urlsafe(32)
    candidate = Candidate(
        name=name.strip(),
        email=email.strip().lower(),
        title=parsed.get("title"),
        location=parsed.get("location"),
        resume_path=str(file_path),
        resume_text=parsed.get("text", "")[:10000],
        resume_hash=resume_hash,
        skills=parsed.get("skills", []),
        experience_years=parsed.get("experience_years", 0),
        resume_snippet=parsed.get("snippet", ""),
        verification_status=overall_status,
        review_note=next(
            (c["badge"] for c in verification["checks"] if c.get("badge")), None
        ),
        consistency_score=1.0 if overall_status == "verified" else 0.7,
        verification_token=token,
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    logger.info(f"New candidate: {name} <{email}> | status={overall_status}")

    return {
        "candidate_id": candidate.id,
        "checks": verification["checks"],
        "overall_status": overall_status,
    }


@router.get("/verify-email/{token}")
def verify_email(token: str, db: Session = Depends(get_db)):
    """Email confirmation click handler."""
    cand = db.query(Candidate).filter(
        Candidate.verification_token == token
    ).first()
    if not cand:
        raise HTTPException(status_code=404, detail="Invalid or expired token.")
    cand.email_confirmed = True
    if cand.verification_status == "pending":
        cand.verification_status = "verified"
    db.commit()
    return {"message": "Email confirmed. Thank you!", "candidate_id": cand.id}
