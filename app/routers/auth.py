"""Recruiter authentication endpoint."""
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


class PasswordCheck(BaseModel):
    password: str


@router.post("/recruiter")
def check_recruiter_password(body: PasswordCheck):
    """Verify recruiter access password. Returns 401 on mismatch."""
    settings = get_settings()
    if body.password != settings.RECRUITER_PASSWORD:
        logger.warning("[auth] Failed recruiter login attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    logger.info("[auth] Recruiter authenticated successfully")
    return {"authenticated": True}
