"""TalentRank AI — FastAPI backend entry point."""
import logging
import threading
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import get_settings
from .database import create_tables
from .routers import candidates, jobs, hackathon, auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _prewarm_embeddings() -> None:
    """Load sentence-transformers + cross-encoder into memory at startup.
    Skipped on Render/production (PREWARM_EMBEDDINGS=false) to stay within free-tier RAM.
    Models load lazily on first pipeline run instead."""
    import os
    if os.getenv("PREWARM_EMBEDDINGS", "true").lower() == "false":
        logger.info("Embedding pre-warm disabled (PREWARM_EMBEDDINGS=false) — lazy load on first use")
        return
    try:
        from .services.embedding import _retriever
        _retriever._get_embed_model()
        _retriever._get_cross_encoder()
        logger.info("Embedding models pre-warmed (sentence-transformers ready)")
    except Exception as e:
        logger.warning(f"Embedding pre-warm skipped ({e}) — models will load on first pipeline run")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Ensure all data directories exist (handles Render /tmp paths and local paths)
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    db_url = settings.DATABASE_URL
    if "sqlite:///" in db_url:
        db_file = db_url.split("sqlite:///")[-1]
        if db_file:
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    create_tables()
    logger.info("Database tables ready")
    if not settings.GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY not set — pipeline will use heuristic fallbacks. "
            "Set it in .env for full LLM-powered scoring."
        )
    else:
        logger.info(f"Groq API ready | fast={settings.FAST_MODEL} | reasoning={settings.REASONING_MODEL}")
    logger.info(f"Recruiter password set (change via RECRUITER_PASSWORD in .env)")
    threading.Thread(target=_prewarm_embeddings, daemon=True, name="embed-prewarm").start()
    yield


settings = get_settings()

app = FastAPI(
    title="TalentRank AI API",
    description="7-layer intelligent candidate ranking pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(candidates.router)
app.include_router(jobs.router)
app.include_router(hackathon.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "TalentRank AI", "v": "tfidf-default"}
