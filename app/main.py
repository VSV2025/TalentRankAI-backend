"""TalentRank AI — FastAPI backend entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import get_settings
from .database import create_tables
from .routers import candidates, jobs, hackathon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    create_tables()
    logger.info("Database tables ready")
    if not settings.GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY not set — pipeline will use heuristic fallbacks. "
            "Set it in .env for full LLM-powered scoring."
        )
    else:
        logger.info(f"Groq API ready | fast={settings.FAST_MODEL} | reasoning={settings.REASONING_MODEL}")
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
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates.router)
app.include_router(jobs.router)
app.include_router(hackathon.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "TalentRank AI"}
