import re
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    from .models import candidate, job  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=engine)
    _migrate_add_columns()
    _backfill_phone()


def _backfill_phone():
    """
    Startup backfill: re-extract phone from the original resume file for any
    candidate where phone IS NULL but resume_path IS NOT NULL.
    Safe to run on every startup — skips rows that already have a phone.
    """
    from .services.resume_parser import extract_phones_from_text, extract_text

    def _clean(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        return raw.strip() if len(digits) >= 7 else ""

    from sqlalchemy import text as sql_text
    with engine.connect() as conn:
        rows = conn.execute(
            sql_text("SELECT id, resume_path FROM candidates WHERE phone IS NULL AND resume_path IS NOT NULL")
        ).fetchall()

    if not rows:
        return

    updated = 0
    with engine.connect() as conn:
        for cid, resume_path in rows:
            if not Path(resume_path).exists():
                continue
            try:
                raw_text = extract_text(resume_path)
                phones = extract_phones_from_text(raw_text)
                phone = next((_clean(p) for p in phones if _clean(p)), None)
                if phone:
                    conn.execute(
                        sql_text("UPDATE candidates SET phone = :ph WHERE id = :id"),
                        {"ph": phone, "id": cid},
                    )
                    updated += 1
            except Exception as e:
                logger.warning(f"Phone backfill failed for candidate {cid}: {e}")
        conn.commit()

    if updated:
        logger.info(f"Phone backfill: updated {updated} candidate(s).")


def _migrate_add_columns():
    """Idempotent: add new columns to existing tables without dropping data."""
    from sqlalchemy import text
    new_cols = [
        ("candidate_scores", "graph_fit_score",              "FLOAT DEFAULT 50.0"),
        ("candidate_scores", "skill_breadth_score",          "FLOAT DEFAULT 50.0"),
        ("candidate_scores", "production_evidence",          "FLOAT DEFAULT 0.0"),
        ("candidates",       "phone",                        "VARCHAR(50)"),
        ("candidate_scores", "career_trajectory_detail",     "TEXT"),
        ("candidate_scores", "gaps",                         "TEXT"),
    ]
    with engine.connect() as conn:
        for table, col, col_def in new_cols:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"))
                conn.commit()
            except Exception:
                pass  # column already exists
