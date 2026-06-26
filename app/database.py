from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import get_settings

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


def _migrate_add_columns():
    """Idempotent: add new columns to existing tables without dropping data."""
    from sqlalchemy import text
    new_cols = [
        ("candidate_scores", "graph_fit_score",    "FLOAT DEFAULT 50.0"),
        ("candidate_scores", "skill_breadth_score", "FLOAT DEFAULT 50.0"),
    ]
    with engine.connect() as conn:
        for table, col, col_def in new_cols:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"))
                conn.commit()
            except Exception:
                pass  # column already exists
