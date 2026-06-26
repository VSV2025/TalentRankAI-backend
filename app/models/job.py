from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(JSON)
    funnel_counts = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scores = relationship("CandidateScore", back_populates="job")
