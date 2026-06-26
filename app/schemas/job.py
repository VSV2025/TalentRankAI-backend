from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class FunnelStage(BaseModel):
    label: str
    count: int
    description: str


class JobCreate(BaseModel):
    title: str
    description: str


class RankRequest(BaseModel):
    description: Optional[str] = None


class JobOut(BaseModel):
    id: int
    title: str
    description: str
    created_at: datetime

    model_config = {"from_attributes": True}
