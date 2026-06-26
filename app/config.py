from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    DATABASE_URL: str = "sqlite:///./talentrank.db"
    UPLOAD_DIR: str = "./uploads"
    QDRANT_PATH: str = "./qdrant_data"
    FAST_MODEL: str = "llama-3.1-8b-instant"
    REASONING_MODEL: str = "llama-3.3-70b-versatile"
    CORS_ORIGINS: list = [
        "http://localhost:5174",
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
