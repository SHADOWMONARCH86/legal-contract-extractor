"""
Application settings loaded from environment variables / .env file.
Uses pydantic-settings for type-safe configuration.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    app_name: str = "Legal Contract Entity Extractor"
    app_env: str = "development"  # development | staging | production
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "text"  # text | json

    # NER Model
    model_path: Optional[str] = None  # Path to fine-tuned HuggingFace model or model hub ID
    spacy_model: str = "en_core_web_lg"  # spaCy model name (fallback)

    # OCR
    tesseract_cmd: Optional[str] = None  # Override Tesseract binary path if needed
    ocr_language: str = "eng"  # Tesseract language codes
    ocr_dpi: int = 300

    # MLflow (optional)
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "legal-ner"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
