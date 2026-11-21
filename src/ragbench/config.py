"""Central configuration (environment variables / .env)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-5-5"
    extraction_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-opus-5-5"

    embedding_model: str = "BAAI/bge-m3"
    qdrant_url: str = "http://localhost:6333"

    # Controlled variables, identical for all pipelines (spec section 3)
    context_token_budget: int = 2048
    max_output_tokens: int = 1024
    temperature: float = 0.0

    max_budget_eur: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
