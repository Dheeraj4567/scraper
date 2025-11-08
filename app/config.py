from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    brave_api_key: str
    brave_search_endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    llm_model_path: Path = Path("./models/llava-v1.6-mistral-7b.Q4_K_M.gguf")
    llm_projector_path: Optional[Path] = Path("./models/llava-v1.6-mistral-7b-mmproj.gguf")
    llm_context_window: int = 8192
    llm_gpu_layers: int = 33
    llm_threads: int = 8
    llm_batch_size: int = 512
    llm_temperature: float = 0.2
    llm_top_p: float = 0.95

    enable_metal_acceleration: bool = True

    brave_result_count: int = 6
    brave_safe_search: str = "moderate"

    max_concurrent_fetches: int = 4
    fetch_timeout_seconds: float = 12.0
    user_agent: str = (
        "ScarperBot/0.1 (+https://github.com/cto-dot-new/scarper; contact=ops@cto.new)"
    )

    summary_max_tokens: int = 256
    summary_chunk_size: int = 1200

    final_answer_max_tokens: int = 768

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SCARPER_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
