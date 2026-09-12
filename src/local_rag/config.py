"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Chat / reasoning LLM
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "local-chat"

    # Text embeddings
    embed_base_url: str = "http://127.0.0.1:8081/v1"
    embed_model: str = "local-embed"

    # Vision-language model (captioning)
    vlm_base_url: str = "http://127.0.0.1:8082/v1"
    vlm_model: str = "local-vlm"

    openai_api_key: str = "not-a-real-key"

    # ColPali visual retrieval
    colpali_model: str = "vidore/colqwen2-v1.0"
    colpali_device: str = "mps"

    # Storage
    data_dir: Path = Path("./data")
    storage_dir: Path = Path("./storage")

    # Retrieval knobs
    text_top_k: int = 5
    visual_top_k: int = 3
    max_agent_iterations: int = 3

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def colpali_dir(self) -> Path:
        return self.storage_dir / "colpali"

    @property
    def page_image_dir(self) -> Path:
        return self.storage_dir / "page_images"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.storage_dir,
            self.chroma_dir,
            self.colpali_dir,
            self.page_image_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
