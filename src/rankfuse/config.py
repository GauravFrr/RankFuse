import os
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rankfuse.exceptions import ConfigError


class RetrieverConfig(BaseSettings):
    """Configuration settings for the RankFuse Retriever."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedder_provider: str = "gemini"
    api_key: str | None = None
    persist_dir: str = "./rankfuse_index"
    reranker_type: Literal["cross_encoder", "llm_judge", "none"] = "cross_encoder"
    chunk_size: int = 500
    chunk_overlap: int = 50
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    @model_validator(mode="after")
    def validate_config(self) -> "RetrieverConfig":
        if self.chunk_size <= 0:
            raise ConfigError("chunk_size must be a positive integer.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                "chunk_overlap must be non-negative and less than chunk_size."
            )
        if self.dense_top_k <= 0 or self.sparse_top_k <= 0 or self.rerank_top_k <= 0:
            raise ConfigError("top_k search parameters must be positive integers.")
        if self.rrf_k <= 0:
            raise ConfigError("rrf_k must be a positive integer.")
        if self.dense_weight < 0.0 or self.sparse_weight < 0.0:
            raise ConfigError(
                "dense_weight and sparse_weight must be non-negative floats."
            )

        # API key validation
        if not self.api_key:
            provider = self.embedder_provider.lower()
            env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            self.api_key = os.environ.get(env_var) or os.environ.get("RANKFUSE_API_KEY")

        if not self.api_key:
            raise ConfigError(
                f"API key is missing for provider '{self.embedder_provider}'. "
                "Pass 'api_key' to RetrieverConfig or set GEMINI_API_KEY."
            )
        return self
