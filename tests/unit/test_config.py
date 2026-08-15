import pytest

from rankfuse.config import RetrieverConfig
from rankfuse.exceptions import ConfigError


def test_config_valid_init():
    config = RetrieverConfig(api_key="test-key")
    assert config.api_key == "test-key"
    assert config.embedder_provider == "gemini"
    assert config.persist_dir == "./rankfuse_index"
    assert config.reranker_type == "cross_encoder"
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.dense_top_k == 20
    assert config.sparse_top_k == 20
    assert config.rerank_top_k == 5
    assert config.rrf_k == 60


def test_config_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RANKFUSE_API_KEY", raising=False)

    with pytest.raises(ConfigError) as exc_info:
        RetrieverConfig()
    assert "API key is missing for provider" in str(exc_info.value)


def test_config_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    config = RetrieverConfig()
    assert config.api_key == "env-gemini-key"


def test_config_openai_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    config = RetrieverConfig(embedder_provider="openai")
    assert config.api_key == "env-openai-key"


def test_config_validation_errors():
    with pytest.raises(ConfigError) as exc_info:
        RetrieverConfig(api_key="key", chunk_size=0)
    assert "chunk_size must be a positive integer" in str(exc_info.value)

    with pytest.raises(ConfigError) as exc_info:
        RetrieverConfig(api_key="key", chunk_overlap=-1)
    assert "chunk_overlap must be non-negative" in str(exc_info.value)

    with pytest.raises(ConfigError) as exc_info:
        RetrieverConfig(api_key="key", chunk_size=100, chunk_overlap=100)
    assert "chunk_overlap must be non-negative and less than chunk_size" in str(
        exc_info.value
    )

    with pytest.raises(ConfigError) as exc_info:
        RetrieverConfig(api_key="key", dense_top_k=0)
    assert "top_k search parameters must be positive integers" in str(exc_info.value)

    with pytest.raises(ConfigError) as exc_info:
        RetrieverConfig(api_key="key", rrf_k=-5)
    assert "rrf_k must be a positive integer" in str(exc_info.value)
