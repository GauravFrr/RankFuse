class RankFuseError(Exception):
    """Base exception for all RankFuse errors."""

    pass


class ConfigError(RankFuseError):
    """Raised when there is an invalid or missing configuration."""

    pass


class EmbeddingError(RankFuseError):
    """Raised when an embedding API or model call fails."""

    pass


class StoreError(RankFuseError):
    """Raised when a vector store read or write operation fails."""

    pass


class RerankerError(RankFuseError):
    """Raised when a reranking model or API call fails."""

    pass
