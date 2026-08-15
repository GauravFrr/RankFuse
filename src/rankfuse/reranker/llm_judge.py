from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from rankfuse.exceptions import RerankerError
from rankfuse.reranker.base import Reranker, RetrievalResult


class RelevanceScore(BaseModel):
    score: float = Field(
        description="Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant)"
    )


class LLMJudgeReranker(Reranker):
    """Reranker implementation using Gemini as an LLM relevance judge."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RerankerError(f"Failed to initialize Google GenAI Client: {e}")

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank a list of candidate documents using Gemini as an LLM judge.

        Args:
            query: The search query.
            candidates: List of RetrievalResult candidates to rerank.
            top_k: Number of top reranked candidates to return.

        Returns:
            A list of top_k RetrievalResult documents, sorted by descending relevance.
        """
        if not candidates:
            return []

        try:
            reranked = []
            for c in candidates:
                prompt = (
                    "Analyze how relevant the document is to the query.\n"
                    "Score from 0.0 (irrelevant) to 1.0 (highly relevant).\n\n"
                    f"Query: {query}\n"
                    f"Document: {c.text}"
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RelevanceScore,
                        temperature=0.0,
                    ),
                )

                if not response or not response.text:
                    raise RerankerError("LLM judge returned an empty response.")

                score_data = RelevanceScore.model_validate_json(response.text)
                score = score_data.score

                reranked.append(
                    RetrievalResult(
                        doc_id=c.doc_id,
                        text=c.text,
                        score=float(score),
                        metadata=c.metadata,
                    )
                )

            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_k]

        except Exception as e:
            if isinstance(e, RerankerError):
                raise e
            raise RerankerError(f"LLM judge reranking failed: {e}")
