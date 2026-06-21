"""Pydantic models for /v1/embeddings."""

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]

