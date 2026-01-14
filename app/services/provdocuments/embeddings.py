from functools import lru_cache
from typing import List

import numpy as np
from openai import OpenAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_chunks(chunks: List[str]):
    """Run embeddings; return numpy array for optional downstream storage."""
    if not chunks:
        return np.array([], dtype=float)
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    client = get_openai_client()
    response = client.embeddings.create(model=settings.EMBED_MODEL, input=chunks)
    vectors = [item.embedding for item in response.data]
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"OpenAI embedding count mismatch: expected {len(chunks)}, got {len(vectors)}"
        )
    embeddings = np.array(vectors, dtype=float)
    return _normalize_embeddings(embeddings)
