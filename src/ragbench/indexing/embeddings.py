"""Local dense embeddings, same model for every pipeline."""
from functools import lru_cache

from ragbench.config import get_settings

EMBEDDING_DIM = 1024  # bge-m3


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(texts, normalize_embeddings=True, batch_size=16)
    return vectors.tolist()
