"""Local cross-encoder reranker (bge-reranker-v2-m3)."""
from functools import lru_cache

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance score of each text for the query (higher is better)."""
    if not texts:
        return []
    return [float(s) for s in _model().predict([(query, t) for t in texts], batch_size=16)]
