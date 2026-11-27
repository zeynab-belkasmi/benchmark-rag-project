"""Local cross-encoder reranker (model set by the `RERANKER_MODEL` setting)."""
from functools import lru_cache

from ragbench.config import get_settings

MAX_LENGTH = 512  # query + window; P2 windows stay below this (measured max ~510 tokens)


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    model = CrossEncoder(settings.reranker_model, max_length=MAX_LENGTH)
    if settings.reranker_int8:
        # Dynamic int8 quantization of the linear layers: ~2.4x faster on CPU for
        # bge-reranker-v2-m3. Must be in place: a quantized copy breaks CrossEncoder.
        import torch

        torch.quantization.quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance score of each text for the query (higher is better)."""
    if not texts:
        return []
    return [float(s) for s in _model().predict([(query, t) for t in texts], batch_size=16)]
