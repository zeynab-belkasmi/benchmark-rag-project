"""Hybrid Qdrant collection: dense bge-m3 vector + explicit BM25 sparse vector, fused with RRF.

The sparse vectors are generated here with fastembed (`Qdrant/bm25`) and stored as
Qdrant sparse vectors; the IDF part of BM25 is computed by Qdrant (`Modifier.IDF`).
"""
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ragbench.config import get_settings
from ragbench.indexing.embeddings import EMBEDDING_DIM, embed

DENSE, SPARSE = "dense", "bm25"


@lru_cache
def _bm25():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding("Qdrant/bm25")


def sparse_docs(texts: list[str]) -> list[SparseVector]:
    return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in _bm25().embed(texts)]


def sparse_query(text: str) -> SparseVector:
    e = next(iter(_bm25().query_embed(text)))
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


class HybridStore:
    def __init__(self, collection: str):
        self.collection = collection
        self.client = QdrantClient(url=get_settings().qdrant_url)

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
        )

    def add(self, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        """Index `texts` (dense + sparse); `payloads` carry the rest (incl. the window text)."""
        dense, sparse = embed(texts), sparse_docs(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),
                vector={DENSE: d, SPARSE: s},
                payload={"chunk_id": cid, **meta},
            )
            for cid, meta, d, s in zip(ids, payloads, dense, sparse)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query: str, limit: int, prefetch: int) -> list[tuple[dict, float]]:
        """Dense and sparse top-`prefetch` lists fused with RRF; returns the top `limit`."""
        res = self.client.query_points(
            self.collection,
            prefetch=[
                Prefetch(query=embed([query])[0], using=DENSE, limit=prefetch),
                Prefetch(query=sparse_query(query), using=SPARSE, limit=prefetch),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
        return [(p.payload, p.score) for p in res.points]
