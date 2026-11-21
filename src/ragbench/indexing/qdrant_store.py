"""Qdrant access: one collection per pipeline, cosine similarity."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ragbench.config import get_settings
from ragbench.indexing.embeddings import EMBEDDING_DIM, embed


class QdrantStore:
    def __init__(self, collection: str):
        self.collection = collection
        self.client = QdrantClient(url=get_settings().qdrant_url)

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    def add(self, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        vectors = embed(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),
                vector=vec,
                payload={"chunk_id": cid, "text": txt, **meta},
            )
            for cid, txt, meta, vec in zip(ids, texts, payloads, vectors)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        vec = embed([query])[0]
        res = self.client.query_points(self.collection, query=vec, limit=top_k)
        return [(p.payload, p.score) for p in res.points]
