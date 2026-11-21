"""Pipeline 1: Naive RAG (baseline). 512/50 chunks, dense cosine, top-k = 4."""
from ragbench.indexing.qdrant_store import QdrantStore
from ragbench.ingestion.chunking import chunk_document
from ragbench.ingestion.parser import Document
from ragbench.llm.base import LLMClient
from ragbench.pipelines.base import RAGPipeline, RetrievedContext


class NaiveRAG(RAGPipeline):
    name = "naive"
    TOP_K = 4

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.store = QdrantStore("naive")

    def index(self, documents: list[Document]) -> None:
        self.store.reset()
        for doc in documents:
            chunks = chunk_document(doc, size=512, overlap=50)
            self.store.add(
                [c.chunk_id for c in chunks],
                [c.text for c in chunks],
                [c.metadata for c in chunks],
            )

    def retrieve(self, query: str) -> list[RetrievedContext]:
        hits = self.store.search(query, self.TOP_K)
        return [
            RetrievedContext(
                text=p["text"],
                score=score,
                metadata={k: v for k, v in p.items() if k != "text"},
            )
            for p, score in hits
        ]
