"""Pipeline 2: Advanced RAG.

Sentence-window indexing, hybrid search (dense + BM25 sparse, RRF fusion), then
cross-encoder reranking down to the top 3 windows. Truncation to the shared
context budget happens in the base class.
"""
from ragbench.indexing.hybrid_store import HybridStore
from ragbench.indexing.reranker import rerank_scores
from ragbench.ingestion.parser import Document
from ragbench.ingestion.sentences import chunk_sentences
from ragbench.llm.base import LLMClient
from ragbench.pipelines.base import RAGPipeline, RetrievedContext


class AdvancedRAG(RAGPipeline):
    name = "advanced"
    WINDOW = 3        # sentences on each side of the matched sentence
    CANDIDATES = 20   # fused candidates sent to the reranker (and prefetch per retriever)
    TOP_K = 3         # windows kept after reranking

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.store = HybridStore("advanced")

    def index(self, documents: list[Document]) -> None:
        self.store.reset()
        for doc in documents:
            chunks = chunk_sentences(doc, window=self.WINDOW)
            self.store.add(
                [c.chunk_id for c in chunks],
                [c.sentence for c in chunks],
                [{**c.metadata, "text": c.window, "sentence": c.sentence} for c in chunks],
            )

    def retrieve(self, query: str) -> list[RetrievedContext]:
        hits = self.store.search(query, limit=self.CANDIDATES, prefetch=self.CANDIDATES)
        # Neighbouring sentences produce near-identical windows: keep one per window text.
        unique = list({p["text"]: p for p, _ in hits}.values())
        scores = rerank_scores(query, [p["text"] for p in unique])
        ranked = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)[: self.TOP_K]
        return [
            RetrievedContext(
                text=p["text"],
                score=score,
                metadata={k: v for k, v in p.items() if k != "text"},
            )
            for p, score in ranked
        ]
