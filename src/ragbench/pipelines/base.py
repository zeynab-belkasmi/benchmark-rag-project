"""Common pipeline contract.

Generation, the prompt and the context-budget truncation live HERE, not in the
subclasses: only the indexing/retrieval strategy varies (fair-comparison rule,
spec section 3).
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragbench.config import get_settings
from ragbench.ingestion.chunking import count_tokens
from ragbench.ingestion.parser import Document
from ragbench.llm.base import LLMClient

SYSTEM_PROMPT = (
    "Answer only from the provided context. "
    "If the context does not contain the answer, say explicitly that you do not know. "
    "Be concise and answer in the language of the question."
)


@dataclass
class RetrievedContext:
    text: str
    score: float
    kind: str = "chunk"  # chunk | community_summary | node
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    pipeline: str
    answer: str
    contexts: list[RetrievedContext]  # contexts actually injected (after budget)
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    ttft_ms: float | None
    input_tokens: int
    output_tokens: int
    trace: list[dict] = field(default_factory=list)


def apply_budget(contexts: list[RetrievedContext], budget: int) -> list[RetrievedContext]:
    """Keep contexts by descending score until the token budget is full."""
    kept, used = [], 0
    for ctx in sorted(contexts, key=lambda c: c.score, reverse=True):
        n = count_tokens(ctx.text)
        if used + n > budget:
            continue
        kept.append(ctx)
        used += n
    return kept


class RAGPipeline(ABC):
    name: str

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.settings = get_settings()

    @abstractmethod
    def index(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedContext]: ...

    def run(self, query: str) -> PipelineResult:
        t0 = time.perf_counter()
        retrieved = self.retrieve(query)
        t1 = time.perf_counter()

        contexts = apply_budget(retrieved, self.settings.context_token_budget)
        block = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(contexts))
        user = f"Context:\n{block}\n\nQuestion: {query}"
        res = self.llm.generate(SYSTEM_PROMPT, user)
        t2 = time.perf_counter()

        return PipelineResult(
            pipeline=self.name,
            answer=res.text,
            contexts=contexts,
            retrieval_ms=(t1 - t0) * 1000,
            generation_ms=(t2 - t1) * 1000,
            total_ms=(t2 - t0) * 1000,
            ttft_ms=res.ttft_ms,
            input_tokens=res.usage.input_tokens,
            output_tokens=res.usage.output_tokens,
        )
