from ragbench.ingestion.chunking import chunk_document, count_tokens
from ragbench.ingestion.parser import Document
from ragbench.pipelines.base import RetrievedContext, apply_budget


def test_chunks_respect_size():
    """Recursive chunking splits a long text into several chunks.

    Checks that every chunk is at most 512 tokens (Pipeline 1 size) and that
    chunk ids follow the `<doc_id>:<index>` format.
    """
    doc = Document("d", "One word. " * 2000)
    chunks = chunk_document(doc, size=512, overlap=50)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 512 for c in chunks)
    assert chunks[0].chunk_id == "d:0"


def test_budget_keeps_best_scores_within_limit():
    """The context budget keeps the best scores without exceeding the limit.

    Three contexts of about 100 tokens and a budget of 250: only the two best
    (scores 0.9 and 0.8) must remain, the third does not fit. This rule is
    shared by all 4 pipelines (fair comparison, spec section 3).
    """
    ctxs = [RetrievedContext("word " * 100, s) for s in (0.9, 0.8, 0.7)]
    kept = apply_budget(ctxs, budget=250)
    assert [c.score for c in kept] == [0.9, 0.8]
    assert sum(count_tokens(c.text) for c in kept) <= 250
