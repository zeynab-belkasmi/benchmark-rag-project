from ragbench.ingestion.parser import Document
from ragbench.ingestion.sentences import chunk_sentences, split_sentences
from ragbench.pipelines import advanced
from ragbench.pipelines.advanced import AdvancedRAG


def test_split_sentences_cuts_lines_and_sentence_ends():
    """Sentences end at '. ' before a capital; table rows and headings are kept apart."""
    text = (
        "The optimal boost was found to be 0.3 overall. It reached the best score here.\n"
        "| a | b |\n"
    )
    out = split_sentences(text)
    assert out[0] == "The optimal boost was found to be 0.3 overall."
    assert out[1].startswith("It reached the best score here.")


def test_short_fragments_are_merged_not_dropped():
    """A short heading is glued to the following sentence; nothing is lost."""
    out = split_sentences("## Intro\nThis sentence is long enough to stand on its own.")
    assert out == ["## Intro This sentence is long enough to stand on its own."]


def test_window_contains_neighbours_within_document_bounds():
    """Window of 1: the first sentence sees itself and the next one only."""
    sentences = [f"Sentence number {i} is a reasonably long sentence." for i in range(4)]
    chunks = chunk_sentences(Document("d", "\n".join(sentences)), window=1)
    assert chunks[0].window == " ".join(sentences[0:2])
    assert chunks[2].window == " ".join(sentences[1:4])
    assert chunks[2].chunk_id == "d:s2" and chunks[2].metadata["sentence_index"] == 2


class FakeStore:
    def search(self, query, limit, prefetch):
        hit = lambda t: ({"text": t, "chunk_id": t, "source": "x.pdf"}, 0.01)
        # "dup" is returned twice (neighbouring sentences share a window)
        return [hit("a"), hit("dup"), hit("b"), hit("dup"), hit("c")]


def test_retrieve_dedupes_windows_reranks_and_keeps_top3(monkeypatch):
    """Duplicated windows are reranked once; the 3 best reranker scores are returned."""
    rerank = {"a": 0.1, "dup": 0.9, "b": 0.5, "c": 0.7}
    seen = []

    def fake_rerank(query, texts):
        seen.extend(texts)
        return [rerank[t] for t in texts]

    monkeypatch.setattr(advanced, "rerank_scores", fake_rerank)
    pipe = AdvancedRAG.__new__(AdvancedRAG)
    pipe.store = FakeStore()
    out = pipe.retrieve("q")
    assert sorted(seen) == ["a", "b", "c", "dup"]
    assert [c.text for c in out] == ["dup", "c", "b"]
    assert [c.score for c in out] == [0.9, 0.7, 0.5]
    assert out[0].metadata["source"] == "x.pdf"
