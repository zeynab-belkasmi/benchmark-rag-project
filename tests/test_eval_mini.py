import pytest

from ragbench.eval import mini
from ragbench.llm.base import LLMResult, LLMUsage


class FakeJudge:
    def __init__(self, text):
        self.text = text

    def generate(self, system, user):
        return LLMResult(self.text, LLMUsage())


def _result(*contexts):
    return {"contexts": [{"text": t, "metadata": {"source": s}} for s, t in contexts]}


def test_retrieval_scores_measure_doc_hit_and_quote_coverage():
    """doc_hit counts reference docs retrieved; recall is the token coverage of the quotes.

    B01 is retrieved and contains half of its quote tokens; A06 is not retrieved.
    """
    res = _result(("B01_2309.15217.pdf", "alpha beta other"))
    passages = [{"doc": "B01", "quote": "alpha beta gamma delta"},
                {"doc": "A06", "quote": "never retrieved"}]
    scores = mini.retrieval_scores(res, passages)
    assert scores["doc_hit"] == 0.5
    assert scores["context_recall"] == pytest.approx((0.5 + 0.0) / 2)


def test_retrieval_scores_are_none_without_reference_passages():
    """no_answer questions have no passages: no retrieval score, not 0."""
    assert mini.retrieval_scores(_result(), []) == {"doc_hit": None, "context_recall": None}


def test_judge_parses_json_embedded_in_text():
    """The judge reply may be wrapped in prose; the JSON object is extracted."""
    q = {"type": "factual", "question": "Q?", "reference_answer": "R"}
    out = mini.judge(FakeJudge('Sure: {"verdict": "partial", "reason": "half"}'), q, "A")
    assert out == {"verdict": "partial", "reason": "half"}


@pytest.mark.parametrize("bad", ["no json here", '{"verdict": "maybe"}', '{"reason": "x"}'])
def test_judge_failure_gives_none_verdict(bad):
    """Unparseable or out-of-vocabulary output yields verdict None (excluded, never 0)."""
    q = {"type": "no_answer", "question": "Q?", "reference_answer": ""}
    assert mini.judge(FakeJudge(bad), q, "A")["verdict"] is None
