from ragbench.llm.base import LLMClient, LLMResult, LLMUsage
from ragbench.pipelines.base import SYSTEM_PROMPT, RAGPipeline, RetrievedContext


class FakeLLM(LLMClient):
    def __init__(self):
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return LLMResult("fake answer", LLMUsage(10, 3), ttft_ms=5.0)

    def stream(self, system, user):
        yield "fake answer"


class FakePipeline(RAGPipeline):
    name = "fake"

    def __init__(self, llm, contexts):
        super().__init__(llm)
        self._contexts = contexts

    def index(self, documents):
        pass

    def retrieve(self, query):
        return self._contexts


def test_run_builds_prompt_from_budgeted_contexts():
    """run() numbers the kept contexts by score and reports the LLM usage.

    The lowest-scored context (about 3,000 tokens) exceeds the 2,048-token
    budget and must not reach the prompt. The system prompt is the shared one.
    """
    llm = FakeLLM()
    contexts = [
        RetrievedContext("low " * 3000, 0.1),
        RetrievedContext("best passage", 0.9),
        RetrievedContext("second passage", 0.8),
    ]
    result = FakePipeline(llm, contexts).run("What?")

    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[1] best passage" in user and "[2] second passage" in user
    assert "low low" not in user
    assert user.endswith("Question: What?")
    assert [c.score for c in result.contexts] == [0.9, 0.8]
    assert (result.answer, result.input_tokens, result.output_tokens) == ("fake answer", 10, 3)
    assert result.ttft_ms == 5.0
    assert result.total_ms >= result.retrieval_ms >= 0
