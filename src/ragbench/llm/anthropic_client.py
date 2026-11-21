"""Claude API backend. Streaming is used to measure TTFT."""
import time
from collections.abc import Iterator

import anthropic

from ragbench.config import get_settings
from ragbench.llm.base import LLMClient, LLMResult, LLMUsage


class AnthropicClient(LLMClient):
    def __init__(self, model: str | None = None):
        s = get_settings()
        self._s = s
        self.model = model or s.generation_model
        self._client = anthropic.Anthropic(api_key=s.anthropic_api_key or None)

    def _kwargs(self, system: str, user: str) -> dict:
        return dict(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self._s.max_output_tokens,
            temperature=self._s.temperature,
        )

    def generate(self, system: str, user: str) -> LLMResult:
        start = time.perf_counter()
        ttft = None
        parts: list[str] = []
        with self._client.messages.stream(**self._kwargs(system, user)) as stream:
            for chunk in stream.text_stream:
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                parts.append(chunk)
            final = stream.get_final_message()
        total = (time.perf_counter() - start) * 1000
        usage = LLMUsage(final.usage.input_tokens, final.usage.output_tokens)
        return LLMResult("".join(parts), usage, ttft_ms=ttft, total_ms=total)

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self._client.messages.stream(**self._kwargs(system, user)) as stream:
            yield from stream.text_stream
