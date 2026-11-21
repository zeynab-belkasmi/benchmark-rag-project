"""Single LLM interface: lets us add a backend (e.g. Ollama) without touching the pipelines."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage
    ttft_ms: float | None = None  # time to first token (streaming)
    total_ms: float = 0.0


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResult:
        """Blocking call; returns the text, token usage and latency."""

    @abstractmethod
    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream of text chunks."""
