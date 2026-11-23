"""Sentence splitting with a context window (Pipeline 2, sentence-window).

Each sentence is indexed on its own (precise matching); the text handed to the
reranker and the LLM is the sentence plus its neighbours (`window` sentences on
each side) from the same document.
"""
import re
from dataclasses import dataclass

from ragbench.ingestion.parser import Document

# Sentence end followed by a capital letter, digit, bracket or Markdown marker.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\*#|\"'])")
MIN_SENTENCE_CHARS = 40  # shorter pieces (headings, fragments) are merged with the next one


@dataclass
class SentenceChunk:
    chunk_id: str
    doc_id: str
    sentence: str
    window: str
    metadata: dict


def split_sentences(text: str) -> list[str]:
    """Split Markdown into sentences: lines first (tables, headings), then sentence ends."""
    pieces: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            pieces.extend(p.strip() for p in _SENTENCE_END.split(line) if p.strip())
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry} {piece}".strip()
        if len(piece) < MIN_SENTENCE_CHARS:
            carry = piece
        else:
            merged.append(piece)
            carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def chunk_sentences(doc: Document, window: int = 3) -> list[SentenceChunk]:
    sentences = split_sentences(doc.text)
    chunks = []
    for i, sentence in enumerate(sentences):
        lo, hi = max(0, i - window), min(len(sentences), i + window + 1)
        chunks.append(
            SentenceChunk(
                chunk_id=f"{doc.doc_id}:s{i}",
                doc_id=doc.doc_id,
                sentence=sentence,
                window=" ".join(sentences[lo:hi]),
                metadata={**doc.metadata, "sentence_index": i},
            )
        )
    return chunks
