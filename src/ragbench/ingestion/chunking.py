"""Fixed-size recursive chunking (Pipeline 1: 512 tokens, overlap 50).

Tokens are counted with tiktoken (cl100k_base), an approximation shared with the
context budget so that all pipelines measure size the same way.
"""
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbench.ingestion.parser import Document

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def chunk_document(doc: Document, size: int = 512, overlap: int = 50) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=size, chunk_overlap=overlap
    )
    pieces = splitter.split_text(doc.text)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}:{i}",
            doc_id=doc.doc_id,
            text=p,
            metadata={**doc.metadata, "chunk_index": i},
        )
        for i, p in enumerate(pieces)
    ]
