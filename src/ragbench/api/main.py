"""Minimal FastAPI app (week 1): health, ingestion, query on Pipeline 1."""
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from ragbench.ingestion.parser import parse_file
from ragbench.llm.anthropic_client import AnthropicClient
from ragbench.pipelines.naive import NaiveRAG

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(title="RAG-BenchLab")
_pipelines: dict[str, NaiveRAG] = {}


def _get(name: str) -> NaiveRAG:
    if name not in _pipelines:
        _pipelines[name] = NaiveRAG(AnthropicClient())
    return _pipelines[name]


class Query(BaseModel):
    question: str
    pipeline: str = "naive"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile]) -> dict:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in files:
        dest = UPLOAD_DIR / Path(f.filename).name
        dest.write_bytes(await f.read())
        docs.append(parse_file(dest))
    _get("naive").index(docs)
    return {"indexed_documents": len(docs)}


@app.post("/query")
def query(q: Query) -> dict:
    return asdict(_get(q.pipeline).run(q.question))
