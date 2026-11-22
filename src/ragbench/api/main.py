"""Minimal FastAPI app (week 1): health, ingestion jobs, query on Pipeline 1.

Ingestion runs in a background job so that /health and /query stay responsive
while parsing and embedding (minutes on CPU). Note: indexing replaces the whole
collection, so ingest the full corpus in a single job.
"""
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from ragbench.ingestion.parser import parse_file
from ragbench.llm.anthropic_client import AnthropicClient
from ragbench.pipelines.naive import NaiveRAG

UPLOAD_DIR = Path("data/uploads")
CORPUS_DIR = Path("corpus/raw")

app = FastAPI(title="RAG-BenchLab")
_pipelines: dict[str, NaiveRAG] = {}
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


def _get(name: str) -> NaiveRAG:
    if name not in _pipelines:
        _pipelines[name] = NaiveRAG(AnthropicClient())
    return _pipelines[name]


class Query(BaseModel):
    question: str
    pipeline: str = "naive"


def _run_ingest(job_id: str, paths: list[Path]) -> None:
    job = _jobs[job_id]
    try:
        docs = []
        for i, path in enumerate(paths, 1):
            job["status"] = f"parsing {i}/{len(paths)}: {path.name}"
            docs.append(parse_file(path))
        job["status"] = "indexing"
        _get("naive").index(docs)
        job.update(status="done", indexed_documents=len(docs))
    except Exception as exc:  # reported through GET /ingest/{job_id}
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        _job_lock.release()


def _start_job(paths: list[Path], background: BackgroundTasks) -> dict:
    if not paths:
        raise HTTPException(400, "No document to ingest")
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "An ingestion job is already running")
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "queued", "documents": [p.name for p in paths]}
    background.add_task(_run_ingest, job_id, paths)
    return {"job_id": job_id, "documents": len(paths)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile], background: BackgroundTasks) -> dict:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        dest = UPLOAD_DIR / Path(f.filename).name
        dest.write_bytes(await f.read())
        paths.append(dest)
    return _start_job(paths, background)


@app.post("/ingest/corpus")
def ingest_corpus(background: BackgroundTasks) -> dict:
    """Ingest every PDF/Markdown file under corpus/raw/ (see scripts/download_corpus.py)."""
    exts = {".pdf", ".md", ".markdown", ".txt"}
    paths = sorted(p for p in CORPUS_DIR.glob("*") if p.suffix.lower() in exts)
    return _start_job(paths, background)


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job")
    return {"job_id": job_id, **_jobs[job_id]}


@app.post("/query")
def query(q: Query) -> dict:
    return asdict(_get(q.pipeline).run(q.question))
