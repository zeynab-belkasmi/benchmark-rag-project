import pytest
from fastapi.testclient import TestClient

from ragbench.api import main
from ragbench.ingestion.parser import Document


class FakePipeline:
    def __init__(self):
        self.indexed = None

    def index(self, docs):
        self.indexed = docs


@pytest.fixture
def client(monkeypatch, tmp_path):
    """API client with a fake pipeline, a fake parser and a temporary corpus folder."""
    pipeline = FakePipeline()
    monkeypatch.setattr(main, "_get", lambda name: pipeline)
    monkeypatch.setattr(main, "parse_file", lambda p: Document(p.stem, "text"))
    monkeypatch.setattr(main, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "up")
    main._jobs.clear()
    c = TestClient(main.app)
    c.pipeline, c.corpus = pipeline, tmp_path
    return c


def test_empty_corpus_is_rejected(client):
    assert client.post("/ingest/corpus").status_code == 400


def test_corpus_job_runs_and_reports_done(client):
    """The job parses every supported file, indexes them in one call and ends 'done'.

    Unsupported files (.zip) are ignored. The lock must be released afterwards.
    """
    (client.corpus / "A01.pdf").write_bytes(b"x")
    (client.corpus / "B01.md").write_text("x")
    (client.corpus / "skip.zip").write_bytes(b"x")
    r = client.post("/ingest/corpus")
    assert r.json()["documents"] == 2
    status = client.get(f"/ingest/{r.json()['job_id']}").json()
    assert status["status"] == "done" and status["indexed_documents"] == 2
    assert [d.doc_id for d in client.pipeline.indexed] == ["A01", "B01"]
    assert main._job_lock.acquire(blocking=False)
    main._job_lock.release()


def test_second_job_while_running_gets_409(client):
    """Indexing replaces the whole collection, so concurrent jobs are refused."""
    (client.corpus / "A01.md").write_text("x")
    assert main._job_lock.acquire(blocking=False)
    try:
        assert client.post("/ingest/corpus").status_code == 409
    finally:
        main._job_lock.release()


def test_failed_job_reports_error_and_releases_lock(client, monkeypatch):
    (client.corpus / "A01.md").write_text("x")

    def boom(p):
        raise RuntimeError("parse error")

    monkeypatch.setattr(main, "parse_file", boom)
    job_id = client.post("/ingest/corpus").json()["job_id"]
    status = client.get(f"/ingest/{job_id}").json()
    assert status["status"] == "failed" and "parse error" in status["error"]
    assert main._job_lock.acquire(blocking=False)
    main._job_lock.release()


def test_unknown_job_is_404(client):
    assert client.get("/ingest/nope").status_code == 404
