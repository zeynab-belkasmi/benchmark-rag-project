# RAG-BenchLab

Side-by-side benchmark of four RAG architectures on one shared corpus: **Naive**, **Advanced** (hybrid search + reranking), **GraphRAG** and **Agentic** (LangGraph).

The question it tries to answer: *when does a more elaborate RAG pipeline actually beat a plain one, and at what cost in latency and tokens?* Every pipeline is evaluated on the same questions, with the same LLM, parser, embeddings, prompt and context budget, so that only the retrieval/orchestration strategy changes.

> **Status: work in progress** (4-week schedule, currently in week 2). Numbers below are early and indicative.

## Pipelines

| # | Pipeline | Strategy | Status |
|---|----------|----------|--------|
| P1 | Naive RAG | 512-token chunks (overlap 50), dense search, top-4 | Done, baseline evaluated |
| P2 | Advanced RAG | Sentence-window indexing, hybrid dense + BM25 with RRF fusion, cross-encoder reranking (top-3) | Implemented, mini-eval done |
| P3 | GraphRAG | LLM entity/relation extraction, communities, local and global search (NetworkX) | Planned (week 2) |
| P4 | Agentic RAG | LangGraph: routing, retrieval, grading, query reformulation, faithfulness check | Planned (week 3) |

A Streamlit UI and the final benchmark (quality, faithfulness, latency, cost, with confidence intervals) are planned for week 4.

## Fair-comparison rules

- Same generation LLM, same parser (`pymupdf4llm`), same embeddings (`bge-m3`) and same prompt for all pipelines.
- The prompt and the context-budget truncation (at most 2,048 tokens) live in the shared base class, not in the pipelines.
- The judge model is different from the generator.
- Metrics that do not apply to a pipeline are reported as `N/A`, never as 0.
- Latency is measured sequentially, with warm-up and repetitions.

## Architecture

```
PDF / Markdown --> parser --> chunking (per pipeline) --> Qdrant (one collection per pipeline)
                                                              |
question --> /query?pipeline=... --> retrieve --> context budget --> Claude --> answer + metrics
```

- **LLM:** Claude API (`claude-sonnet-5-5` for generation, `claude-haiku-5-5` for graph extraction, `claude-opus-5-5` as judge).
- **Local models:** `BAAI/bge-m3` embeddings and `BAAI/bge-reranker-v2-m3` reranker (the Anthropic API has no embeddings endpoint).
- **Vector store:** Qdrant in server mode, persistent on disk, one collection per pipeline. A persistent database was chosen over an in-memory library such as FAISS because indexing is expensive on CPU, and because Advanced RAG needs hybrid dense + sparse search with RRF fusion and metadata filtering, which Qdrant provides server-side. FAISS would suit a small static corpus that can be rebuilt in seconds.
- **Stack:** Python 3.11, FastAPI, Qdrant, fastembed (BM25 sparse vectors), sentence-transformers, NetworkX, LangGraph, RAGAS, Streamlit, Docker Compose.

## Corpus

23 arXiv papers on LLM systems, RAG, evaluation and agents, all under **CC BY 4.0**. The list is in [corpus/SOURCES.md](corpus/SOURCES.md). PDFs are not stored in the repository: `scripts/download_corpus.py` downloads them and re-checks each paper's license before downloading.

## Quick start

Requirements: Docker, Python 3 (for the download script) and an Anthropic API key.

```bash
# 1. Configuration: put your API key in .env (never commit it)
cp .env.example .env

# 2. Download the corpus (license is re-checked for each paper)
python scripts/download_corpus.py

# 3. Build and start the stack (Qdrant + backend)
docker build -t benchmark-rag-project-backend .
docker compose up -d --no-build

# 4. Download the local models (about 4.5 GB, once)
docker compose exec backend python scripts/init_models.py

# 5. Index the corpus for all pipelines (slow on CPU; check progress with the job id)
curl -X POST localhost:8000/ingest/corpus
curl localhost:8000/ingest/<job_id>

# 6. Ask a question
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "Which three quality dimensions does Ragas measure?", "pipeline": "advanced"}'
```

The response contains the answer, the contexts actually injected into the prompt, latencies (retrieval, time to first token, total) and token counts.

Notes:

- `docker-compose.yml` mounts a `certs/` folder and sets `SSL_CERT_FILE`-style variables to trust a TLS-inspecting antivirus. On a machine without interception, remove those lines and the `certs` volume.
- Indexing replaces a collection as a whole, so ingest the full corpus in one job.

## Evaluation

A first mini evaluation script runs a small question set ([data/eval_mini.json](data/eval_mini.json): factual, multi-passage, relational and no-answer questions) and judges answers with a separate model:

```bash
docker compose exec backend python -m ragbench.eval.mini naive      # or: advanced
```

**Mini evaluation (10 questions, indicative only, judge = `claude-opus-5-5`):**

| Metric | P1 Naive | P2 Advanced |
|--------|----------|-------------|
| Answer correctness (correct = 1, partial = 0.5) | 0.70 | 0.80 |
| Document hit | 0.96 | 0.80 |
| Median input tokens | 2,901 | 1,143 |
| Median time to first token | 0.9 s | 0.8 s |
| Median total latency | 2.4 s | 18.8 s |

P1 failures are mostly retrieval misses (table values, or the second paper of a multi-document question), and the model abstains rather than inventing an answer. P2's reranker recovers several of them (factual questions 0.88 -> 1.00, multi-passage 0.50 -> 0.67), but its three sentence windows carry less than half of P1's context, and one multi-document question is lost.

P2's latency is almost entirely the `bge-reranker-v2-m3` cross-encoder running on CPU (~16 s for 20 candidate windows). Faster variants were measured: dynamic int8 quantization (~10.6 s total, correctness 0.70) and `ms-marco-MiniLM-L-6-v2` (~2.9 s total, correctness 0.65). The reranker is configurable (`RERANKER_MODEL`, `RERANKER_INT8`), so this trade-off can be measured properly on the full question set. With n = 10 these figures are a sanity check, not a result.

## Tests

```bash
docker cp tests <backend_container>:/app/tests
docker compose exec backend python -m pytest -q
```

## Repository layout

```
src/ragbench/
  api/         FastAPI app (health, ingestion jobs, query)
  ingestion/   parser, chunking, sentence-window splitting
  indexing/    embeddings, Qdrant stores (dense, hybrid), reranker
  llm/         LLM client interface and Anthropic backend
  pipelines/   shared base class, Naive RAG, Advanced RAG
  eval/        mini evaluation script
corpus/        sources list (PDFs are downloaded, not committed)
data/          evaluation set
scripts/       corpus download, model download
tests/         unit tests
```

## License

See [LICENSE](LICENSE). Corpus documents keep their own licenses (CC BY 4.0); see [corpus/SOURCES.md](corpus/SOURCES.md).
