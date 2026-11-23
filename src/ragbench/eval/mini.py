"""Week-1 mini evaluation of Pipeline 1 on data/eval_mini.json.

Not the final RAGAS protocol (week 3): a quick sanity check that retrieves,
answers, and judges each question.

- doc_hit / context_recall: how much of each reference quote shows up in the
  retrieved chunks of the right document (token coverage, a cheap proxy).
- verdict: correct / partial / incorrect from the judge model (distinct from the
  generator). For no_answer questions, the verdict is "correct" iff the system abstains.
- Judge failures (invalid JSON) are excluded from averages and reported.

Run inside the backend container, after /ingest/corpus has finished:
    docker compose exec backend python -m ragbench.eval.mini [naive|advanced]
"""
import json
import re
import statistics
import sys
import urllib.request
from datetime import date
from pathlib import Path

from ragbench.config import get_settings
from ragbench.llm.anthropic_client import AnthropicClient

API = "http://localhost:8000"
DATASET = Path("data/eval_mini.json")
OUT_DIR = Path("data/results")

JUDGE_SYSTEM = (
    "You are a strict evaluator of question answering. Compare the candidate answer "
    "with the reference answer. Reply with JSON only: "
    '{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}. '
    "'correct': same key facts and numbers as the reference. 'partial': some key facts "
    "right, some missing or wrong. 'incorrect': wrong or no relevant facts."
)
ABSTAIN_SYSTEM = (
    "The question cannot be answered from the corpus. Decide whether the candidate "
    "answer correctly declines to answer (says it does not know / the context does not "
    'say) instead of inventing a value. Reply with JSON only: {"verdict": "correct" if '
    'it declines, "incorrect" if it gives an answer, "reason": "<one sentence>"}.'
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def query(question: str, pipeline: str = "naive") -> dict:
    req = urllib.request.Request(
        f"{API}/query",
        data=json.dumps({"question": question, "pipeline": pipeline}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def retrieval_scores(result: dict, passages: list[dict]) -> dict:
    sources = {c["metadata"].get("source", "") for c in result["contexts"]}
    if not passages:
        return {"doc_hit": None, "context_recall": None}
    docs = {p["doc"] for p in passages}
    doc_hit = sum(any(s.startswith(d) for s in sources) for d in docs) / len(docs)
    coverages = []
    for p in passages:
        text = " ".join(
            c["text"] for c in result["contexts"]
            if c["metadata"].get("source", "").startswith(p["doc"])
        )
        quote = _tokens(p["quote"])
        coverages.append(len(quote & _tokens(text)) / len(quote) if quote else 0.0)
    return {"doc_hit": doc_hit, "context_recall": statistics.mean(coverages)}


def judge(client: AnthropicClient, q: dict, answer: str) -> dict:
    if q["type"] == "no_answer":
        system, user = ABSTAIN_SYSTEM, f"Question: {q['question']}\n\nCandidate answer: {answer}"
    else:
        system = JUDGE_SYSTEM
        user = (f"Question: {q['question']}\n\nReference answer: {q['reference_answer']}\n\n"
                f"Candidate answer: {answer}")
    res = client.generate(system, user)
    try:
        raw = re.search(r"\{.*\}", res.text, re.S).group(0)
        data = json.loads(raw)
        if data["verdict"] not in {"correct", "partial", "incorrect"}:
            raise ValueError(data["verdict"])
        return {"verdict": data["verdict"], "reason": data.get("reason", "")}
    except Exception:
        return {"verdict": None, "reason": f"unparseable judge output: {res.text[:120]!r}"}


def main() -> int:
    pipeline = sys.argv[1] if len(sys.argv) > 1 else "naive"
    questions = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]
    judge_client = AnthropicClient(model=get_settings().judge_model)
    rows = []
    for q in questions:
        if not q["question"]:
            continue
        r = query(q["question"], pipeline)
        row = {
            "id": q["id"], "type": q["type"], "answer": r["answer"],
            "sources": sorted({c["metadata"].get("source", "")[:3] for c in r["contexts"]}),
            "retrieval_ms": round(r["retrieval_ms"]), "ttft_ms": round(r["ttft_ms"] or 0),
            "total_ms": round(r["total_ms"]),
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
            **retrieval_scores(r, q["reference_passages"]),
            **judge(judge_client, q, r["answer"]),
        }
        rows.append(row)
        print(f"{row['id']} {row['type']:<13} {str(row['verdict']):<10} "
              f"recall={row['context_recall'] if row['context_recall'] is None else round(row['context_recall'], 2)} "
              f"src={row['sources']} ttft={row['ttft_ms']}ms | {row['reason'][:90]}")

    judged = [r for r in rows if r["verdict"]]
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    print("\n--- summary ---")
    print(f"questions: {len(rows)}, judge failures (excluded): {len(rows) - len(judged)}")
    if judged:
        print(f"answer correctness (correct=1, partial=0.5): "
              f"{statistics.mean(score[r['verdict']] for r in judged):.2f}")
    with_ref = [r for r in rows if r["context_recall"] is not None]
    if with_ref:
        print(f"doc hit: {statistics.mean(r['doc_hit'] for r in with_ref):.2f}  "
              f"context recall (token proxy): {statistics.mean(r['context_recall'] for r in with_ref):.2f}")
    for t in sorted({r["type"] for r in judged}):
        sub = [r for r in judged if r["type"] == t]
        print(f"  {t:<13} n={len(sub)} correctness={statistics.mean(score[r['verdict']] for r in sub):.2f}")
    print(f"median ttft: {statistics.median(r['ttft_ms'] for r in rows)} ms, "
          f"median total: {statistics.median(r['total_ms'] for r in rows)} ms")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mini_eval_{pipeline}_{date.today().isoformat()}.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
