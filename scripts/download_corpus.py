"""Download the corpus PDFs listed in corpus/SOURCES.md into corpus/raw/.

Each document's license is re-read on its arXiv abstract page and the download
is refused unless it is CC BY 4.0.

Usage: python scripts/download_corpus.py
"""

import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "corpus" / "SOURCES.md"
RAW = ROOT / "corpus" / "raw"
CC_BY = "creativecommons.org/licenses/by/4.0"
ROW = re.compile(r"^\|\s*([A-Z]\d{2})\s*\|.*?\|\s*`(\d{4}\.\d{4,5})`\s*\|", re.M)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-benchlab-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    entries = ROW.findall(SOURCES.read_text(encoding="utf-8"))
    if not entries:
        print("No entries found in SOURCES.md")
        return 1
    RAW.mkdir(parents=True, exist_ok=True)
    failures = []
    for doc_id, arxiv_id in entries:
        target = RAW / f"{doc_id}_{arxiv_id}.pdf"
        if target.exists():
            print(f"skip   {target.name} (already downloaded)")
            continue
        try:
            abstract = fetch(f"https://arxiv.org/abs/{arxiv_id}").decode("utf-8", "ignore")
            if CC_BY not in abstract:
                print(f"REFUSE {doc_id} {arxiv_id}: not CC BY 4.0")
                failures.append(doc_id)
                continue
            target.write_bytes(fetch(f"https://arxiv.org/pdf/{arxiv_id}"))
            print(f"ok     {target.name}")
        except Exception as exc:  # network errors, 404, ...
            print(f"FAIL   {doc_id} {arxiv_id}: {exc}")
            failures.append(doc_id)
        time.sleep(3)  # be polite to arXiv
    print(f"\n{len(entries) - len(failures)}/{len(entries)} documents available")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
