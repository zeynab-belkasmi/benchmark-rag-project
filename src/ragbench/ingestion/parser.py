"""Single parsing step shared by all 4 pipelines (spec section 3): PDF/Markdown -> Document."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    text: str  # Markdown (headings and tables preserved)
    metadata: dict = field(default_factory=dict)


def parse_file(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(str(path))
    elif suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return Document(doc_id=path.stem, text=text, metadata={"source": path.name})
