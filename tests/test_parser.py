import pytest

from ragbench.ingestion.parser import parse_file


def test_markdown_file_is_parsed_as_is(tmp_path):
    """A Markdown file becomes a Document keeping its text, id and source name."""
    f = tmp_path / "paper.md"
    f.write_text("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    doc = parse_file(f)
    assert doc.doc_id == "paper"
    assert doc.metadata == {"source": "paper.md"}
    assert "| 1 | 2 |" in doc.text


def test_unsupported_format_is_rejected(tmp_path):
    """Unknown extensions raise instead of being silently skipped."""
    f = tmp_path / "data.docx"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        parse_file(f)
