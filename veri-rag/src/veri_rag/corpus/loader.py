"""Document loader — reads .txt, .md, .jsonl files into Document objects."""

from __future__ import annotations

import json
from pathlib import Path

from veri_rag.config.schema import Document


def load_documents(data_dir: str | Path, file_types: list[str] | None = None) -> list[Document]:
    """Recursively load documents from a directory.

    Args:
        data_dir: Path to the directory containing source files.
        file_types: List of file extensions to include (e.g., ['.txt', '.md']).

    Returns:
        List of Document objects.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_types = file_types or [".txt", ".md", ".jsonl"]
    documents: list[Document] = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in file_types:
            continue

        if file_path.suffix.lower() == ".jsonl":
            documents.extend(_load_jsonl(file_path))
        else:
            documents.append(_load_text_file(file_path))

    return documents


def _load_text_file(file_path: Path) -> Document:
    """Load a single text/markdown file as a Document."""
    text = file_path.read_text(encoding="utf-8")
    return Document(
        source=str(file_path),
        title=file_path.stem,
        text=text,
        metadata={"file_type": file_path.suffix, "file_name": file_path.name},
    )


def _load_jsonl(file_path: Path) -> list[Document]:
    """Load a JSONL file where each line is a document dict."""
    documents: list[Document] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                doc = Document(
                    document_id=data.get("document_id", f"{file_path.stem}_{line_num}"),
                    source=data.get("source", str(file_path)),
                    title=data.get("title", ""),
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                    is_attack=data.get("is_attack", False),
                )
                documents.append(doc)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping line {line_num} in {file_path}: {e}")
    return documents
