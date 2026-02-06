from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass
class ContextDocument:
    source_id: str
    title: str
    text: str
    chunks: list[str]


class ContextLibrary:
    def __init__(self, docs: list[ContextDocument], seed: int | None = None) -> None:
        if not docs:
            raise ValueError("No context documents found. Add .txt or .md files in context_dir.")
        self.docs = docs
        self.random = random.Random(seed)

    @classmethod
    def from_dir(cls, context_dir: str, max_chars_per_chunk: int = 1800, seed: int | None = None) -> "ContextLibrary":
        root = Path(context_dir)
        if not root.exists():
            raise ValueError(f"context_dir does not exist: {context_dir}")
        docs: list[ContextDocument] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            chunks = _chunk_text(text=text, max_chars=max_chars_per_chunk)
            docs.append(
                ContextDocument(
                    source_id=str(path.relative_to(root)),
                    title=path.stem,
                    text=text,
                    chunks=chunks,
                )
            )
        return cls(docs=docs, seed=seed)

    def sample(self, n_docs: int) -> list[dict[str, str]]:
        selected_docs = self.docs if n_docs >= len(self.docs) else self.random.sample(self.docs, k=n_docs)
        contexts: list[dict[str, str]] = []
        for doc in selected_docs:
            chunks = doc.chunks
            for index, chunk in enumerate(chunks):
                contexts.append(
                    {
                        "source_id": doc.source_id,
                        "title": doc.title,
                        "chunk_id": str(index),
                        "text": chunk,
                    }
                )
        return contexts


def _chunk_text(text: str, max_chars: int) -> list[str]:
    normalized = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    if not normalized:
        return [""]
    pieces: list[str] = []
    current = ""
    for paragraph in normalized.split("\n"):
        candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = paragraph[:max_chars]
    if current:
        pieces.append(current)
    return pieces
