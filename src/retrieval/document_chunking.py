"""
Stage 11: turn page-aware PDF text into retrieval-ready chunks.

Input:
    data/processed/refinery/documents/document_pages.jsonl

Output:
    data/processed/refinery/documents/document_chunks.jsonl

Every chunk keeps:
    document_id, file_name, page, chunk_id, chunk position, text,
    and provenance.

This is the simple MVP chunker. Later we can replace it with semantic
or token-aware chunking without changing the output contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/processed/refinery/documents/document_pages.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/processed/refinery/documents/document_chunks.jsonl"
)

DEFAULT_MAX_CHARS = 1600
DEFAULT_OVERLAP = 250


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return records


def make_chunk_id(
    document_id: str,
    page: int,
    chunk_index: int,
) -> str:
    raw = f"{document_id}|{page}|{chunk_index}"
    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]
    return f"chunk_{digest}"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def split_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Character-based overlapping chunks.

    We prefer breaking at whitespace so words are not chopped in half.
    This is intentionally simple for the first MVP.
    """
    text = normalize_text(text)

    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0.")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be >= 0 and < max_chars."
        )

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        hard_end = min(
            start + max_chars,
            len(text),
        )

        end = hard_end

        if hard_end < len(text):
            boundary = text.rfind(
                " ",
                start + max_chars // 2,
                hard_end,
            )

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap_chars,
            start + 1,
        )

    return chunks


def build_chunks(
    pages: list[dict[str, Any]],
    max_chars: int,
    overlap_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    empty_pages = 0

    for page in pages:
        text = str(page.get("text", "")).strip()

        page_chunks = split_text(
            text=text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        if not page_chunks:
            empty_pages += 1
            continue

        document_id = str(page["document_id"])
        file_name = str(page["file_name"])
        page_number = int(page["page"])

        for index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        document_id,
                        page_number,
                        index,
                    ),
                    "document_id": document_id,
                    "file_name": file_name,
                    "page": page_number,
                    "chunk_index": index,
                    "chunk_count_on_page": len(page_chunks),
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "provenance": {
                        "source_id": document_id,
                        "source_type": "pdf",
                        "location": {
                            "file": file_name,
                            "page": page_number,
                            "chunk_index": index,
                        },
                        "extraction_method": page.get(
                            "provenance", {}
                        ).get(
                            "extraction_method",
                            "PyMuPDF",
                        ),
                    },
                }
            )

    return chunks, empty_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create retrieval-ready PDF text chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
    )

    args = parser.parse_args()

    pages = load_jsonl(args.input.resolve())

    chunks, empty_pages = build_chunks(
        pages=pages,
        max_chars=args.max_chars,
        overlap_chars=args.overlap,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 60)
    print("DOCUMENT CHUNKING")
    print("=" * 60)
    print(f"Pages processed : {len(pages)}")
    print(f"Empty pages     : {empty_pages}")
    print(f"Chunks created  : {len(chunks)}")
    print(f"Max chars       : {args.max_chars}")
    print(f"Overlap chars   : {args.overlap}")
    print()
    print(f"Created: {output_path}")
    print()
    print(
        "Next stage: embeddings + vector retrieval."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()