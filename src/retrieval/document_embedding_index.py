"""
Stage 12 of the Industrial AI MVP:
    document_chunks.jsonl
            ↓
    local embedding model
            ↓
    normalized embeddings
            ↓
    persistent vector index

For the MVP, numpy dot-product search is used instead of adding a separate
vector database. With ~1,500 chunks this is fast, simple, and easy to debug.
FAISS/Chroma can replace this later without changing the chunk format.

Usage:
    uv add sentence-transformers numpy
    uv run python src/retrieval/document_embedding_index.py

Outputs:
    data/processed/refinery/documents/
        embeddings.npy
        chunk_metadata.jsonl
        embedding_index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INPUT = Path(
    "data/processed/refinery/documents/document_chunks.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/refinery/documents"
)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 32


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


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.where(norms == 0, 1.0, norms)

    return embeddings / norms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a persistent local embedding index."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformers model name or local model path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0.")

    chunks = load_jsonl(args.input.resolve())

    if not chunks:
        raise ValueError("No document chunks were found.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        str(chunk.get("text", "")).strip()
        for chunk in chunks
    ]

    if any(not text for text in texts):
        raise ValueError(
            "At least one chunk has empty text. "
            "Fix the chunking stage before building embeddings."
        )

    print("=" * 60)
    print("DOCUMENT EMBEDDING INDEX")
    print("=" * 60)
    print(f"Chunks        : {len(chunks)}")
    print(f"Model         : {args.model}")
    print(f"Batch size    : {args.batch_size}")
    print()
    print("Loading embedding model...")

    model = SentenceTransformer(args.model)

    print("Encoding chunks...")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    embeddings = l2_normalize(embeddings)

    embeddings_path = output_dir / "embeddings.npy"
    np.save(
        embeddings_path,
        embeddings,
    )

    metadata_path = output_dir / "chunk_metadata.jsonl"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for chunk in chunks:
            # Keep exactly the information needed to retrieve and cite
            # the original source later. Do not duplicate unnecessary text.
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "text": chunk["text"],
                "provenance": chunk["provenance"],
            }

            f.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                )
                + "\n"
            )

    index_info = {
        "index_type": "numpy_cosine",
        "model": args.model,
        "metric": "cosine_similarity",
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": int(embeddings.shape[0]),
        "source_chunks": str(args.input.resolve()),
        "embeddings_file": str(embeddings_path),
        "metadata_file": str(metadata_path),
    }

    with (
        output_dir / "embedding_index.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            index_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Embedding shape : {embeddings.shape}")
    print(f"Created         : {embeddings_path}")
    print(f"Created         : {metadata_path}")
    print(f"Created         : {output_dir / 'embedding_index.json'}")
    print()
    print(
        "Next stage: run semantic retrieval against the local index."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
