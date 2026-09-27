"""
Stage 13 of the Industrial AI MVP:
    natural-language query
            ↓
    local embedding
            ↓
    cosine similarity
            ↓
    top-k document chunks with provenance

Usage:
    uv run python src/retrieval/document_search.py \
        "What does the pressure signal indicate?"

Options:
    --query "..."
    --top-k 5
    --model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_DIR = Path(
    "data/processed/refinery/documents"
)
DEFAULT_EMBEDDINGS = DEFAULT_DIR / "embeddings.npy"
DEFAULT_METADATA = DEFAULT_DIR / "chunk_metadata.jsonl"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Query embedding has zero norm.")

    return vector / norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the local document embedding index."
    )

    parser.add_argument(
        "query",
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be > 0.")

    embeddings_path = args.embeddings.resolve()
    metadata_path = args.metadata.resolve()

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            "Run document_embedding_index.py first."
        )

    embeddings = np.load(
        embeddings_path,
    )

    metadata = load_metadata(metadata_path)

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata count ({len(metadata)}) does not match "
            f"embedding count ({len(embeddings)})."
        )

    print("=" * 60)
    print("SEMANTIC DOCUMENT SEARCH")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Model: {args.model}")
    print()

    model = SentenceTransformer(args.model)

    query_embedding = model.encode(
        [args.query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )[0]

    query_embedding = normalize(
        np.asarray(
            query_embedding,
            dtype=np.float32,
        )
    )

    # Embeddings were L2-normalized during indexing, so dot product is
    # cosine similarity.
    scores = embeddings @ query_embedding

    top_k = min(args.top_k, len(scores))

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        item = metadata[int(index)]

        print(f"[{rank}] score={float(scores[index]):.4f}")
        print(
            f"    file : {item['file_name']}"
        )
        print(
            f"    page : {item['page']}"
        )
        print(
            f"    chunk: {item['chunk_id']}"
        )
        print(
            f"    text : {item['text'][:700]}"
        )
        print(
            f"    provenance: "
            f"{item['provenance']['location']}"
        )
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()