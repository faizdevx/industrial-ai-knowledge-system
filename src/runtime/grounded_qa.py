"""
Grounded QA evidence builder for the refinery MVP.

This stage does NOT generate an answer with an LLM yet.

It builds a machine-readable evidence bundle from:
  - local BGE document retrieval
  - DocumentChunk -> Entity MENTIONS links in Neo4j
  - Entity -> Event DETECTED_BY links in Neo4j

The bundle is designed to become the input to the later local LLM/RAG
reasoning stage.

Usage:
  uv run python src/runtime/grounded_qa.py "compressor instrumentation"
  uv run python src/runtime/grounded_qa.py "What happened around a pressure anomaly?" --top-k 8
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
OUTPUT_DIR = ROOT / "data" / "processed" / "refinery" / "runtime"

EMBEDDINGS = DOC_DIR / "embeddings.npy"
METADATA = DOC_DIR / "chunk_metadata.jsonl"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def semantic_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Embedding/metadata mismatch: {len(embeddings)} embeddings vs "
            f"{len(metadata)} metadata records."
        )

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_vector
    indices = np.argsort(-scores)[:top_k]

    hits = []
    for rank, idx in enumerate(indices, start=1):
        meta = metadata[int(idx)]
        hits.append({
            "rank": rank,
            "score": round(float(scores[int(idx)]), 6),
            "chunk_id": meta.get("chunk_id"),
            "document_id": meta.get("document_id"),
            "file": meta.get("file_name") or meta.get("file"),
            "page": meta.get("page"),
            "headings": meta.get("headings"),
            "text": meta.get("text", ""),
            "provenance": meta.get("provenance", {}),
        })

    return hits


def fetch_graph_evidence(
    driver,
    chunk_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not chunk_ids:
        return {"entities": [], "events": []}

    query = """
    MATCH (d:DocumentChunk)-[:MENTIONS]->(e:Entity)
    WHERE d.id IN $chunk_ids
    OPTIONAL MATCH (ev:Event)-[:DETECTED_BY]->(e)
    RETURN
        d.id AS chunk_id,
        e.id AS entity_id,
        e.source_column AS source_column,
        e.candidate_class AS candidate_class,
        e.tag_family AS tag_family,
        e.confidence AS entity_confidence,
        ev.id AS event_id,
        ev.event_type AS event_type,
        ev.start AS event_start,
        ev.end AS event_end,
        ev.peak_value AS peak_value,
        ev.peak_robust_z AS peak_robust_z,
        ev.direction AS direction,
        ev.status AS event_status,
        ev.interpretation AS interpretation
    """

    entities: dict[tuple[str, str], dict[str, Any]] = {}
    events: dict[tuple[str, str], dict[str, Any]] = {}

    with driver.session() as session:
        result = session.run(query, chunk_ids=chunk_ids)

        for row in result:
            chunk_id = row["chunk_id"]
            entity_id = row["entity_id"]

            if entity_id:
                entity_key = (chunk_id, entity_id)
                entities[entity_key] = {
                    "chunk_id": chunk_id,
                    "entity_id": entity_id,
                    "source_column": row["source_column"],
                    "candidate_class": row["candidate_class"],
                    "tag_family": row["tag_family"],
                    "confidence": row["entity_confidence"],
                    "evidence": "DocumentChunk MENTIONS Entity",
                }

            if row["event_id"]:
                event_key = (entity_id, row["event_id"])
                events[event_key] = {
                    "entity_id": entity_id,
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "start": row["event_start"],
                    "end": row["event_end"],
                    "peak_value": row["peak_value"],
                    "peak_robust_z": row["peak_robust_z"],
                    "direction": row["direction"],
                    "status": row["event_status"],
                    "interpretation": row["interpretation"],
                    "evidence": "Event DETECTED_BY Entity",
                }

    return {
        "entities": list(entities.values()),
        "events": list(events.values()),
    }


def build_bundle(
    query: str,
    hits: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "query": query,
        "retrieval": {
            "method": "semantic_document_retrieval_plus_neo4j_evidence",
            "embedding_model": MODEL_NAME,
            "top_k": len(hits),
        },
        "document_evidence": hits,
        "graph_evidence": graph,
        "answer_policy": {
            "must_use_only_retrieved_evidence": True,
            "must_preserve_source_provenance": True,
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
            "express_uncertainty_when_evidence_is_insufficient": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "industrial-ai"),
    )
    args = parser.parse_args()

    if not EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing embeddings: {EMBEDDINGS}")

    metadata = load_metadata(METADATA)
    embeddings = np.load(EMBEDDINGS)

    print("=" * 72)
    print("GROUNDED QA EVIDENCE BUILDER")
    print("=" * 72)
    print(f"Query        : {args.query}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Embedding rows: {len(embeddings)}")
    print()

    model = SentenceTransformer(MODEL_NAME)

    hits = semantic_search(
        args.query,
        model,
        embeddings,
        metadata,
        max(1, args.top_k),
    )

    chunk_ids = [
        hit["chunk_id"]
        for hit in hits
        if hit.get("chunk_id")
    ]

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    try:
        driver.verify_connectivity()
        graph = fetch_graph_evidence(driver, chunk_ids)
    finally:
        driver.close()

    bundle = build_bundle(args.query, hits, graph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in args.query.lower()
    ).strip("_")[:80]

    if not safe_name:
        safe_name = "query"

    output_path = OUTPUT_DIR / f"evidence_{safe_name}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False, default=str)

    print("DOCUMENT EVIDENCE")
    print("-" * 72)

    for hit in hits:
        print(
            f"[{hit['rank']}] score={hit['score']} | "
            f"{hit['file']} p.{hit['page']} | {hit['chunk_id']}"
        )

    print()
    print("GRAPH EVIDENCE")
    print("-" * 72)
    print(f"Entities linked : {len(graph['entities'])}")
    print(f"Events linked   : {len(graph['events'])}")

    for entity in graph["entities"][:20]:
        print(
            f"Entity: {entity['entity_id']} | "
            f"class={entity['candidate_class']} | "
            f"column={entity['source_column']}"
        )

    print()
    print(f"Evidence bundle : {output_path}")
    print()
    print("No LLM answer was generated.")
    print("This stage only assembles grounded evidence for later reasoning.")
    print("=" * 72)


if __name__ == "__main__":
    main()
