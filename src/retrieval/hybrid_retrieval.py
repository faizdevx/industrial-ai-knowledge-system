"""
Hybrid retrieval for the refinery MVP.

Combines:
  1. local BGE semantic document search
  2. Neo4j entity/event context

No LLM is required yet. The output is a grounded context bundle
that the later RAG/agent layer can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
EMBEDDINGS = DOC_DIR / "embeddings.npy"
METADATA = DOC_DIR / "chunk_metadata.jsonl"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def semantic_search(query: str, top_k: int) -> list[tuple[int, float]]:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = np.load(EMBEDDINGS)
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]


def graph_context(driver, sensor_tags: list[str]) -> dict:
    if not sensor_tags:
        return {"sensors": [], "events": []}

    q = """
    MATCH (e:Entity)
    WHERE any(tag IN $tags
              WHERE toLower(coalesce(e.source_column, e.id)) = toLower(tag))
    OPTIONAL MATCH (ev:Event)-[:DETECTED_BY]->(e)
    RETURN e.id AS sensor_id,
           e.source_column AS source_column,
           e.candidate_class AS candidate_class,
           collect(DISTINCT {
               id: ev.id,
               event_type: ev.event_type,
               start: ev.start,
               end: ev.end,
               status: ev.status,
               interpretation: ev.interpretation
           }) AS events
    """
    with driver.session() as session:
        rows = list(session.run(q, tags=sensor_tags))

    sensors = []
    events = []
    for row in rows:
        sensors.append({
            "id": row["sensor_id"],
            "source_column": row["source_column"],
            "candidate_class": row["candidate_class"],
        })
        for event in row["events"]:
            if event.get("id"):
                events.append(event)

    return {"sensors": sensors, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="industrial-ai")
    args = parser.parse_args()

    metadata = load_metadata(METADATA)
    hits = semantic_search(args.query, args.top_k)

    selected = []
    sensor_tags = set()

    for idx, score in hits:
        meta = metadata[idx]
        text = meta.get("text", "")
        selected.append({
            "rank": len(selected) + 1,
            "score": round(score, 4),
            "chunk_id": meta.get("chunk_id"),
            "file": meta.get("file_name"),
            "page": meta.get("page"),
            "text": text,
            "provenance": {
                "file": meta.get("file_name"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
            },
        })

    # Pull graph entities mentioned in the selected chunks.
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            rows = session.run("""
                MATCH (d:DocumentChunk)-[:MENTIONS]->(e:Entity)
                WHERE d.id IN $chunk_ids
                RETURN DISTINCT coalesce(e.source_column, e.id) AS tag
            """, chunk_ids=[x["chunk_id"] for x in selected])
            sensor_tags = {row["tag"] for row in rows if row["tag"]}

        context = graph_context(driver, sorted(sensor_tags))
    finally:
        driver.close()

    print("=" * 72)
    print("HYBRID RETRIEVAL")
    print("=" * 72)
    print(f"Query       : {args.query}")
    print(f"Semantic hits: {len(selected)}")
    print(f"Graph tags   : {len(sensor_tags)}")
    print()

    print("DOCUMENT EVIDENCE")
    print("-" * 72)
    for item in selected:
        print(f"[{item['rank']}] score={item['score']}")
        print(f"    {item['file']} p.{item['page']}  {item['chunk_id']}")
        print(f"    {item['text'][:500].replace(chr(10), ' ')}")
        print()

    print("GRAPH CONTEXT")
    print("-" * 72)
    for sensor in context["sensors"]:
        print(
            f"Sensor: {sensor['id']} | "
            f"class={sensor['candidate_class']} | "
            f"column={sensor['source_column']}"
        )

    print(f"\nCandidate events: {len(context['events'])}")
    print()
    print("Important: retrieved events are statistical candidates,")
    print("not confirmed faults. Document mention is not causal evidence.")
    print("=" * 72)


if __name__ == "__main__":
    main()
