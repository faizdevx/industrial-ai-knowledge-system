"""
Link retrieved document chunks to existing Neo4j entities.

Inputs:
  data/processed/refinery/documents/document_chunks.jsonl
  Neo4j graph containing Sensor nodes

Outputs:
  data/processed/refinery/documents/document_entity_links.jsonl

Also writes Neo4j:
  (:DocumentChunk)-[:MENTIONS]->(:Entity)

This is deliberately conservative:
  - only exact tag matches
  - no inferred causality
  - no guessed asset/component assignment
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "processed" / "refinery" / "documents" / "document_chunks.jsonl"
OUTPUT = ROOT / "data" / "processed" / "refinery" / "documents" / "document_entity_links.jsonl"


def load_chunks(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_tag(value: str) -> str:
    return value.strip().lower()


def fetch_sensor_tags(driver) -> list[str]:
    q = """
    MATCH (e:Entity)
    WHERE e.tag_family IS NOT NULL
    RETURN e.id AS id, e.source_column AS source_column
    ORDER BY e.id
    """
    with driver.session() as session:
        rows = session.run(q)
        tags = []
        for row in rows:
            tag = row.get("source_column") or row.get("id")
            if tag:
                tags.append(str(tag))
        return sorted(set(tags), key=len, reverse=True)


def make_pattern(tags: list[str]) -> re.Pattern | None:
    if not tags:
        return None
    escaped = [re.escape(t) for t in tags]
    return re.compile(r"(?<![A-Za-z0-9_.-])(" + "|".join(escaped) + r")(?![A-Za-z0-9_.-])", re.IGNORECASE)


def upsert_mention(driver, chunk: dict, sensor_tag: str) -> None:
    q = """
    MERGE (d:DocumentChunk {id: $chunk_id})
    SET d.document_id = $document_id,
        d.file_name = $file_name,
        d.page = $page,
        d.text = $text
    WITH d
    MATCH (e:Entity)
    WHERE toLower(coalesce(e.source_column, e.id)) = toLower($sensor_tag)
    MERGE (d)-[r:MENTIONS]->(e)
    SET r.evidence_type = 'exact_text_match',
        r.status = 'candidate',
        r.source_id = $file_name,
        r.page = $page
    """
    with driver.session() as session:
        session.run(
            q,
            chunk_id=chunk["chunk_id"],
            document_id=chunk.get("document_id"),
            file_name=chunk.get("file_name"),
            page=chunk.get("page"),
            text=chunk.get("text", ""),
            sensor_tag=sensor_tag,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="industrial-ai")
    args = parser.parse_args()

    if not CHUNKS.exists():
        raise FileNotFoundError(f"Missing chunk file: {CHUNKS}")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        driver.verify_connectivity()
        tags = fetch_sensor_tags(driver)
        pattern = make_pattern(tags)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        total_chunks = 0
        matched_chunks = 0
        total_links = 0

        with OUTPUT.open("w", encoding="utf-8") as out:
            for chunk in load_chunks(CHUNKS):
                total_chunks += 1
                text = chunk.get("text", "") or ""
                found = []

                if pattern:
                    seen = set()
                    for match in pattern.finditer(text):
                        tag = match.group(1)
                        key = normalize_tag(tag)
                        if key not in seen:
                            seen.add(key)
                            found.append(tag)

                if found:
                    matched_chunks += 1

                record = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk.get("document_id"),
                    "file_name": chunk.get("file_name"),
                    "page": chunk.get("page"),
                    "matched_sensor_tags": found,
                    "match_method": "exact_text_match",
                    "status": "candidate",
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                for tag in found:
                    upsert_mention(driver, chunk, tag)
                    total_links += 1

        print("=" * 60)
        print("DOCUMENT → ENTITY LINKING")
        print("=" * 60)
        print(f"Document chunks : {total_chunks}")
        print(f"Chunks with tags: {matched_chunks}")
        print(f"MENTIONS links  : {total_links}")
        print(f"Created         : {OUTPUT}")
        print()
        print("Important: exact text match is evidence of mention, not")
        print("proof of asset assignment, physical connection, or causality.")
        print("=" * 60)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
