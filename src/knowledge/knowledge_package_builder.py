"""
Build a reproducible refinery knowledge package from the artifacts produced
by the current MVP pipeline.

Inputs:
  data/processed/refinery/domain_hypothesis.json
  data/processed/refinery/ontology_candidate.json
  data/processed/refinery/validated_knowledge.json
  data/processed/refinery/validation_report.json
  data/processed/refinery/timeseries/timeseries_profile.json
  data/processed/refinery/timeseries/sensor_timeseries_summary.csv
  data/processed/refinery/timeseries/event_candidates.json
  data/processed/refinery/documents/document_catalog.json
  data/processed/refinery/documents/document_chunks.jsonl
  data/processed/refinery/documents/document_entity_links.jsonl
  data/processed/refinery/documents/embedding_index.json
  data/processed/refinery/documents/embeddings.npy
  data/processed/refinery/documents/chunk_metadata.jsonl

Neo4j:
  Export graph nodes and relationships into the package.

Output:
  data/knowledge_package/refinery/
    manifest.json
    README.md
    ontology/
    graph/
    documents/
    timeseries/
    events/
    provenance/
    validation/
    metadata/

This stage packages evidence. It does not add new inferred engineering facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "refinery"
OUT = ROOT / "data" / "knowledge_package" / "refinery"

TEXT_JSON_INPUTS = {
    "domain_hypothesis": PROCESSED / "domain_hypothesis.json",
    "ontology_candidate": PROCESSED / "ontology_candidate.json",
    "validated_knowledge": PROCESSED / "validated_knowledge.json",
    "validation_report": PROCESSED / "validation_report.json",
}

TS_INPUTS = {
    "timeseries_profile": PROCESSED / "timeseries" / "timeseries_profile.json",
    "sensor_timeseries_summary": PROCESSED / "timeseries" / "sensor_timeseries_summary.csv",
    "event_candidates": PROCESSED / "timeseries" / "event_candidates.json",
}

DOC_INPUTS = {
    "document_catalog": PROCESSED / "documents" / "document_catalog.json",
    "document_chunks": PROCESSED / "documents" / "document_chunks.jsonl",
    "document_entity_links": PROCESSED / "documents" / "document_entity_links.jsonl",
    "embedding_index": PROCESSED / "documents" / "embedding_index.json",
    "embeddings": PROCESSED / "documents" / "embeddings.npy",
    "chunk_metadata": PROCESSED / "documents" / "chunk_metadata.jsonl",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src: Path, dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "source": str(src.relative_to(ROOT)),
        "package_path": str(dest.relative_to(OUT)),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)


def extract_provenance(validated: dict[str, Any]) -> dict[str, Any]:
    entities = validated.get("entities", [])
    relations = validated.get("relations", [])

    entity_provenance = []
    for entity in entities:
        entity_provenance.append({
            "entity_id": entity.get("id"),
            "source_id": entity.get("source_id"),
            "source_type": entity.get("source_type"),
            "source_sheet": entity.get("source_sheet"),
            "source_column": entity.get("source_column"),
            "source_row_start": entity.get("source_row_start"),
            "source_row_end": entity.get("source_row_end"),
            "extraction_method": entity.get("extraction_method"),
            "confidence": entity.get("confidence"),
            "status": entity.get("status"),
        })

    relation_provenance = []
    for relation in relations:
        relation_provenance.append({
            "subject": relation.get("subject"),
            "predicate": relation.get("predicate"),
            "object": relation.get("object"),
            "source_id": relation.get("source_id"),
            "source_type": relation.get("source_type"),
            "evidence": relation.get("evidence"),
            "extraction_method": relation.get("extraction_method"),
            "confidence": relation.get("confidence"),
            "status": relation.get("status"),
        })

    return {
        "description": "Machine-readable provenance index for packaged knowledge.",
        "generated_at": utc_now(),
        "entities": entity_provenance,
        "relations": relation_provenance,
    }


def export_neo4j(
    uri: str,
    user: str,
    password: str,
    out_dir: Path,
) -> dict[str, Any]:
    nodes_path = out_dir / "graph_nodes.jsonl"
    rels_path = out_dir / "graph_relationships.jsonl"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()

        with driver.session() as session:
            node_result = session.run("""
                MATCH (n)
                RETURN elementId(n) AS element_id,
                       labels(n) AS labels,
                       properties(n) AS properties
                ORDER BY elementId(n)
            """)

            node_count = 0
            with nodes_path.open("w", encoding="utf-8") as f:
                for row in node_result:
                    record = {
                        "element_id": row["element_id"],
                        "labels": list(row["labels"] or []),
                        "properties": dict(row["properties"] or {}),
                    }
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    node_count += 1

            rel_result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN elementId(r) AS element_id,
                       elementId(a) AS start_element_id,
                       elementId(b) AS end_element_id,
                       type(r) AS type,
                       properties(r) AS properties
                ORDER BY elementId(r)
            """)

            rel_count = 0
            with rels_path.open("w", encoding="utf-8") as f:
                for row in rel_result:
                    record = {
                        "element_id": row["element_id"],
                        "start_element_id": row["start_element_id"],
                        "end_element_id": row["end_element_id"],
                        "type": row["type"],
                        "properties": dict(row["properties"] or {}),
                    }
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    rel_count += 1

        return {
            "status": "connected",
            "node_count": node_count,
            "relationship_count": rel_count,
            "nodes_file": str(nodes_path.relative_to(OUT)),
            "relationships_file": str(rels_path.relative_to(OUT)),
        }
    finally:
        driver.close()


def build_readme(manifest: dict[str, Any]) -> str:
    return f"""# Refinery Knowledge Package

Generated: {manifest["generated_at"]}

This package is a reproducible snapshot of the current refinery MVP
knowledge artifacts.

## Contents

- `ontology/` domain and ontology candidates
- `graph/` Neo4j graph export
- `documents/` PyMuPDF chunks, embeddings and document/entity links
- `timeseries/` sensor summaries and statistical analysis
- `events/` candidate statistical events
- `provenance/` source/evidence mappings
- `validation/` validation artifacts
- `metadata/` package metadata and generation details

## Evidence boundary

This package preserves candidate status and provenance from the upstream
pipeline. A statistical event is not a confirmed engineering fault.
A document mention is not proof of physical asset assignment or causality.

## Intended next consumer

The next runtime/training stages can use this package to build grounded
evidence bundles before any model fine-tuning.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="refinery")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "industrial-ai"))
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Build the package without exporting the Neo4j graph.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the existing package directory.",
    )
    args = parser.parse_args()

    out = ROOT / "data" / "knowledge_package" / args.domain

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Package already exists: {out}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(out)

    print("=" * 64)
    print("KNOWLEDGE PACKAGE BUILDER")
    print("=" * 64)
    print(f"Domain   : {args.domain}")
    print(f"Output   : {out}")

    out.mkdir(parents=True, exist_ok=True)

    missing_required: list[str] = []
    files_manifest: list[dict[str, Any]] = []

    # Core JSON artifacts.
    for key, src in TEXT_JSON_INPUTS.items():
        if not src.exists():
            missing_required.append(str(src))
            continue

        target_dir = {
            "domain_hypothesis": out / "metadata",
            "ontology_candidate": out / "ontology",
            "validated_knowledge": out / "provenance",
            "validation_report": out / "validation",
        }[key]

        files_manifest.append(copy_file(src, target_dir / src.name))

    # Time-series artifacts.
    for key, src in TS_INPUTS.items():
        if not src.exists():
            missing_required.append(str(src))
            continue

        target_dir = out / ("events" if key == "event_candidates" else "timeseries")
        files_manifest.append(copy_file(src, target_dir / src.name))

    # Document artifacts.
    for key, src in DOC_INPUTS.items():
        if not src.exists():
            missing_required.append(str(src))
            continue
        files_manifest.append(copy_file(src, out / "documents" / src.name))

    if missing_required:
        print("\nMissing required artifacts:")
        for item in missing_required:
            print(f"  - {item}")
        raise FileNotFoundError(
            f"\nKnowledge package cannot be built until {len(missing_required)} "
            "required artifact(s) exist."
        )

    # Build a compact provenance index from validated knowledge.
    validated = load_json(TEXT_JSON_INPUTS["validated_knowledge"])
    write_json(
        out / "provenance" / "provenance_index.json",
        extract_provenance(validated),
    )
    files_manifest.append({
        "source": "derived from validated_knowledge.json",
        "package_path": "provenance/provenance_index.json",
        "size_bytes": (out / "provenance" / "provenance_index.json").stat().st_size,
        "sha256": sha256_file(out / "provenance" / "provenance_index.json"),
    })

    # Record embedding shape without duplicating the binary into another file.
    embeddings = np.load(DOC_INPUTS["embeddings"], mmap_mode="r")
    embedding_metadata = {
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "file": "documents/embeddings.npy",
    }
    write_json(out / "documents" / "embedding_metadata.json", embedding_metadata)

    # Neo4j graph export.
    graph_status: dict[str, Any]
    if args.skip_neo4j:
        graph_status = {"status": "skipped"}
    else:
        print("\nExporting Neo4j graph...")
        graph_status = export_neo4j(
            args.uri,
            args.user,
            args.password,
            out / "graph",
        )

        files_manifest.extend([
            {
                "source": "Neo4j",
                "package_path": "graph/graph_nodes.jsonl",
                "size_bytes": (out / "graph" / "graph_nodes.jsonl").stat().st_size,
                "sha256": sha256_file(out / "graph" / "graph_nodes.jsonl"),
            },
            {
                "source": "Neo4j",
                "package_path": "graph/graph_relationships.jsonl",
                "size_bytes": (out / "graph" / "graph_relationships.jsonl").stat().st_size,
                "sha256": sha256_file(out / "graph" / "graph_relationships.jsonl"),
            },
        ])

    # Summaries for quick consumption by downstream code.
    validation_report = load_json(TEXT_JSON_INPUTS["validation_report"])
    domain_hypothesis = load_json(TEXT_JSON_INPUTS["domain_hypothesis"])
    ontology = load_json(TEXT_JSON_INPUTS["ontology_candidate"])

    summary = {
        "generated_at": utc_now(),
        "domain": args.domain,
        "domain_hypothesis": {
            "candidate_domain": domain_hypothesis.get("candidate_domain"),
            "subdomain": domain_hypothesis.get("subdomain"),
            "confidence": domain_hypothesis.get("confidence"),
            "status": domain_hypothesis.get("status"),
        },
        "ontology": {
            "class_count": len(ontology.get("classes", [])),
            "relation_count": len(ontology.get("relations", [])),
            "entity_candidate_count": len(ontology.get("entity_candidates", [])),
        },
        "validation": {
            "status": validation_report.get("status"),
            "errors": validation_report.get("errors"),
            "warnings": validation_report.get("warnings"),
        },
        "embeddings": embedding_metadata,
        "neo4j": graph_status,
    }

    write_json(out / "metadata" / "package_summary.json", summary)
    write_json(
        out / "manifest.json",
        {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "domain": args.domain,
            "source_root": str(PROCESSED.relative_to(ROOT)),
            "package_root": str(out.relative_to(ROOT)),
            "files": files_manifest,
            "neo4j": graph_status,
            "evidence_policy": {
                "statistical_event_is_not_confirmed_fault": True,
                "document_mention_is_not_physical_assignment": True,
                "correlation_is_not_causation": True,
            },
        },
    )

    manifest = load_json(out / "manifest.json")
    (out / "README.md").write_text(build_readme(manifest), encoding="utf-8")

    print("\nPackage summary")
    print("-" * 64)
    print(f"Files packaged       : {len(files_manifest)}")
    print(f"Validation status    : {summary['validation']['status']}")
    print(f"Ontology classes     : {summary['ontology']['class_count']}")
    print(f"Ontology relations   : {summary['ontology']['relation_count']}")
    print(f"Embedding shape      : {embedding_metadata['shape']}")
    print(f"Neo4j status         : {graph_status.get('status')}")
    if graph_status.get("node_count") is not None:
        print(f"Graph nodes          : {graph_status['node_count']}")
        print(f"Graph relationships  : {graph_status['relationship_count']}")
    print(f"\nCreated: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
