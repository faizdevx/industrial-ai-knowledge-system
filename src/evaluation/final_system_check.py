"""
Final integration check for the Industrial AI Knowledge System MVP.

Checks:
  - required source/processed/knowledge-package files
  - document index shape
  - training/evaluation artifacts
  - Neo4j connectivity and basic node/relationship counts
  - Ollama availability
  - canonical runtime/application modules

This script does not modify project data.

Usage:
    uv run python src/evaluation/final_system_check.py
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed" / "refinery"
PACKAGE = ROOT / "data" / "knowledge_package" / "refinery"

REQUIRED_FILES = [
    PROCESSED / "workbook_profile.json",
    PROCESSED / "sensor_profile.csv",
    PROCESSED / "Sheet1.csv",
    PROCESSED / "domain_discovery_input.json",
    PROCESSED / "domain_hypothesis.json",
    PROCESSED / "ontology_candidate.json",
    PROCESSED / "knowledge_objects.json",
    PROCESSED / "validated_knowledge.json",
    PROCESSED / "validation_report.json",
    PROCESSED / "timeseries" / "timeseries_profile.json",
    PROCESSED / "timeseries" / "sensor_timeseries_summary.csv",
    PROCESSED / "timeseries" / "event_candidates.json",
    PROCESSED / "timeseries" / "correlated_sensor_pairs.csv",
    PROCESSED / "documents" / "document_catalog.json",
    PROCESSED / "documents" / "document_chunks.jsonl",
    PROCESSED / "documents" / "embeddings.npy",
    PROCESSED / "documents" / "chunk_metadata.jsonl",
    PROCESSED / "documents" / "embedding_index.json",
    PROCESSED / "documents" / "document_entity_links.jsonl",
    PROCESSED / "training" / "training_examples.jsonl",
    PROCESSED / "training" / "training_validation_report.json",
    PROCESSED / "training" / "splits" / "train.jsonl",
    PROCESSED / "training" / "splits" / "validation.jsonl",
    PROCESSED / "training" / "splits" / "test.jsonl",
    PROCESSED / "training" / "sft" / "train.jsonl",
    PROCESSED / "training" / "sft" / "validation.jsonl",
    PROCESSED / "training" / "sft" / "test.jsonl",
]

PACKAGE_FILES = [
    PACKAGE / "manifest.json",
    PACKAGE / "README.md",
    PACKAGE / "ontology" / "ontology_candidate.json",
    PACKAGE / "graph" / "graph_nodes.jsonl",
    PACKAGE / "graph" / "graph_relationships.jsonl",
    PACKAGE / "documents" / "document_chunks.jsonl",
    PACKAGE / "documents" / "embeddings.npy",
    PACKAGE / "documents" / "document_entity_links.jsonl",
    PACKAGE / "events" / "event_candidates.json",
    PACKAGE / "timeseries" / "sensor_timeseries_summary.csv",
    PACKAGE / "provenance" / "provenance_index.json",
    PACKAGE / "validation" / "validation_report.json",
]

RUNTIME_FILES = [
    ROOT / "src" / "retrieval" / "document_search.py",
    ROOT / "src" / "runtime" / "grounded_qa.py",
    ROOT / "src" / "runtime" / "local_rag.py",
    ROOT / "src" / "runtime" / "timeseries_tool.py",
    ROOT / "src" / "runtime" / "agent_runtime_v2.py",
    ROOT / "src" / "app" / "app_v3.py",
]

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def check_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    present = []
    missing = []
    for path in paths:
        if path.exists():
            present.append(path)
        else:
            missing.append(path)
    return present, missing


def check_neo4j() -> dict:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "industrial-ai")

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
    )

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            node_count = session.run(
                "MATCH (n) RETURN count(n) AS c"
            ).single()["c"]

            relationship_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]

            labels = session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN DISTINCT label
                ORDER BY label
                """
            )
            labels = [row["label"] for row in labels]

        return {
            "status": "passed",
            "nodes": int(node_count),
            "relationships": int(relationship_count),
            "labels": labels,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }
    finally:
        driver.close()


def check_ollama() -> dict:
    tags_url = OLLAMA_URL.replace(
        "/api/generate",
        "/api/tags",
    )

    try:
        with urllib.request.urlopen(
            tags_url,
            timeout=5,
        ) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )

        models = [
            item.get("name")
            for item in payload.get("models", [])
        ]

        wanted = os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:7b",
        )

        return {
            "status": "passed",
            "models": models,
            "configured_model": wanted,
            "configured_model_present": any(
                str(name).startswith(wanted.split(":")[0])
                for name in models
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }


def main() -> None:
    print("=" * 72)
    print("INDUSTRIAL AI KNOWLEDGE SYSTEM")
    print("FINAL MVP INTEGRATION CHECK")
    print("=" * 72)

    present, missing = check_paths(REQUIRED_FILES)

    package_present, package_missing = check_paths(PACKAGE_FILES)

    runtime_present, runtime_missing = check_paths(RUNTIME_FILES)

    print("\nCore artifacts")
    print("-" * 72)
    print(
        f"Present : {len(present)}/{len(REQUIRED_FILES)}"
    )
    print(
        f"Missing : {len(missing)}"
    )

    if missing:
        for p in missing:
            print(f"  MISSING: {p.relative_to(ROOT)}")

    print("\nKnowledge package")
    print("-" * 72)
    print(
        f"Present : {len(package_present)}/{len(PACKAGE_FILES)}"
    )
    print(
        f"Missing : {len(package_missing)}"
    )

    if package_missing:
        for p in package_missing:
            print(f"  MISSING: {p.relative_to(ROOT)}")

    print("\nRuntime/application")
    print("-" * 72)
    print(
        f"Present : {len(runtime_present)}/{len(RUNTIME_FILES)}"
    )
    print(
        f"Missing : {len(runtime_missing)}"
    )

    if runtime_missing:
        for p in runtime_missing:
            print(f"  MISSING: {p.relative_to(ROOT)}")

    print("\nArtifact consistency")
    print("-" * 72)

    consistency_errors = []

    embeddings_path = PROCESSED / "documents" / "embeddings.npy"
    metadata_path = PROCESSED / "documents" / "chunk_metadata.jsonl"

    if embeddings_path.exists() and metadata_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r")
        metadata_count = count_jsonl(metadata_path)

        print(
            f"Embeddings : {embeddings.shape}"
        )
        print(
            f"Metadata   : {metadata_count} rows"
        )

        if embeddings.shape[0] != metadata_count:
            consistency_errors.append(
                "embedding row count does not match chunk metadata count"
            )

    validated_path = PROCESSED / "validation_report.json"

    if validated_path.exists():
        report = load_json(validated_path)
        print(
            f"Knowledge validation: {report.get('status')}"
        )

        if report.get("status") != "passed":
            consistency_errors.append(
                "knowledge validation report is not passed"
            )

    training_validation_path = (
        PROCESSED
        / "training"
        / "training_validation_report.json"
    )

    if training_validation_path.exists():
        report = load_json(training_validation_path)
        print(
            f"Training validation: {report.get('status')}"
        )

        if report.get("invalid_examples", 0) != 0:
            consistency_errors.append(
                "training data contains invalid examples"
            )

    neo4j = check_neo4j()

    print("\nNeo4j")
    print("-" * 72)
    print(
        f"Status         : {neo4j['status']}"
    )

    if neo4j["status"] == "passed":
        print(
            f"Nodes          : {neo4j['nodes']}"
        )
        print(
            f"Relationships  : {neo4j['relationships']}"
        )
        print(
            f"Labels         : {', '.join(neo4j['labels'])}"
        )
    else:
        print(
            f"Error          : {neo4j['error']}"
        )

    ollama = check_ollama()

    print("\nOllama")
    print("-" * 72)
    print(
        f"Status         : {ollama['status']}"
    )

    if ollama["status"] == "passed":
        print(
            f"Configured     : {ollama['configured_model']}"
        )
        print(
            f"Model present  : {ollama['configured_model_present']}"
        )
        print(
            f"Models         : {', '.join(ollama['models'])}"
        )
    else:
        print(
            f"Error          : {ollama['error']}"
        )

    print("\nFinal status")
    print("-" * 72)

    passed = (
        not missing
        and not package_missing
        and not runtime_missing
        and not consistency_errors
        and neo4j["status"] == "passed"
        and ollama["status"] == "passed"
    )

    if passed:
        print("STATUS: PASSED")
        print()
        print("The refinery MVP has all required artifacts and runtime")
        print("dependencies visible to this integration check.")
    else:
        print("STATUS: FAILED")
        print()
        if consistency_errors:
            print("Consistency errors:")
            for error in consistency_errors:
                print(f"  - {error}")

        if neo4j["status"] != "passed":
            print("  - Neo4j connectivity failed.")

        if ollama["status"] != "passed":
            print("  - Ollama connectivity failed.")

    print("=" * 72)


if __name__ == "__main__":
    main()
