"""
Stage 3 of the MVP:
    domain_discovery_input.json
            ↓
    evidence-based domain hypothesis
            ↓
    domain_hypothesis.json

This is a deterministic baseline. It deliberately produces a hypothesis,
not a claimed truth. A local LLM can be added later to enrich/validate it.

Usage:
    uv run python src/discovery/domain_discovery.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/processed/refinery/domain_discovery_input.json"
)
DEFAULT_OUTPUT = Path(
    "data/processed/refinery/domain_hypothesis.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_evidence(
    evidence: list[dict[str, Any]],
    statement: str,
    strength: str,
) -> None:
    evidence.append(
        {
            "statement": statement,
            "strength": strength,
        }
    )


def infer_domain(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source", {})
    dataset = payload.get("dataset", {})
    tag_summary = payload.get("tag_family_summary", {})
    temporal = payload.get("temporal_evidence", {})

    file_name = str(source.get("file_name", "")).lower()

    evidence: list[dict[str, Any]] = []

    # Scores are intentionally conservative.
    scores = {
        "industrial_process": 0.0,
        "rotating_equipment": 0.0,
        "manufacturing": 0.0,
        "aerospace": 0.0,
    }

    # Evidence 1: industrial instrument-tag families.
    process_families = {"PI", "PDI", "TI", "XI", "ZI", "SI"}
    observed_families = set(tag_summary.keys())
    matched = sorted(process_families & observed_families)

    if matched:
        scores["industrial_process"] += 0.55
        add_evidence(
            evidence,
            (
                "The dataset contains industrial instrument-tag family "
                f"hints: {', '.join(matched)}."
            ),
            "medium",
        )

    # Evidence 2: many numeric signals + regular timestamped observations.
    numeric_count = int(
        payload.get("candidate_signals", {}).get(
            "numeric_column_count", 0
        )
    )

    if temporal.get("timestamp_column") and numeric_count >= 5:
        scores["industrial_process"] += 0.20
        scores["rotating_equipment"] += 0.10

        add_evidence(
            evidence,
            (
                f"The dataset contains {numeric_count} numeric measurement "
                "columns organized as timestamped observations."
            ),
            "medium",
        )

    # Evidence 3: source-name hint only.
    # We do NOT treat "RCSD" as proof of refinery/compressor semantics.
    if re.search(r"compress|rotat|rcsd", file_name):
        scores["rotating_equipment"] += 0.25

        add_evidence(
            evidence,
            (
                f"The source filename '{source.get('file_name')}' contains "
                "a rotating-equipment/compressor-related naming hint."
            ),
            "weak",
        )

    # Evidence 4: long continuous time range.
    if temporal.get("start") and temporal.get("end"):
        scores["industrial_process"] += 0.05
        add_evidence(
            evidence,
            "The source contains a continuous time-indexed measurement record.",
            "weak",
        )

    # Normalize.
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_domain, raw_score = ranked[0]

    # Keep the confidence deliberately capped because this is only discovery.
    confidence = round(min(raw_score, 0.85), 2)

    hypothesis = {
        "status": "hypothesis",
        "candidate_domain": best_domain,
        "candidate_subdomain": (
            "rotating_equipment"
            if best_domain == "rotating_equipment"
            else None
        ),
        "confidence": confidence,
        "evidence": evidence,
        "alternatives": [
            {
                "domain": domain,
                "score": round(score, 2),
            }
            for domain, score in ranked[1:]
            if score > 0
        ],
        "dataset_summary": {
            "file_name": source.get("file_name"),
            "rows": dataset.get("rows"),
            "columns": dataset.get("columns"),
            "tag_families": tag_summary,
        },
        "next_stage": "ontology_discovery",
    }

    return hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an evidence-based domain hypothesis."
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

    args = parser.parse_args()

    payload = load_json(args.input.resolve())
    hypothesis = infer_domain(payload)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            hypothesis,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("DOMAIN DISCOVERY")
    print("=" * 60)
    print(f"Candidate domain : {hypothesis['candidate_domain']}")
    print(f"Subdomain        : {hypothesis['candidate_subdomain']}")
    print(f"Confidence       : {hypothesis['confidence']}")
    print("Status           : hypothesis")
    print()
    print("Evidence:")
    for item in hypothesis["evidence"]:
        print(f"  [{item['strength']}] {item['statement']}")
    print()
    print(f"Created: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
