"""
Validate the grounded training-example candidates before train/val/test split.

Checks:
  - required schema fields
  - non-empty instruction/response
  - source records present for grounded examples
  - referenced document chunks exist
  - referenced entity ids exist in validated knowledge
  - referenced event ids exist in event candidates
  - hard-negative policy statements are present
  - duplicate example IDs
  - duplicate instruction/response pairs

Outputs:
  data/processed/refinery/training/
    training_validation_results.jsonl
    training_validation_report.json

No examples are silently modified. Invalid examples are reported so they can
be fixed upstream.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "data" / "knowledge_package" / "refinery"
DEFAULT_TRAINING = ROOT / "data" / "processed" / "refinery" / "training"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_event_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("events") or value.get("event_candidates") or []
    return []


def validate_example(
    item: dict[str, Any],
    valid_chunks: set[str],
    valid_entities: set[str],
    valid_events: set[str],
) -> list[str]:
    errors: list[str] = []

    required = [
        "id",
        "schema_version",
        "task_type",
        "instruction",
        "response",
        "context",
        "source_records",
        "grounding_policy",
    ]

    for field in required:
        if field not in item:
            errors.append(f"missing_field:{field}")

    if not str(item.get("id", "")).strip():
        errors.append("empty_id")

    if not str(item.get("instruction", "")).strip():
        errors.append("empty_instruction")

    if not str(item.get("response", "")).strip():
        errors.append("empty_response")

    if not isinstance(item.get("context"), list):
        errors.append("context_not_list")

    if not isinstance(item.get("source_records"), list):
        errors.append("source_records_not_list")

    policy = item.get("grounding_policy", {})
    if not isinstance(policy, dict):
        errors.append("grounding_policy_not_object")
    else:
        for key in (
            "source_grounded",
            "statistical_event_is_not_confirmed_fault",
            "document_mention_is_not_physical_assignment",
            "correlation_is_not_causation",
        ):
            if key not in policy:
                errors.append(f"missing_policy:{key}")

    if not item.get("source_records"):
        errors.append("no_source_records")

    for record in item.get("context", []):
        if not isinstance(record, dict):
            continue

        chunk_id = record.get("chunk_id")
        if chunk_id and chunk_id not in valid_chunks:
            errors.append(f"unknown_chunk:{chunk_id}")

        entity_id = record.get("entity_id")
        if entity_id and entity_id not in valid_entities:
            errors.append(f"unknown_entity:{entity_id}")

        event_id = record.get("event_id")
        if event_id and event_id not in valid_events:
            errors.append(f"unknown_event:{event_id}")

    task = str(item.get("task_type", ""))

    if task == "timeseries_event_interpretation":
        response = str(item.get("response", "")).lower()
        if "does not" not in response or "confirmed" not in response:
            errors.append("event_response_missing_uncertainty_guard")

    if task == "uncertainty_reasoning":
        response = str(item.get("response", "")).lower()
        uncertainty_terms = (
            "no.",
            "not",
            "insufficient",
            "cannot",
            "does not",
        )
        if not any(term in response for term in uncertainty_terms):
            errors.append("uncertainty_example_may_be_overconfident")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--training", default=str(DEFAULT_TRAINING))
    args = parser.parse_args()

    package = Path(args.package)
    training = Path(args.training)

    examples_path = training / "training_examples.jsonl"
    chunks_path = package / "documents" / "document_chunks.jsonl"
    validated_path = package / "provenance" / "validated_knowledge.json"
    events_path = package / "events" / "event_candidates.json"

    required = [
        examples_path,
        chunks_path,
        validated_path,
        events_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        raise FileNotFoundError(
            "Generate the training candidates and build the knowledge package first."
        )

    examples = load_jsonl(examples_path)
    chunks = load_jsonl(chunks_path)
    validated = load_json(validated_path)
    events = get_event_records(load_json(events_path))

    valid_chunks = {
        str(item.get("chunk_id"))
        for item in chunks
        if item.get("chunk_id")
    }
    valid_entities = {
        str(item.get("id"))
        for item in validated.get("entities", [])
        if item.get("id")
    }
    valid_events = {
        str(item.get("id") or item.get("event_id"))
        for item in events
        if item.get("id") or item.get("event_id")
    }

    id_counts = Counter(str(x.get("id")) for x in examples)
    pair_counts = Counter(
        (
            str(x.get("instruction", "")).strip().lower(),
            str(x.get("response", "")).strip().lower(),
        )
        for x in examples
    )

    results = []
    valid_count = 0
    invalid_count = 0

    for item in examples:
        errors = validate_example(
            item,
            valid_chunks,
            valid_entities,
            valid_events,
        )

        example_id = str(item.get("id", ""))
        duplicate_id = id_counts[example_id] > 1
        duplicate_pair = pair_counts[
            (
                str(item.get("instruction", "")).strip().lower(),
                str(item.get("response", "")).strip().lower(),
            )
        ] > 1

        if duplicate_id:
            errors.append("duplicate_example_id")
        if duplicate_pair:
            errors.append("duplicate_instruction_response")

        status = "passed" if not errors else "failed"

        if status == "passed":
            valid_count += 1
        else:
            invalid_count += 1

        results.append({
            "id": example_id,
            "task_type": item.get("task_type"),
            "difficulty": item.get("difficulty"),
            "status": status,
            "errors": errors,
        })

    training.mkdir(parents=True, exist_ok=True)

    results_path = training / "training_validation_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    task_counts = Counter(
        str(x.get("task_type"))
        for x in examples
    )
    failed_task_counts = Counter(
        str(x.get("task_type"))
        for x, result in zip(examples, results)
        if result["status"] == "failed"
    )

    report = {
        "schema_version": "1.0",
        "status": "passed" if invalid_count == 0 else "failed",
        "total_examples": len(examples),
        "valid_examples": valid_count,
        "invalid_examples": invalid_count,
        "validation_rates": {
            "example_pass_rate": (
                valid_count / len(examples)
                if examples else 0.0
            )
        },
        "counts_by_task": dict(task_counts),
        "failed_counts_by_task": dict(failed_task_counts),
        "duplicate_example_ids": [
            key for key, count in id_counts.items()
            if count > 1
        ],
        "design_policy": {
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
        },
        "note": (
            "Passing this validator does not establish engineering truth. "
            "Human review is still required before fine-tuning."
        ),
    }

    report_path = training / "training_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("TRAINING DATA VALIDATION")
    print("=" * 72)
    print(f"Total examples : {report['total_examples']}")
    print(f"Valid          : {report['valid_examples']}")
    print(f"Invalid        : {report['invalid_examples']}")
    print(
        f"Pass rate      : "
        f"{report['validation_rates']['example_pass_rate']:.2%}"
    )
    print(f"Status         : {report['status']}")
    print()
    print(f"Results : {results_path}")
    print(f"Report  : {report_path}")
    print()
    if invalid_count:
        print(
            "Fix or remove failed examples before creating the train/val/test split."
        )
    else:
        print(
            "Schema/reference checks passed. Human review is still required."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
