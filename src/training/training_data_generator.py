"""
Training data generator for the Industrial AI Knowledge System MVP.

Purpose:
    Convert validated knowledge artifacts into a small, grounded instruction
    dataset. This is NOT synthetic "knowledge". It creates examples whose
    answers are derived directly from packaged artifacts.

Inputs:
    data/knowledge_package/refinery/
        ontology/ontology_candidate.json
        provenance/validated_knowledge.json
        provenance/provenance_index.json
        documents/document_chunks.jsonl
        documents/document_entity_links.jsonl
        events/event_candidates.json
        timeseries/sensor_timeseries_summary.csv

Output:
    data/processed/refinery/training/
        training_examples.jsonl
        training_summary.json

The generator includes:
    - entity/class identification
    - provenance questions
    - document evidence questions
    - statistical-event interpretation
    - uncertainty / hard-negative examples

Important:
    A statistical anomaly is not a confirmed fault.
    A document mention is not proof of physical asset assignment.
    Correlation is not causation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PACKAGE = ROOT / "data" / "knowledge_package" / "refinery"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "refinery" / "training"

ONTOLOGY = DEFAULT_PACKAGE / "ontology" / "ontology_candidate.json"
VALIDATED = DEFAULT_PACKAGE / "provenance" / "validated_knowledge.json"
PROVENANCE = DEFAULT_PACKAGE / "provenance" / "provenance_index.json"
DOC_CHUNKS = DEFAULT_PACKAGE / "documents" / "document_chunks.jsonl"
DOC_LINKS = DEFAULT_PACKAGE / "documents" / "document_entity_links.jsonl"
EVENTS = DEFAULT_PACKAGE / "events" / "event_candidates.json"
SENSOR_SUMMARY = DEFAULT_PACKAGE / "timeseries" / "sensor_timeseries_summary.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(clean_text(x) for x in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def example(
    example_id: str,
    task_type: str,
    question: str,
    answer: str,
    context: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    difficulty: str = "standard",
) -> dict[str, Any]:
    return {
        "id": example_id,
        "schema_version": "1.0",
        "task_type": task_type,
        "difficulty": difficulty,
        "instruction": question,
        "response": answer,
        "context": context,
        "source_records": source_records,
        "grounding_policy": {
            "source_grounded": True,
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
        },
    }


def generate_entity_examples(validated: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []

    entities = validated.get("entities", [])
    for entity in entities:
        entity_id = clean_text(entity.get("id"))
        source_column = clean_text(entity.get("source_column"))
        candidate_class = clean_text(entity.get("candidate_class"))
        confidence = entity.get("confidence")

        if not entity_id:
            continue

        question = f"What class was proposed for sensor entity {entity_id}?"
        answer = (
            f"The candidate class recorded for {entity_id} is "
            f"{candidate_class or 'not specified'}."
        )
        context = [{
            "type": "entity",
            "id": entity_id,
            "source_column": source_column,
            "candidate_class": candidate_class,
            "confidence": confidence,
        }]
        source = [{
            "artifact": "validated_knowledge.json",
            "entity_id": entity_id,
            "source_column": source_column,
        }]

        examples.append(
            example(
                stable_id("entity_class", entity_id),
                "entity_classification",
                question,
                answer,
                context,
                source,
            )
        )

        source_parts = []
        if entity.get("source_id"):
            source_parts.append(f"source {entity['source_id']}")
        if entity.get("source_sheet"):
            source_parts.append(f"sheet {entity['source_sheet']}")
        if entity.get("source_column"):
            source_parts.append(f"column {entity['source_column']}")
        if entity.get("source_row_start") is not None:
            source_parts.append(f"rows {entity['source_row_start']}")
            if entity.get("source_row_end") is not None:
                source_parts[-1] += f"–{entity['source_row_end']}"

        source_text = ", ".join(source_parts) or "no source details recorded"

        examples.append(
            example(
                stable_id("entity_provenance", entity_id),
                "provenance",
                f"Where did the knowledge system obtain evidence for {entity_id}?",
                f"The recorded provenance points to {source_text}.",
                context,
                source,
            )
        )

    return examples


def generate_document_examples(
    chunks: list[dict[str, Any]],
    links: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    link_by_chunk = {
        clean_text(x.get("chunk_id")): x
        for x in links
        if x.get("chunk_id")
    }

    for chunk in chunks[:limit]:
        text = clean_text(chunk.get("text"))
        if not text:
            continue

        chunk_id = clean_text(chunk.get("chunk_id"))
        file_name = clean_text(chunk.get("file_name") or chunk.get("file"))
        page = clean_text(chunk.get("page"))
        tags = link_by_chunk.get(chunk_id, {}).get("matched_sensor_tags", [])

        citation = f"{file_name} p.{page} chunk:{chunk_id}"

        question = (
            f"What does the retrieved document evidence at {file_name} page "
            f"{page} say?"
        )

        # Keep the answer as a source excerpt, not a new interpretation.
        excerpt = " ".join(text.split())
        if len(excerpt) > 700:
            excerpt = excerpt[:700].rsplit(" ", 1)[0] + "..."

        answer = (
            f"The retrieved evidence from {citation} states: {excerpt}"
        )

        examples.append(
            example(
                stable_id("doc_evidence", chunk_id),
                "document_evidence",
                question,
                answer,
                [{
                    "type": "document_chunk",
                    "chunk_id": chunk_id,
                    "file": file_name,
                    "page": page,
                    "text": text,
                    "mentioned_sensor_tags": tags,
                }],
                [{
                    "artifact": "document_chunks.jsonl",
                    "chunk_id": chunk_id,
                    "file": file_name,
                    "page": page,
                }],
            )
        )

    return examples


def generate_event_examples(
    events: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    examples = []

    for event in events[:limit]:
        event_id = clean_text(event.get("id") or event.get("event_id"))
        sensor = clean_text(event.get("sensor_tag") or event.get("entity_id"))
        event_type = clean_text(event.get("event_type"))
        status = clean_text(event.get("status"))
        interpretation = clean_text(event.get("interpretation"))
        start = clean_text(event.get("start"))
        end = clean_text(event.get("end"))
        peak = event.get("peak_value")
        robust_z = event.get("peak_robust_z")

        if not event_id:
            continue

        answer_parts = []
        if event_type:
            answer_parts.append(f"The recorded event type is {event_type}.")
        if sensor:
            answer_parts.append(f"The associated sensor is {sensor}.")
        if start or end:
            answer_parts.append(
                f"The recorded time window is {start or 'unspecified'} "
                f"to {end or 'unspecified'}."
            )
        if peak is not None:
            answer_parts.append(f"The recorded peak value is {peak}.")
        if robust_z is not None:
            answer_parts.append(f"The recorded peak robust z-score is {robust_z}.")
        if interpretation:
            answer_parts.append(f"Recorded interpretation: {interpretation}")
        if status:
            answer_parts.append(f"Status: {status}.")

        answer = " ".join(answer_parts)
        answer += (
            " This is a statistical candidate event and does not by itself "
            "establish a confirmed equipment fault."
        )

        examples.append(
            example(
                stable_id("event", event_id),
                "timeseries_event_interpretation",
                f"What does candidate event {event_id} tell us?",
                answer,
                [{
                    "type": "event",
                    "event_id": event_id,
                    "sensor": sensor,
                    "event_type": event_type,
                    "status": status,
                    "start": start,
                    "end": end,
                    "peak_value": peak,
                    "peak_robust_z": robust_z,
                    "interpretation": interpretation,
                }],
                [{
                    "artifact": "event_candidates.json",
                    "event_id": event_id,
                }],
            )
        )

    return examples


def generate_hard_negative_examples(
    validated: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples = []

    entities = validated.get("entities", [])
    first_entity = entities[0] if entities else {}
    sensor = clean_text(first_entity.get("id") or first_entity.get("source_column"))

    event = events[0] if events else {}
    event_id = clean_text(event.get("id") or event.get("event_id"))

    examples.append(
        example(
            "hard_negative_failure",
            "uncertainty_reasoning",
            "Does a statistical anomaly prove that the equipment has failed?",
            "No. A statistical anomaly is evidence of unusual behavior in the "
            "measured signal. It does not by itself prove equipment failure. "
            "Additional engineering evidence and validation are required.",
            [{
                "type": "policy",
                "rule": "statistical_event_is_not_confirmed_fault",
                "event_id": event_id,
                "sensor": sensor,
            }],
            [{
                "artifact": "event_candidates.json",
                "event_id": event_id,
            }] if event_id else [],
            difficulty="hard",
        )
    )

    examples.append(
        example(
            "hard_negative_assignment",
            "uncertainty_reasoning",
            "Does a document mentioning a sensor tag prove which physical asset it belongs to?",
            "No. A document mention establishes textual evidence that the tag appears "
            "in that document. It does not by itself prove physical asset assignment "
            "or component connectivity.",
            [{
                "type": "policy",
                "rule": "document_mention_is_not_physical_assignment",
            }],
            [{
                "artifact": "document_entity_links.jsonl",
                "method": "exact_text_match",
            }],
            difficulty="hard",
        )
    )

    examples.append(
        example(
            "hard_negative_causality",
            "uncertainty_reasoning",
            "If two sensor signals are correlated, can we conclude that one caused the other?",
            "No. Correlation alone does not establish causation. A causal conclusion "
            "requires additional evidence.",
            [{
                "type": "policy",
                "rule": "correlation_is_not_causation",
            }],
            [{
                "artifact": "correlated_sensor_pairs.csv",
            }],
            difficulty="hard",
        )
    )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--document-limit",
        type=int,
        default=25,
        help="Maximum number of document examples to generate.",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=50,
        help="Maximum number of event examples to generate.",
    )
    args = parser.parse_args()

    package = Path(args.package)
    output = Path(args.output)

    ontology_path = package / "ontology" / "ontology_candidate.json"
    validated_path = package / "provenance" / "validated_knowledge.json"
    provenance_path = package / "provenance" / "provenance_index.json"
    chunks_path = package / "documents" / "document_chunks.jsonl"
    links_path = package / "documents" / "document_entity_links.jsonl"
    events_path = package / "events" / "event_candidates.json"

    required = [
        ontology_path,
        validated_path,
        provenance_path,
        chunks_path,
        links_path,
        events_path,
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required knowledge-package artifacts:")
        for item in missing:
            print(f"  - {item}")
        raise FileNotFoundError(
            "Build/verify the knowledge package and document linking first."
        )

    ontology = load_json(ontology_path)
    validated = load_json(validated_path)
    _provenance = load_json(provenance_path)
    chunks = load_jsonl(chunks_path)
    links = load_jsonl(links_path)
    events = load_json(events_path)

    # event_candidates.json may be either a list or a wrapper object.
    if isinstance(events, dict):
        events = (
            events.get("events")
            or events.get("event_candidates")
            or []
        )

    examples: list[dict[str, Any]] = []

    examples.extend(generate_entity_examples(validated))
    examples.extend(
        generate_document_examples(
            chunks,
            links,
            max(0, args.document_limit),
        )
    )
    examples.extend(
        generate_event_examples(
            events,
            max(0, args.event_limit),
        )
    )
    examples.extend(generate_hard_negative_examples(validated, events))

    # Use ontology only as metadata for the summary. No new facts are invented.
    class_count = len(ontology.get("classes", []))
    relation_count = len(ontology.get("relations", []))

    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "training_examples.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}

    for item in examples:
        by_type[item["task_type"]] = by_type.get(item["task_type"], 0) + 1
        by_difficulty[item["difficulty"]] = (
            by_difficulty.get(item["difficulty"], 0) + 1
        )

    summary = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "source_package": str(package),
        "total_examples": len(examples),
        "examples_by_task": by_type,
        "examples_by_difficulty": by_difficulty,
        "ontology": {
            "class_count": class_count,
            "relation_count": relation_count,
        },
        "design_notes": [
            "Examples are grounded in existing artifacts.",
            "No model-generated knowledge is used to create the initial examples.",
            "Statistical candidate events are not treated as confirmed faults.",
            "Document mention is not treated as physical asset assignment.",
            "Correlation is not treated as causation.",
            "Human review is required before using examples for fine-tuning.",
        ],
    }

    summary_path = output / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("TRAINING DATA GENERATOR")
    print("=" * 72)
    print(f"Knowledge package : {package}")
    print(f"Examples generated: {len(examples)}")
    print()
    print("By task type:")
    for key, value in sorted(by_type.items()):
        print(f"  {key}: {value}")
    print()
    print("By difficulty:")
    for key, value in sorted(by_difficulty.items()):
        print(f"  {key}: {value}")
    print()
    print(f"Created : {jsonl_path}")
    print(f"Created : {summary_path}")
    print()
    print("Important: review the generated examples before fine-tuning.")
    print("This stage creates grounded candidates, not ground truth.")
    print("=" * 72)


if __name__ == "__main__":
    main()
