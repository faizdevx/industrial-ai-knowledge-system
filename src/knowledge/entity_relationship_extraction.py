"""
Stage 5 of the Industrial AI MVP:
    ontology_candidate.json + domain_discovery_input.json
                         ↓
             entity + relationship candidates
                         ↓
                  knowledge_objects.json

Important:
    - This stage produces CANDIDATES, not certified engineering facts.
    - Relationships that are not supported by the dataset are not invented.
    - Every object carries provenance back to the source workbook.

Usage:
    uv run python src/knowledge/entity_relationship_extraction.py

Expected files:
    data/processed/refinery/domain_discovery_input.json
    data/processed/refinery/ontology_candidate.json

Output:
    data/processed/refinery/knowledge_objects.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_INPUT = Path(
    "data/processed/refinery/domain_discovery_input.json"
)
DEFAULT_ONTOLOGY_INPUT = Path(
    "data/processed/refinery/ontology_candidate.json"
)
DEFAULT_OUTPUT = Path(
    "data/processed/refinery/knowledge_objects.json"
)


TAG_TO_SENSOR_CLASS = {
    "PI": "PressureSensor",
    "PDI": "DifferentialPressureSensor",
    "TI": "TemperatureSensor",
    "SI": "SpeedSensor",
    "XI": "PositionOrDisplacementSensor",
    "ZI": "PositionSensor",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_provenance(
    source_file: str,
    sheet: str,
    column: str,
    extraction_method: str,
    confidence: float,
    row_start: int | None = None,
    row_end: int | None = None,
) -> dict[str, Any]:
    location: dict[str, Any] = {
        "sheet": sheet,
        "column": column,
    }

    if row_start is not None and row_end is not None:
        location["data_rows"] = {
            "start": row_start,
            "end": row_end,
        }

    return {
        "source_id": source_file,
        "source_type": "xlsx",
        "location": location,
        "extraction_method": extraction_method,
        "confidence": confidence,
    }


def extract_sensor_entities(
    domain_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    source = domain_payload.get("source", {})
    dataset = domain_payload.get("dataset", {})
    sensors = domain_payload.get("sensor_evidence", [])

    source_file = str(source.get("file_name", "unknown.xlsx"))
    sheet = str(source.get("sheet", "Sheet1"))
    rows = int(dataset.get("rows", 0))

    entities: list[dict[str, Any]] = []

    for sensor in sensors:
        tag = str(sensor.get("tag"))
        family = sensor.get("tag_family_hint")
        sensor_class = TAG_TO_SENSOR_CLASS.get(
            family,
            "Sensor",
        )

        confidence = 0.70 if family else 0.40

        entity_id = make_id("sensor", tag)
        series_id = make_id("series", tag)

        provenance = make_provenance(
            source_file=source_file,
            sheet=sheet,
            column=tag,
            extraction_method="deterministic_tag_and_column_mapping",
            confidence=confidence,
            # Excel row 1 is assumed to be the header. Therefore data
            # begins at row 2 and ends at rows+1.
            row_start=2 if rows else None,
            row_end=rows + 1 if rows else None,
        )

        entities.append(
            {
                "id": entity_id,
                "entity_type": "Sensor",
                "label": tag,
                "candidate_class": sensor_class,
                "status": "candidate",
                "confidence": confidence,
                "attributes": {
                    "source_tag": tag,
                    "tag_family": family,
                    "data_type": sensor.get("data_type"),
                    "missing_fraction": sensor.get("missing_fraction"),
                    "unique_count": sensor.get("unique_count"),
                    "statistics": sensor.get("statistics"),
                },
                "provenance": provenance,
            }
        )

        # A time-series column is represented separately from the sensor
        # entity. This lets the graph store meaning while a time-series
        # store later keeps the actual numerical observations.
        entities.append(
            {
                "id": series_id,
                "entity_type": "MeasurementSeries",
                "label": f"{tag} measurement series",
                "candidate_class": "Measurement",
                "status": "candidate",
                "confidence": confidence,
                "attributes": {
                    "source_tag": tag,
                    "timestamp_column": "Timestamp",
                    "row_count": rows,
                },
                "provenance": provenance,
            }
        )

    return entities


def extract_structural_relations(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []

    for entity in entities:
        if entity["entity_type"] != "Sensor":
            continue

        sensor_id = entity["id"]
        class_name = entity["candidate_class"]

        relations.append(
            {
                "id": make_id(
                    "rel",
                    f"{sensor_id}|INSTANCE_OF|{class_name}",
                ),
                "source": sensor_id,
                "relation": "INSTANCE_OF",
                "target": class_name,
                "status": "candidate",
                "confidence": entity["confidence"],
                "provenance": entity["provenance"],
                "evidence_type": "dataset_structure",
            }
        )

        series_id = make_id(
            "series",
            str(entity["attributes"]["source_tag"]),
        )

        relations.append(
            {
                "id": make_id(
                    "rel",
                    f"{sensor_id}|HAS_MEASUREMENT|{series_id}",
                ),
                "source": sensor_id,
                "relation": "HAS_MEASUREMENT",
                "target": series_id,
                "status": "candidate",
                "confidence": 0.95,
                "provenance": entity["provenance"],
                "evidence_type": "dataset_structure",
            }
        )

    return relations


def build_knowledge_objects(
    domain_payload: dict[str, Any],
    ontology_payload: dict[str, Any],
) -> dict[str, Any]:
    entities = extract_sensor_entities(domain_payload)
    relations = extract_structural_relations(entities)

    # We deliberately do not create:
    #   Sensor -> Compressor
    #   Compressor -> Bearing
    #   Low pressure -> causes failure
    # because those facts are not established by the current workbook alone.

    return {
        "status": "candidate",
        "source": domain_payload.get("source"),
        "domain_hypothesis": ontology_payload.get(
            "domain_hypothesis"
        ),
        "entities": entities,
        "relations": relations,
        "not_inferred": [
            "asset_identity",
            "component_identity",
            "sensor_to_asset_assignment",
            "physical_units",
            "causal_relationships",
            "failure_events",
        ],
        "validation_required": [
            "Validate tag semantics against technical documentation.",
            "Validate which physical asset each signal belongs to.",
            "Validate engineering units and operating limits.",
            "Validate any causal or failure-related relationship before graph promotion.",
        ],
        "next_stage": "provenance_validation_and_graph_build",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract candidate entities and relations."
    )

    parser.add_argument(
        "--domain-input",
        type=Path,
        default=DEFAULT_DOMAIN_INPUT,
    )
    parser.add_argument(
        "--ontology-input",
        type=Path,
        default=DEFAULT_ONTOLOGY_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    domain_payload = load_json(args.domain_input.resolve())
    ontology_payload = load_json(args.ontology_input.resolve())

    result = build_knowledge_objects(
        domain_payload=domain_payload,
        ontology_payload=ontology_payload,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    sensor_count = sum(
        1 for entity in result["entities"]
        if entity["entity_type"] == "Sensor"
    )

    series_count = sum(
        1 for entity in result["entities"]
        if entity["entity_type"] == "MeasurementSeries"
    )

    print("=" * 60)
    print("ENTITY + RELATIONSHIP EXTRACTION")
    print("=" * 60)
    print(f"Sensor entities       : {sensor_count}")
    print(f"Measurement series    : {series_count}")
    print(f"Relation candidates   : {len(result['relations'])}")
    print()
    print("Important exclusions:")
    for item in result["not_inferred"]:
        print(f"  - {item}")
    print()
    print(f"Created: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
