"""
Stage 4 of the Industrial AI MVP:
    Domain hypothesis + sensor evidence
                ↓
       candidate ontology
                ↓
       entity candidates
       sensor classes
       relations
       provenance requirements

This stage deliberately produces a CANDIDATE ontology.
It does not claim that every inferred class/relation is factually true.

Usage:
    uv run python src/discovery/ontology_discovery.py

Expected input:
    data/processed/refinery/domain_hypothesis.json
    data/processed/refinery/domain_discovery_input.json

Output:
    data/processed/refinery/ontology_candidate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_INPUT = Path(
    "data/processed/refinery/domain_discovery_input.json"
)
DEFAULT_HYPOTHESIS_INPUT = Path(
    "data/processed/refinery/domain_hypothesis.json"
)
DEFAULT_OUTPUT = Path(
    "data/processed/refinery/ontology_candidate.json"
)


CORE_CLASSES = [
    {
        "name": "Asset",
        "description": "A physical industrial asset being monitored or operated.",
        "source": "project_universal_core",
    },
    {
        "name": "Component",
        "description": "A physical sub-part of an asset.",
        "source": "project_universal_core",
    },
    {
        "name": "Subsystem",
        "description": "A functional subsystem associated with an asset.",
        "source": "project_universal_core",
    },
    {
        "name": "Sensor",
        "description": "An instrument or signal used to observe a physical quantity.",
        "source": "project_universal_core",
    },
    {
        "name": "Measurement",
        "description": "A value produced by a sensor at a point in time.",
        "source": "project_universal_core",
    },
    {
        "name": "Process",
        "description": "An industrial process or operating activity.",
        "source": "project_universal_core",
    },
    {
        "name": "Event",
        "description": "A temporally bounded event derived from observations or records.",
        "source": "project_universal_core",
    },
    {
        "name": "Condition",
        "description": "An observed operating or equipment condition.",
        "source": "project_universal_core",
    },
    {
        "name": "Maintenance",
        "description": "A maintenance activity or maintenance record.",
        "source": "project_universal_core",
    },
    {
        "name": "Document",
        "description": "A source document containing technical or operational knowledge.",
        "source": "project_universal_core",
    },
]


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


def make_sensor_classes(
    sensor_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observed_families = sorted(
        {
            sensor.get("tag_family_hint")
            for sensor in sensor_evidence
            if sensor.get("tag_family_hint")
        }
    )

    classes = []

    for family in observed_families:
        class_name = TAG_TO_SENSOR_CLASS.get(
            family,
            f"{family}Sensor",
        )

        classes.append(
            {
                "name": class_name,
                "parent": "Sensor",
                "tag_family": family,
                "status": "candidate",
                "confidence": 0.70,
                "evidence": (
                    f"Observed {family} industrial instrument-tag family "
                    "in the dataset."
                ),
                "validation_required": True,
            }
        )

    return classes


def make_domain_classes(
    domain: str | None,
    subdomain: str | None,
) -> list[dict[str, Any]]:
    classes: list[dict[str, Any]] = []

    # The source architecture uses rotating equipment examples such as
    # Compressor/Bearing/LubricationSystem. We keep these as candidates,
    # because the current stage is ontology induction rather than truth
    # certification.
    if domain == "industrial_process":
        classes.extend(
            [
                {
                    "name": "IndustrialEquipment",
                    "parent": "Asset",
                    "status": "candidate",
                    "confidence": 0.75,
                    "evidence": (
                        "Dataset contains multiple industrial instrument-tag "
                        "families and timestamped measurements."
                    ),
                },
                {
                    "name": "IndustrialMeasurement",
                    "parent": "Measurement",
                    "status": "candidate",
                    "confidence": 0.75,
                    "evidence": (
                        "Dataset contains 25 numeric measurement columns "
                        "attached to a timestamp."
                    ),
                },
            ]
        )

    if subdomain == "rotating_equipment":
        classes.extend(
            [
                {
                    "name": "RotatingEquipment",
                    "parent": "IndustrialEquipment",
                    "status": "candidate",
                    "confidence": 0.60,
                    "evidence": (
                        "The domain hypothesis includes a rotating-equipment "
                        "alternative and the source filename contains RCSD."
                    ),
                },
                {
                    "name": "Compressor",
                    "parent": "RotatingEquipment",
                    "status": "candidate",
                    "confidence": 0.55,
                    "evidence": (
                        "Compressor is an example equipment class in the "
                        "project's refinery use case."
                    ),
                },
                {
                    "name": "Bearing",
                    "parent": "Component",
                    "status": "candidate",
                    "confidence": 0.50,
                    "evidence": (
                        "Bearing is a component class in the project's "
                        "compressor example."
                    ),
                },
                {
                    "name": "LubricationSystem",
                    "parent": "Subsystem",
                    "status": "candidate",
                    "confidence": 0.50,
                    "evidence": (
                        "Lubrication system is a subsystem class in the "
                        "project's compressor example."
                    ),
                },
            ]
        )

    return classes


def make_relations() -> list[dict[str, Any]]:
    return [
        {
            "name": "HAS_COMPONENT",
            "domain": "Asset",
            "range": "Component",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "HAS_SUBSYSTEM",
            "domain": "Asset",
            "range": "Subsystem",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "MONITORED_BY",
            "domain": "Asset",
            "range": "Sensor",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "MEASURES",
            "domain": "Sensor",
            "range": "Measurement",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "HAS_MEASUREMENT",
            "domain": "Asset",
            "range": "Measurement",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "OBSERVED_DURING",
            "domain": "Measurement",
            "range": "Event",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "HAS_CONDITION",
            "domain": "Asset",
            "range": "Condition",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "DOCUMENTED_BY",
            "domain": "Asset",
            "range": "Document",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "RELATED_TO",
            "domain": "Event",
            "range": "Event",
            "status": "candidate",
            "requires_evidence": True,
        },
        {
            "name": "HAS_MAINTENANCE",
            "domain": "Asset",
            "range": "Maintenance",
            "status": "candidate",
            "requires_evidence": True,
        },
    ]


def make_entity_candidates(
    sensor_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    # The current dataset provides strong evidence for sensors/signals,
    # but it does not directly expose a confirmed asset identifier.
    for sensor in sensor_evidence:
        tag = sensor.get("tag")
        family = sensor.get("tag_family_hint")

        entities.append(
            {
                "candidate_id": tag,
                "class": TAG_TO_SENSOR_CLASS.get(
                    family,
                    "Sensor",
                ),
                "source_tag": tag,
                "status": "candidate",
                "confidence": 0.70 if family else 0.40,
                "evidence": (
                    f"Observed source signal tag '{tag}' in Sheet1."
                ),
                "requires_validation": True,
            }
        )

    return entities


def build_ontology(
    domain_payload: dict[str, Any],
    hypothesis_payload: dict[str, Any],
) -> dict[str, Any]:
    sensor_evidence = domain_payload.get("sensor_evidence", [])

    candidate_domain = hypothesis_payload.get("candidate_domain")
    candidate_subdomain = hypothesis_payload.get("candidate_subdomain")

    classes = list(CORE_CLASSES)

    sensor_classes = make_sensor_classes(sensor_evidence)
    domain_classes = make_domain_classes(
        candidate_domain,
        candidate_subdomain,
    )

    # Avoid accidental duplicate class names.
    seen = {item["name"] for item in classes}
    for item in sensor_classes + domain_classes:
        if item["name"] not in seen:
            classes.append(item)
            seen.add(item["name"])

    return {
        "status": "candidate",
        "purpose": (
            "Candidate ontology induced from dataset evidence and the "
            "project's universal industrial knowledge-model core."
        ),
        "domain_hypothesis": {
            "domain": candidate_domain,
            "subdomain": candidate_subdomain,
            "confidence": hypothesis_payload.get("confidence"),
        },
        "classes": classes,
        "relations": make_relations(),
        "entity_candidates": make_entity_candidates(sensor_evidence),
        "provenance_requirements": {
            "required_for_entity": True,
            "required_for_relation": True,
            "required_for_measurement": True,
            "required_fields": [
                "source_id",
                "source_type",
                "location",
                "extraction_method",
                "confidence",
            ],
        },
        "validation": {
            "requires_human_validation": True,
            "why": [
                "Tag-family semantics are treated as hypotheses.",
                "Equipment/component classes are candidates, not confirmed facts.",
                "Relationships require source evidence before becoming trusted graph facts.",
            ],
        },
        "next_stage": "entity_and_relationship_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a candidate ontology from domain-discovery evidence."
    )
    parser.add_argument(
        "--domain-input",
        type=Path,
        default=DEFAULT_DOMAIN_INPUT,
    )
    parser.add_argument(
        "--hypothesis-input",
        type=Path,
        default=DEFAULT_HYPOTHESIS_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    domain_payload = load_json(args.domain_input.resolve())
    hypothesis_payload = load_json(args.hypothesis_input.resolve())

    ontology = build_ontology(
        domain_payload=domain_payload,
        hypothesis_payload=hypothesis_payload,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            ontology,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("ONTOLOGY DISCOVERY")
    print("=" * 60)
    print(f"Domain     : {ontology['domain_hypothesis']['domain']}")
    print(f"Subdomain  : {ontology['domain_hypothesis']['subdomain']}")
    print(f"Classes    : {len(ontology['classes'])}")
    print(f"Relations  : {len(ontology['relations'])}")
    print(f"Entities   : {len(ontology['entity_candidates'])}")
    print()
    print("Candidate sensor classes:")
    for item in make_sensor_classes(
        domain_payload.get("sensor_evidence", [])
    ):
        print(
            f"  {item['name']} "
            f"<-- {item['tag_family']} "
            f"(confidence={item['confidence']})"
        )

    print()
    print(f"Created: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
