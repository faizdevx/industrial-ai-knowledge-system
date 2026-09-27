"""
Stage 9 of the Industrial AI MVP:
    event_candidates.json + Neo4j sensor graph
                      ↓
               Event validation
                      ↓
              Event nodes in Neo4j
                      ↓
         Event -> DETECTED_BY -> Sensor

This stage does NOT turn statistical events into faults.
It stores them explicitly as candidate events.

Inputs:
    data/processed/refinery/timeseries/event_candidates.json
    data/processed/refinery/validated_knowledge.json

Environment:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=...

Usage:
    uv add neo4j
    uv run python src/knowledge/event_graph_integration.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


DEFAULT_EVENTS = Path(
    "data/processed/refinery/timeseries/event_candidates.json"
)
DEFAULT_KNOWLEDGE = Path(
    "data/processed/refinery/validated_knowledge.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_id(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"event_{digest}"


def validate_event(event: dict[str, Any]) -> tuple[bool, str]:
    if not event.get("event_type"):
        return False, "missing event_type"

    if not event.get("sensor_tag"):
        return False, "missing sensor_tag"

    if event["event_type"] == "statistical_spike":
        if not event.get("start") or not event.get("end"):
            return False, "spike event missing start/end"

    if event["event_type"] == "change_point_candidate":
        if not event.get("timestamp"):
            return False, "change point missing timestamp"

    return True, ""


def build_sensor_map(
    knowledge: dict[str, Any],
) -> dict[str, str]:
    """
    source_tag -> Neo4j entity id
    """
    sensor_map: dict[str, str] = {}

    for entity in knowledge.get("entities", []):
        if entity.get("entity_type") != "Sensor":
            continue

        tag = entity.get("attributes", {}).get("source_tag")

        if tag:
            sensor_map[str(tag)] = str(entity["id"])

    return sensor_map


def event_properties(
    event: dict[str, Any],
    event_id: str,
) -> dict[str, Any]:
    """
    Keep event properties flat because these will become Neo4j properties.
    """
    properties: dict[str, Any] = {
        "id": event_id,
        "event_type": event.get("event_type"),
        "sensor_tag": event.get("sensor_tag"),
        "status": event.get("status", "candidate"),
        "interpretation": event.get("interpretation"),
        "source_id": "RCSD-1YD.xlsx",
        "source_type": "time_series_analysis",
        "analysis_method": "statistical",
        "analysis_artifact": "event_candidates.json",
    }

    for key in [
        "start",
        "end",
        "peak_timestamp",
        "timestamp",
        "direction",
        "sample_count",
        "peak_value",
        "peak_robust_z",
        "change_score",
    ]:
        if key in event:
            properties[key] = event[key]

    return properties


class EventGraphLoader:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ) -> None:
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
        )

    def close(self) -> None:
        self.driver.close()

    def create_constraint(self) -> None:
        query = """
        CREATE CONSTRAINT event_id_unique IF NOT EXISTS
        FOR (e:Event)
        REQUIRE e.id IS UNIQUE
        """

        with self.driver.session() as session:
            session.run(query).consume()

    def load_event(
        self,
        event: dict[str, Any],
        sensor_id: str,
        event_id: str,
    ) -> None:
        props = event_properties(event, event_id)

        query = """
        MATCH (s:Entity {id: $sensor_id})
        MERGE (e:Event {id: $id})
        SET
            e.event_type = $event_type,
            e.sensor_tag = $sensor_tag,
            e.status = $status,
            e.interpretation = $interpretation,
            e.source_id = $source_id,
            e.source_type = $source_type,
            e.analysis_method = $analysis_method,
            e.analysis_artifact = $analysis_artifact,
            e.start = $start,
            e.end = $end,
            e.peak_timestamp = $peak_timestamp,
            e.timestamp = $timestamp,
            e.direction = $direction,
            e.sample_count = $sample_count,
            e.peak_value = $peak_value,
            e.peak_robust_z = $peak_robust_z,
            e.change_score = $change_score
        MERGE (e)-[r:DETECTED_BY]->(s)
        SET
            r.status = 'candidate',
            r.source_id = 'RCSD-1YD.xlsx',
            r.evidence_type = 'time_series_analysis',
            r.analysis_artifact = 'event_candidates.json'
        """

        # Neo4j SET with missing parameters is fine, but all names above are
        # passed explicitly so the query remains deterministic.
        params = {
            "sensor_id": sensor_id,
            **props,
            "start": props.get("start"),
            "end": props.get("end"),
            "peak_timestamp": props.get("peak_timestamp"),
            "timestamp": props.get("timestamp"),
            "direction": props.get("direction"),
            "sample_count": props.get("sample_count"),
            "peak_value": props.get("peak_value"),
            "peak_robust_z": props.get("peak_robust_z"),
            "change_score": props.get("change_score"),
        }

        with self.driver.session() as session:
            session.run(query, **params).consume()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate time-series event candidates into Neo4j."
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=DEFAULT_KNOWLEDGE,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional event limit for testing. 0 = all events.",
    )

    args = parser.parse_args()

    events_payload = load_json(args.events.resolve())
    knowledge = load_json(args.knowledge.resolve())

    events = events_payload.get("events", [])
    sensor_map = build_sensor_map(knowledge)

    print("=" * 60)
    print("EVENT → KNOWLEDGE GRAPH INTEGRATION")
    print("=" * 60)
    print(f"Candidate events : {len(events)}")
    print(f"Known sensors    : {len(sensor_map)}")

    if args.limit > 0:
        events = events[:args.limit]
        print(f"Test limit       : {args.limit}")

    valid_events: list[dict[str, Any]] = []
    invalid_events: list[dict[str, Any]] = []
    unresolved_sensor_events: list[dict[str, Any]] = []

    seen_ids: set[str] = set()

    for event in events:
        valid, reason = validate_event(event)

        if not valid:
            invalid_events.append(
                {
                    "event": event,
                    "reason": reason,
                }
            )
            continue

        sensor_tag = str(event["sensor_tag"])
        sensor_id = sensor_map.get(sensor_tag)

        if sensor_id is None:
            unresolved_sensor_events.append(
                {
                    "event": event,
                    "reason": (
                        f"No sensor entity found for source tag "
                        f"'{sensor_tag}'."
                    ),
                }
            )
            continue

        event_id = stable_id(
            event.get("event_type"),
            sensor_tag,
            event.get("start"),
            event.get("end"),
            event.get("timestamp"),
            event.get("peak_timestamp"),
        )

        if event_id in seen_ids:
            continue

        seen_ids.add(event_id)

        valid_events.append(
            {
                "event": event,
                "event_id": event_id,
                "sensor_id": sensor_id,
            }
        )

    print(f"Valid events     : {len(valid_events)}")
    print(f"Invalid events   : {len(invalid_events)}")
    print(f"Unresolved       : {len(unresolved_sensor_events)}")

    if invalid_events:
        print("\nFirst validation problems:")
        for item in invalid_events[:5]:
            print(f"  - {item['reason']}")

    if unresolved_sensor_events:
        print("\nFirst unresolved sensors:")
        for item in unresolved_sensor_events[:5]:
            print(f"  - {item['reason']}")

    if invalid_events or unresolved_sensor_events:
        print(
            "\nNothing has been loaded yet because the event set "
            "contains unresolved candidates."
        )
        print(
            "Fix the issues or rerun after validating the sensor map."
        )
        print("=" * 60)
        raise SystemExit(1)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set."
        )

    loader = EventGraphLoader(
        uri=uri,
        user=user,
        password=password,
    )

    try:
        loader.create_constraint()

        for item in valid_events:
            loader.load_event(
                event=item["event"],
                sensor_id=item["sensor_id"],
                event_id=item["event_id"],
            )
    finally:
        loader.close()

    print()
    print(f"Loaded event nodes : {len(valid_events)}")
    print("Relationship       : Event -[:DETECTED_BY]-> Sensor")
    print("Status             : candidate")
    print("Important          : statistical anomaly ≠ confirmed fault")
    print()
    print(f"Neo4j URI          : {uri}")
    print("Result             : event graph integration complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
