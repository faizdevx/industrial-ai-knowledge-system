"""
Stage 7 of the Industrial AI MVP:
    validated_knowledge.json
                ↓
            Neo4j graph

The loader:
    - creates uniqueness constraints
    - creates Sensor / MeasurementSeries / Class nodes
    - loads candidate relationships
    - preserves provenance as node/relationship properties
    - is idempotent through MERGE
    - supports --validate-only so the JSON can be checked without Neo4j

Environment variables:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=change_me

Usage:
    uv add neo4j
    uv run python src/knowledge/neo4j_loader.py

Validate only:
    uv run python src/knowledge/neo4j_loader.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


DEFAULT_INPUT = Path(
    "data/processed/refinery/validated_knowledge.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_payload(payload: dict[str, Any]) -> None:
    """Fail fast on malformed input before touching Neo4j."""
    if not isinstance(payload, dict):
        raise ValueError("Knowledge payload must be a JSON object.")

    if "entities" not in payload:
        raise ValueError("Knowledge payload has no 'entities' field.")

    if "relations" not in payload:
        raise ValueError("Knowledge payload has no 'relations' field.")

    if not isinstance(payload["entities"], list):
        raise ValueError("'entities' must be a list.")

    if not isinstance(payload["relations"], list):
        raise ValueError("'relations' must be a list.")

    entity_ids = {
        str(entity.get("id"))
        for entity in payload["entities"]
        if entity.get("id")
    }

    for entity in payload["entities"]:
        if not entity.get("id"):
            raise ValueError("Entity missing id.")
        if not entity.get("entity_type"):
            raise ValueError(
                f"Entity {entity.get('id')} missing entity_type."
            )

    for relation in payload["relations"]:
        if not relation.get("id"):
            raise ValueError("Relation missing id.")
        if not relation.get("source"):
            raise ValueError(
                f"Relation {relation.get('id')} missing source."
            )
        if not relation.get("target"):
            raise ValueError(
                f"Relation {relation.get('id')} missing target."
            )
        if not relation.get("relation"):
            raise ValueError(
                f"Relation {relation.get('id')} missing relation."
            )

        source = relation["source"]
        # Class nodes are valid targets for INSTANCE_OF in this MVP.
        if (
            source not in entity_ids
            and relation["relation"] != "INSTANCE_OF"
        ):
            raise ValueError(
                f"Relation {relation['id']} references unknown source "
                f"entity '{source}'."
            )


def class_name_from_relation(relation: dict[str, Any]) -> str | None:
    if relation.get("relation") == "INSTANCE_OF":
        target = relation.get("target")
        if isinstance(target, str) and target:
            return target
    return None


def make_node_properties(entity: dict[str, Any]) -> dict[str, Any]:
    provenance = entity.get("provenance", {})
    location = provenance.get("location", {})

    return {
        "id": str(entity["id"]),
        "label": str(entity.get("label", entity["id"])),
        "entity_type": str(entity.get("entity_type", "Entity")),
        "candidate_class": str(
            entity.get("candidate_class", "Entity")
        ),
        "status": str(entity.get("status", "candidate")),
        "confidence": float(entity.get("confidence", 0.0)),
        "source_tag": entity.get("attributes", {}).get("source_tag"),
        "tag_family": entity.get("attributes", {}).get("tag_family"),
        "source_id": provenance.get("source_id"),
        "source_type": provenance.get("source_type"),
        "source_sheet": location.get("sheet"),
        "source_column": location.get("column"),
        "source_row_start": location.get("data_rows", {}).get("start"),
        "source_row_end": location.get("data_rows", {}).get("end"),
        "extraction_method": provenance.get("extraction_method"),
    }


def make_relation_properties(
    relation: dict[str, Any],
) -> dict[str, Any]:
    provenance = relation.get("provenance", {})
    location = provenance.get("location", {})

    return {
        "id": str(relation["id"]),
        "status": str(relation.get("status", "candidate")),
        "confidence": float(relation.get("confidence", 0.0)),
        "evidence_type": relation.get("evidence_type"),
        "source_id": provenance.get("source_id"),
        "source_type": provenance.get("source_type"),
        "source_sheet": location.get("sheet"),
        "source_column": location.get("column"),
        "source_row_start": location.get("data_rows", {}).get("start"),
        "source_row_end": location.get("data_rows", {}).get("end"),
        "extraction_method": provenance.get("extraction_method"),
    }


def sanitize_relationship_type(value: str) -> str:
    """
    Neo4j relationship types cannot be parameters, so they must be
    validated before interpolation into Cypher.
    """
    clean = value.strip().upper()

    allowed = {
        "INSTANCE_OF",
        "HAS_MEASUREMENT",
        "HAS_COMPONENT",
        "HAS_SUBSYSTEM",
        "MONITORED_BY",
        "MEASURES",
        "HAS_CONDITION",
        "DOCUMENTED_BY",
        "RELATED_TO",
        "HAS_MAINTENANCE",
    }

    if clean not in allowed:
        raise ValueError(
            f"Unsupported relationship type: {clean}"
        )

    return clean


class KnowledgeGraphLoader:
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

    def create_constraints(self) -> None:
        statements = [
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (n:Entity)
            REQUIRE n.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT class_name_unique IF NOT EXISTS
            FOR (n:Class)
            REQUIRE n.name IS UNIQUE
            """,
        ]

        with self.driver.session() as session:
            for statement in statements:
                session.run(statement).consume()

    def load_classes(
        self,
        class_names: set[str],
    ) -> None:
        if not class_names:
            return

        statement = """
        UNWIND $classes AS class_name
        MERGE (c:Class {name: class_name})
        ON CREATE SET
            c.status = 'candidate',
            c.source = 'ontology_candidate'
        """

        with self.driver.session() as session:
            session.run(
                statement,
                classes=sorted(class_names),
            ).consume()

    def load_entities(
        self,
        entities: list[dict[str, Any]],
    ) -> None:
        statement = """
        MERGE (n:Entity {id: $id})
        SET
            n.label = $label,
            n.entity_type = $entity_type,
            n.candidate_class = $candidate_class,
            n.status = $status,
            n.confidence = $confidence,
            n.source_tag = $source_tag,
            n.tag_family = $tag_family,
            n.source_id = $source_id,
            n.source_type = $source_type,
            n.source_sheet = $source_sheet,
            n.source_column = $source_column,
            n.source_row_start = $source_row_start,
            n.source_row_end = $source_row_end,
            n.extraction_method = $extraction_method
        WITH n
        SET n:Entity
        """

        with self.driver.session() as session:
            for entity in entities:
                props = make_node_properties(entity)
                session.run(statement, **props).consume()

    def load_relations(
        self,
        relations: list[dict[str, Any]],
    ) -> None:
        with self.driver.session() as session:
            for relation in relations:
                relation_type = sanitize_relationship_type(
                    str(relation["relation"])
                )

                properties = make_relation_properties(relation)

                source = str(relation["source"])
                target = str(relation["target"])

                if relation_type == "INSTANCE_OF":
                    query = f"""
                    MATCH (source:Entity {{id: $source}})
                    MERGE (target:Class {{name: $target}})
                    MERGE (source)-[r:{relation_type}]->(target)
                    SET
                        r.id = $id,
                        r.status = $status,
                        r.confidence = $confidence,
                        r.evidence_type = $evidence_type,
                        r.source_id = $source_id,
                        r.source_type = $source_type,
                        r.source_sheet = $source_sheet,
                        r.source_column = $source_column,
                        r.source_row_start = $source_row_start,
                        r.source_row_end = $source_row_end,
                        r.extraction_method = $extraction_method
                    """
                else:
                    query = f"""
                    MATCH (source:Entity {{id: $source}})
                    MATCH (target:Entity {{id: $target}})
                    MERGE (source)-[r:{relation_type}]->(target)
                    SET
                        r.id = $id,
                        r.status = $status,
                        r.confidence = $confidence,
                        r.evidence_type = $evidence_type,
                        r.source_id = $source_id,
                        r.source_type = $source_type,
                        r.source_sheet = $source_sheet,
                        r.source_column = $source_column,
                        r.source_row_start = $source_row_start,
                        r.source_row_end = $source_row_end,
                        r.extraction_method = $extraction_method
                    """

                session.run(
                    query,
                    source=source,
                    target=target,
                    **properties,
                ).consume()

    def load(self, payload: dict[str, Any]) -> None:
        self.create_constraints()

        class_names = {
            class_name
            for relation in payload["relations"]
            if (class_name := class_name_from_relation(relation))
        }

        self.load_classes(class_names)
        self.load_entities(payload["entities"])
        self.load_relations(payload["relations"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load validated industrial knowledge into Neo4j."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Validated knowledge file (default: {DEFAULT_INPUT})",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate JSON structure without connecting to Neo4j.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = load_json(args.input.resolve())
    validate_payload(payload)

    print("=" * 60)
    print("NEO4J KNOWLEDGE GRAPH LOADER")
    print("=" * 60)
    print(
        f"Entities   : {len(payload['entities'])}"
    )
    print(
        f"Relations  : {len(payload['relations'])}"
    )

    if args.validate_only:
        print("Mode       : validate-only")
        print("Result     : input structure valid")
        print("=" * 60)
        return

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. "
            "Set it in your environment before loading the graph."
        )

    loader = KnowledgeGraphLoader(
        uri=uri,
        user=user,
        password=password,
    )

    try:
        loader.load(payload)
    finally:
        loader.close()

    print(f"URI        : {uri}")
    print("Result     : graph loaded successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
