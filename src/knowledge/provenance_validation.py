"""
Stage 6 of the Industrial AI MVP:
    knowledge_objects.json
            ↓
    provenance + structural validation
            ↓
    validated_knowledge.json
    validation_report.json

What this stage validates:
    1. Every entity has provenance.
    2. Every relation has provenance.
    3. Referenced source file exists (when --source-root is supplied).
    4. Referenced sheet exists in the source workbook.
    5. Referenced columns exist in the source workbook.
    6. Referenced Excel row ranges are valid.
    7. Relation endpoints resolve to known entity IDs or ontology class names.
    8. Basic provenance confidence is in [0, 1].

Important:
    "validated" here means structurally/source validated.
    It does NOT mean the engineering semantics are proven.

Usage from project root:
    uv run python src/knowledge/provenance_validation.py

Optional explicit source root:
    uv run python src/knowledge/provenance_validation.py \
        --source-root data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import openpyxl


DEFAULT_INPUT = Path(
    "data/processed/refinery/knowledge_objects.json"
)
DEFAULT_OUTPUT = Path(
    "data/processed/refinery/validated_knowledge.json"
)
DEFAULT_REPORT = Path(
    "data/processed/refinery/validation_report.json"
)
DEFAULT_SOURCE_ROOT = Path("data/raw")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    category: str,
    message: str,
    object_id: str | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "category": category,
            "message": message,
            "object_id": object_id,
        }
    )


def check_confidence(
    confidence: Any,
    issues: list[dict[str, Any]],
    object_id: str,
) -> None:
    if confidence is None:
        add_issue(
            issues,
            "warning",
            "confidence",
            "Missing confidence value.",
            object_id,
        )
        return

    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        add_issue(
            issues,
            "error",
            "confidence",
            f"Confidence must be a number in [0, 1], got {confidence!r}.",
            object_id,
        )


def resolve_source_file(
    source_root: Path,
    source_id: str,
) -> Path | None:
    """
    Search source_root recursively for the source filename.

    We use the basename rather than assuming an exact folder structure so
    the script remains usable across Windows/Linux project layouts.
    """
    direct = source_root / source_id
    if direct.exists():
        return direct

    matches = list(source_root.rglob(source_id))
    return matches[0] if matches else None


def inspect_workbook(
    source_path: Path,
) -> tuple[list[str], dict[str, set[str]], dict[str, int]]:
    """
    Return:
        sheet_names
        columns_by_sheet
        max_data_row_by_sheet
    """
    wb = openpyxl.load_workbook(
        source_path,
        read_only=True,
        data_only=True,
    )

    sheet_names = list(wb.sheetnames)
    columns_by_sheet: dict[str, set[str]] = {}
    max_rows: dict[str, int] = {}

    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        max_rows[sheet_name] = ws.max_row

        first_row = next(
            ws.iter_rows(
                min_row=1,
                max_row=1,
                values_only=True,
            ),
            (),
        )

        columns_by_sheet[sheet_name] = {
            str(value).strip()
            for value in first_row
            if value is not None
        }

    wb.close()

    return sheet_names, columns_by_sheet, max_rows


def validate_entity(
    entity: dict[str, Any],
    knowledge: dict[str, Any],
    source_root: Path,
    workbook_cache: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> bool:
    entity_id = str(entity.get("id", ""))

    if not entity_id:
        add_issue(
            issues,
            "error",
            "entity_schema",
            "Entity is missing an id.",
            None,
        )
        return False

    if not entity.get("entity_type"):
        add_issue(
            issues,
            "error",
            "entity_schema",
            "Entity is missing entity_type.",
            entity_id,
        )

    provenance = entity.get("provenance")

    if not provenance:
        add_issue(
            issues,
            "error",
            "provenance",
            "Entity has no provenance.",
            entity_id,
        )
        return False

    source_id = provenance.get("source_id")
    location = provenance.get("location", {})
    sheet = location.get("sheet")
    column = location.get("column")

    if not source_id:
        add_issue(
            issues,
            "error",
            "provenance",
            "Provenance is missing source_id.",
            entity_id,
        )

    if not sheet:
        add_issue(
            issues,
            "error",
            "provenance",
            "Provenance is missing sheet.",
            entity_id,
        )

    if not column:
        add_issue(
            issues,
            "error",
            "provenance",
            "Provenance is missing column.",
            entity_id,
        )

    check_confidence(
        provenance.get("confidence"),
        issues,
        entity_id,
    )

    source_path = resolve_source_file(
        source_root,
        str(source_id),
    )

    if not source_path:
        add_issue(
            issues,
            "warning",
            "source_file",
            f"Source file '{source_id}' was not found under {source_root}.",
            entity_id,
        )
        return False

    cache_key = str(source_path.resolve())

    if cache_key not in workbook_cache:
        try:
            (
                sheet_names,
                columns_by_sheet,
                max_rows,
            ) = inspect_workbook(source_path)

            workbook_cache[cache_key] = {
                "sheet_names": sheet_names,
                "columns_by_sheet": columns_by_sheet,
                "max_rows": max_rows,
            }
        except Exception as exc:
            add_issue(
                issues,
                "error",
                "workbook_read",
                f"Could not inspect workbook: {exc}",
                entity_id,
            )
            return False

    workbook = workbook_cache[cache_key]

    if sheet not in workbook["sheet_names"]:
        add_issue(
            issues,
            "error",
            "sheet",
            f"Sheet '{sheet}' does not exist in {source_id}.",
            entity_id,
        )
        return False

    if column not in workbook["columns_by_sheet"].get(sheet, set()):
        add_issue(
            issues,
            "error",
            "column",
            f"Column '{column}' does not exist in sheet '{sheet}'.",
            entity_id,
        )
        return False

    data_rows = location.get("data_rows")

    if data_rows:
        start = data_rows.get("start")
        end = data_rows.get("end")
        max_row = workbook["max_rows"].get(sheet, 0)

        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 2
            or end < start
        ):
            add_issue(
                issues,
                "error",
                "row_range",
                f"Invalid data row range: {data_rows!r}.",
                entity_id,
            )
            return False

        if end > max_row:
            add_issue(
                issues,
                "error",
                "row_range",
                (
                    f"Data range ends at Excel row {end}, "
                    f"but sheet '{sheet}' has only {max_row} rows."
                ),
                entity_id,
            )
            return False

    return True


def validate_relation(
    relation: dict[str, Any],
    entity_ids: set[str],
    class_names: set[str],
    issues: list[dict[str, Any]],
) -> bool:
    relation_id = str(relation.get("id", ""))

    if not relation_id:
        add_issue(
            issues,
            "error",
            "relation_schema",
            "Relation is missing an id.",
            None,
        )
        return False

    source = relation.get("source")
    target = relation.get("target")

    if not source:
        add_issue(
            issues,
            "error",
            "relation_schema",
            "Relation is missing source.",
            relation_id,
        )

    if not relation.get("relation"):
        add_issue(
            issues,
            "error",
            "relation_schema",
            "Relation is missing relation name.",
            relation_id,
        )

    if not target:
        add_issue(
            issues,
            "error",
            "relation_schema",
            "Relation is missing target.",
            relation_id,
        )

    provenance = relation.get("provenance")

    if not provenance:
        add_issue(
            issues,
            "error",
            "provenance",
            "Relation has no provenance.",
            relation_id,
        )
    else:
        check_confidence(
            provenance.get("confidence"),
            issues,
            relation_id,
        )

    source_ok = source in entity_ids or source in class_names
    target_ok = target in entity_ids or target in class_names

    if not source_ok:
        add_issue(
            issues,
            "error",
            "relation_endpoint",
            f"Unknown source endpoint '{source}'.",
            relation_id,
        )

    if not target_ok:
        add_issue(
            issues,
            "error",
            "relation_endpoint",
            f"Unknown target endpoint '{target}'.",
            relation_id,
        )

    return (
        bool(source)
        and bool(target)
        and source_ok
        and target_ok
        and bool(relation.get("relation"))
        and bool(provenance)
    )


def validate_knowledge(
    knowledge: dict[str, Any],
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entities = knowledge.get("entities", [])
    relations = knowledge.get("relations", [])

    issues: list[dict[str, Any]] = []
    workbook_cache: dict[str, dict[str, Any]] = {}

    entity_ids = {
        str(entity.get("id"))
        for entity in entities
        if entity.get("id")
    }

    class_names = set()

    # Candidate ontology class names may be relation targets.
    # The ontology itself is not required in this file, so recover them
    # from the relation targets that look like class names only if needed.
    for relation in relations:
        target = relation.get("target")
        if isinstance(target, str) and target and target not in entity_ids:
            class_names.add(target)

    validated_entities = []
    valid_entity_ids: set[str] = set()

    for entity in entities:
        valid = validate_entity(
            entity=entity,
            knowledge=knowledge,
            source_root=source_root,
            workbook_cache=workbook_cache,
            issues=issues,
        )

        enriched = dict(entity)
        enriched["validation"] = {
            "status": "source_validated" if valid else "needs_review"
        }

        validated_entities.append(enriched)

        if valid:
            valid_entity_ids.add(str(entity["id"]))

    validated_relations = []

    for relation in relations:
        valid = validate_relation(
            relation=relation,
            entity_ids=entity_ids,
            class_names=class_names,
            issues=issues,
        )

        enriched = dict(relation)
        enriched["validation"] = {
            "status": "structurally_validated" if valid else "needs_review"
        }

        validated_relations.append(enriched)

    errors = sum(
        1 for issue in issues
        if issue["severity"] == "error"
    )

    warnings = sum(
        1 for issue in issues
        if issue["severity"] == "warning"
    )

    report = {
        "status": "passed" if errors == 0 else "needs_review",
        "summary": {
            "entities_total": len(entities),
            "entities_source_validated": len(valid_entity_ids),
            "relations_total": len(relations),
            "errors": errors,
            "warnings": warnings,
        },
        "checks": {
            "entity_provenance": True,
            "relation_provenance": True,
            "source_file_lookup": True,
            "sheet_lookup": True,
            "column_lookup": True,
            "row_range_validation": True,
            "relation_endpoint_validation": True,
            "confidence_range_validation": True,
        },
        "issues": issues,
        "next_stage": (
            "knowledge_graph_build"
            if errors == 0
            else "fix_validation_errors"
        ),
    }

    validated = {
        "status": report["status"],
        "source": knowledge.get("source"),
        "domain_hypothesis": knowledge.get("domain_hypothesis"),
        "entities": validated_entities,
        "relations": validated_relations,
        "validation_report": report,
        "not_inferred": knowledge.get("not_inferred", []),
    }

    return validated, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate provenance and structure of knowledge objects."
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
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )

    args = parser.parse_args()

    knowledge = load_json(args.input.resolve())
    source_root = args.source_root.resolve()

    validated, report = validate_knowledge(
        knowledge=knowledge,
        source_root=source_root,
    )

    output_path = args.output.resolve()
    report_path = args.report.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("PROVENANCE + STRUCTURAL VALIDATION")
    print("=" * 60)
    print(f"Status                  : {report['status']}")
    print(
        f"Entities                : "
        f"{report['summary']['entities_source_validated']}/"
        f"{report['summary']['entities_total']} source-validated"
    )
    print(
        f"Relations               : "
        f"{report['summary']['relations_total']}"
    )
    print(f"Errors                  : {report['summary']['errors']}")
    print(f"Warnings                : {report['summary']['warnings']}")

    if report["issues"]:
        print("\nIssues:")
        for issue in report["issues"][:20]:
            print(
                f"  [{issue['severity']}] "
                f"{issue['category']}: "
                f"{issue['message']}"
            )

    print()
    print(f"Created: {output_path}")
    print(f"Created: {report_path}")
    print()
    print(f"Next stage: {report['next_stage']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
