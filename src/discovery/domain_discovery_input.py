"""
Prepare structured evidence for the domain-discovery stage.

This script does NOT ask an LLM to invent an ontology.
It converts the already-profiled refinery dataset into a compact,
machine-readable evidence package for the next stage.

Usage:
    uv run python src/discovery/domain_discovery_input.py

Expected input:
    data/raw/refinery/RCSD-1YD.xlsx

Output:
    data/processed/refinery/domain_discovery_input.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/raw/refinery/RCSD-1YD.xlsx")
DEFAULT_OUTPUT = Path(
    "data/processed/refinery/domain_discovery_input.json"
)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy-like values into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return value


def infer_tag_family(tag: str) -> str | None:
    """
    Extract the instrument family from tags such as:
        75PI823.pv -> PI
        75PDI853.pv -> PDI
        75TI821.pv -> TI
        75XI821BX.pv -> XI
        75ZI800BA.pv -> ZI
        75SI865R.pv -> SI
    """
    match = re.search(
        r"\d+(PDI|PI|TI|XI|ZI|SI)",
        tag.upper(),
    )
    return match.group(1) if match else None


def describe_numeric_column(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "q01": None,
            "q99": None,
        }

    return {
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "median": float(numeric.median()),
        "q01": float(numeric.quantile(0.01)),
        "q99": float(numeric.quantile(0.99)),
    }


def build_sensor_evidence(df: pd.DataFrame) -> list[dict[str, Any]]:
    sensors = []

    for column in df.columns:
        name = str(column)

        if name.lower() == "timestamp":
            continue

        series = df[column]

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_valid = numeric.dropna()

        evidence = {
            "tag": name,
            "tag_family_hint": infer_tag_family(name),
            "data_type": str(series.dtype),
            "numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "missing_count": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": [
                json_safe(v)
                for v in series.dropna().head(5).tolist()
            ],
        }

        if not numeric_valid.empty:
            evidence["statistics"] = describe_numeric_column(series)

            # A rough variability indicator.
            # This is descriptive only, not an engineering classification.
            mean_abs = abs(float(numeric_valid.mean()))
            std = float(numeric_valid.std())
            evidence["relative_std"] = (
                std / mean_abs if mean_abs > 1e-12 else None
            )

        sensors.append(evidence)

    return sensors


def build_temporal_evidence(df: pd.DataFrame) -> dict[str, Any]:
    if "Timestamp" not in df.columns:
        return {
            "timestamp_column": None,
            "status": "not_found",
        }

    ts = pd.to_datetime(df["Timestamp"], errors="coerce")

    valid = ts.dropna()

    result = {
        "timestamp_column": "Timestamp",
        "parse_failures": int(ts.isna().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "start": json_safe(valid.min()) if not valid.empty else None,
        "end": json_safe(valid.max()) if not valid.empty else None,
    }

    if len(valid) >= 2:
        deltas = valid.sort_values().diff().dropna()
        counts = deltas.value_counts()

        result["sampling"] = {
            "most_common_interval": str(counts.index[0]),
            "most_common_interval_count": int(counts.iloc[0]),
            "unique_intervals": int(deltas.nunique()),
        }

    return result


def build_tag_family_summary(
    sensors: list[dict[str, Any]],
) -> dict[str, int]:
    summary: dict[str, int] = {}

    for sensor in sensors:
        family = sensor.get("tag_family_hint") or "UNKNOWN"
        summary[family] = summary.get(family, 0) + 1

    return dict(sorted(summary.items()))


def build_domain_input(
    input_path: Path,
    df: pd.DataFrame,
) -> dict[str, Any]:
    sensors = build_sensor_evidence(df)

    return {
        "purpose": (
            "Evidence package for industrial domain discovery. "
            "Descriptions are hypotheses to be validated; this file "
            "does not assert engineering semantics."
        ),
        "source": {
            "file_name": input_path.name,
            "sheet": "Sheet1",
        },
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "temporal_evidence": build_temporal_evidence(df),
        "tag_family_summary": build_tag_family_summary(sensors),
        "sensor_evidence": sensors,
        "candidate_signals": {
            "contains_timestamp": "Timestamp" in df.columns,
            "numeric_column_count": int(
                sum(
                    pd.api.types.is_numeric_dtype(df[c])
                    for c in df.columns
                    if str(c).lower() != "timestamp"
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare domain-discovery evidence from RCSD-1YD."
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

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOMAIN DISCOVERY INPUT BUILDER")
    print("=" * 60)
    print(f"Reading: {input_path}")

    df = pd.read_excel(
        input_path,
        sheet_name="Sheet1",
    )

    payload = build_domain_input(
        input_path=input_path,
        df=df,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(payload),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Rows: {payload['dataset']['rows']}")
    print(f"Columns: {payload['dataset']['columns']}")
    print(
        "Tag families:",
        payload["tag_family_summary"],
    )
    print(
        "Time range:",
        payload["temporal_evidence"].get("start"),
        "to",
        payload["temporal_evidence"].get("end"),
    )
    print(f"\nCreated: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()