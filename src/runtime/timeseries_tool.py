"""
Time-series tools for the Industrial AI Knowledge System.

The tools read the existing refinery artifacts instead of putting every
sensor row into Neo4j.

Inputs:
  data/processed/refinery/Sheet1.csv
  data/processed/refinery/timeseries/sensor_timeseries_summary.csv
  data/processed/refinery/timeseries/event_candidates.json

Tools:
  - identify_sensors(query)
  - sensor_summary(sensor_tag)
  - recent_values(sensor_tag, rows)
  - sensor_events(sensor_tag, limit)
  - compare_sensors(sensor_a, sensor_b, rows)

Usage:
  uv run python src/runtime/timeseries_tool.py summary 75PI823.pv
  uv run python src/runtime/timeseries_tool.py events 75PI823.pv
  uv run python src/runtime/timeseries_tool.py recent 75PI823.pv --rows 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SHEET = ROOT / "data" / "processed" / "refinery" / "Sheet1.csv"
SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "timeseries"
    / "sensor_timeseries_summary.csv"
)
EVENTS = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "timeseries"
    / "event_candidates.json"
)


def load_sensor_data() -> pd.DataFrame:
    if not SHEET.exists():
        raise FileNotFoundError(f"Missing sensor data: {SHEET}")

    df = pd.read_csv(SHEET)
    if "Timestamp" not in df.columns:
        raise ValueError("Sheet1.csv must contain a Timestamp column.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    return df


def load_summary() -> pd.DataFrame:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing sensor summary: {SUMMARY}")
    return pd.read_csv(SUMMARY)


def load_events() -> list[dict[str, Any]]:
    if not EVENTS.exists():
        raise FileNotFoundError(f"Missing event candidates: {EVENTS}")

    with EVENTS.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "events",
            "event_candidates",
            "candidates",
            "items",
        ):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Unsupported event_candidates.json structure.")


def available_sensors() -> list[str]:
    df = load_sensor_data()
    return [c for c in df.columns if c != "Timestamp"]


def identify_sensors(query: str, limit: int = 10) -> list[str]:
    sensors = available_sensors()
    q = query.lower()

    exact = [
        s for s in sensors
        if s.lower() in q
    ]

    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_.-]+", query)
        if len(t) >= 3
    ]

    fuzzy = []
    for sensor in sensors:
        sl = sensor.lower()
        score = sum(
            1 for token in tokens
            if token in sl
        )
        if score:
            fuzzy.append((score, sensor))

    fuzzy.sort(key=lambda x: (-x[0], x[1]))

    output = []
    seen = set()

    for sensor in exact + [x[1] for x in fuzzy]:
        if sensor not in seen:
            seen.add(sensor)
            output.append(sensor)

    return output[:limit]


def require_sensor(df: pd.DataFrame, sensor_tag: str) -> None:
    if sensor_tag not in df.columns:
        matches = [
            c for c in df.columns
            if c.lower() == sensor_tag.lower()
        ]
        if matches:
            return
        raise KeyError(
            f"Unknown sensor '{sensor_tag}'. "
            f"Available sensors: {', '.join(c for c in df.columns if c != 'Timestamp')}"
        )


def canonical_sensor(df: pd.DataFrame, sensor_tag: str) -> str:
    require_sensor(df, sensor_tag)
    for column in df.columns:
        if column.lower() == sensor_tag.lower():
            return column
    return sensor_tag


def sensor_summary(sensor_tag: str) -> dict[str, Any]:
    df = load_sensor_data()
    sensor_tag = canonical_sensor(df, sensor_tag)

    s = pd.to_numeric(df[sensor_tag], errors="coerce").dropna()

    if s.empty:
        raise ValueError(f"No numeric values for {sensor_tag}")

    summary_df = load_summary()
    summary_row = summary_df[
        summary_df.apply(
            lambda row: any(
                str(value).lower() == sensor_tag.lower()
                for value in row.values
            ),
            axis=1,
        )
    ]

    result = {
        "sensor": sensor_tag,
        "rows": int(s.size),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "first_timestamp": df["Timestamp"].min().isoformat(),
        "last_timestamp": df["Timestamp"].max().isoformat(),
        "source": "Sheet1.csv",
    }

    if not summary_row.empty:
        result["analysis_summary"] = summary_row.iloc[0].to_dict()

    return result


def recent_values(sensor_tag: str, rows: int = 20) -> dict[str, Any]:
    df = load_sensor_data()
    sensor_tag = canonical_sensor(df, sensor_tag)

    output = df[["Timestamp", sensor_tag]].tail(max(1, rows)).copy()
    output[sensor_tag] = pd.to_numeric(
        output[sensor_tag],
        errors="coerce",
    )

    return {
        "sensor": sensor_tag,
        "rows": [
            {
                "timestamp": row["Timestamp"].isoformat(),
                "value": (
                    None
                    if pd.isna(row[sensor_tag])
                    else float(row[sensor_tag])
                ),
            }
            for _, row in output.iterrows()
        ],
        "source": "Sheet1.csv",
    }


def sensor_events(sensor_tag: str, limit: int = 20) -> list[dict[str, Any]]:
    df = load_sensor_data()
    sensor_tag = canonical_sensor(df, sensor_tag)
    events = load_events()

    matches = []

    for event in events:
        event_sensor = (
            event.get("sensor_tag")
            or event.get("sensor")
            or event.get("entity_id")
            or event.get("source_column")
        )

        if event_sensor and str(event_sensor).lower() == sensor_tag.lower():
            matches.append(event)

    matches.sort(
        key=lambda e: str(
            e.get("start")
            or e.get("timestamp")
            or ""
        ),
        reverse=True,
    )

    return matches[:max(1, limit)]


def compare_sensors(
    sensor_a: str,
    sensor_b: str,
    rows: int = 500,
) -> dict[str, Any]:
    df = load_sensor_data()

    sensor_a = canonical_sensor(df, sensor_a)
    sensor_b = canonical_sensor(df, sensor_b)

    window = df[["Timestamp", sensor_a, sensor_b]].tail(
        max(2, rows)
    ).copy()

    a = pd.to_numeric(window[sensor_a], errors="coerce")
    b = pd.to_numeric(window[sensor_b], errors="coerce")

    valid = a.notna() & b.notna()
    if valid.sum() < 2:
        correlation = None
    else:
        correlation = float(a[valid].corr(b[valid]))

    return {
        "sensor_a": sensor_a,
        "sensor_b": sensor_b,
        "rows_analyzed": int(valid.sum()),
        "pearson_correlation": correlation,
        "start": window["Timestamp"].min().isoformat(),
        "end": window["Timestamp"].max().isoformat(),
        "warning": "Correlation does not establish causation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="command", required=True)

    p_identify = sub.add_parser("identify")
    p_identify.add_argument("query")

    p_summary = sub.add_parser("summary")
    p_summary.add_argument("sensor")

    p_recent = sub.add_parser("recent")
    p_recent.add_argument("sensor")
    p_recent.add_argument("--rows", type=int, default=20)

    p_events = sub.add_parser("events")
    p_events.add_argument("sensor")
    p_events.add_argument("--limit", type=int, default=20)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("sensor_a")
    p_compare.add_argument("sensor_b")
    p_compare.add_argument("--rows", type=int, default=500)

    args = parser.parse_args()

    result: Any

    if args.command == "identify":
        result = identify_sensors(args.query)

    elif args.command == "summary":
        result = sensor_summary(args.sensor)

    elif args.command == "recent":
        result = recent_values(args.sensor, args.rows)

    elif args.command == "events":
        result = sensor_events(args.sensor, args.limit)

    elif args.command == "compare":
        result = compare_sensors(
            args.sensor_a,
            args.sensor_b,
            args.rows,
        )

    else:
        raise RuntimeError("Unknown command.")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
