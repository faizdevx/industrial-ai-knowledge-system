"""
Stage 8 of the Industrial AI MVP:
    RCSD-1YD.xlsx
        ↓
    time-series analysis
        ↓
    sensor trends / spikes / change points / correlations
        ↓
    event_candidates.json
    timeseries_profile.json

This stage is intentionally statistical, not causal.
It does NOT claim that one sensor causes another or that an anomaly
proves equipment failure.

Usage:
    uv run python src/timeseries/timeseries_analysis.py

Expected input:
    data/raw/refinery/RCSD-1YD.xlsx

Outputs:
    data/processed/refinery/timeseries/
        timeseries_profile.json
        sensor_timeseries_summary.csv
        event_candidates.json
        correlated_sensor_pairs.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/raw/refinery/RCSD-1YD.xlsx")
DEFAULT_OUTPUT_DIR = Path("data/processed/refinery/timeseries")

ROLLING_WINDOW = 96       # 24 hours at 15-minute sampling
TREND_WINDOW = 96         # compare one-day start/end windows
ROBUST_Z_THRESHOLD = 6.0
CHANGE_SCORE_THRESHOLD = 3.0
CORRELATION_THRESHOLD = 0.80


TAG_FAMILY_HINTS = {
    "PI": "Pressure",
    "PDI": "DifferentialPressure",
    "TI": "Temperature",
    "SI": "Speed",
    "XI": "PositionOrDisplacement",
    "ZI": "Position",
}


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def infer_tag_family(tag: str) -> str | None:
    upper = tag.upper()

    # Longer prefixes first.
    for family in ("PDI", "PI", "TI", "XI", "ZI", "SI"):
        if family in upper:
            return family

    return None


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Median/MAD based robust z-score.

    0.6745 converts MAD to the standard-normal scale approximately.
    """
    median = series.median()
    mad = (series - median).abs().median()

    if pd.isna(mad) or mad == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
            dtype=float,
        )

    return 0.6745 * (series - median) / mad


def trend_summary(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()

    if len(clean) < TREND_WINDOW * 2:
        return {
            "label": "insufficient_data",
            "start_mean": None,
            "end_mean": None,
            "difference": None,
        }

    start_mean = float(clean.iloc[:TREND_WINDOW].mean())
    end_mean = float(clean.iloc[-TREND_WINDOW:].mean())
    difference = end_mean - start_mean

    std = float(clean.std())
    normalized_change = (
        abs(difference) / std if std > 1e-12 else 0.0
    )

    if normalized_change < 1.0:
        label = "stable"
    elif difference > 0:
        label = "rising"
    else:
        label = "falling"

    return {
        "label": label,
        "start_mean": start_mean,
        "end_mean": end_mean,
        "difference": difference,
        "normalized_change": normalized_change,
    }


def detect_spike_events(
    timestamp: pd.Series,
    series: pd.Series,
    tag: str,
) -> list[dict[str, Any]]:
    z = robust_zscore(series)
    mask = z.abs() >= ROBUST_Z_THRESHOLD

    indices = list(series.index[mask])

    if not indices:
        return []

    events: list[dict[str, Any]] = []

    # Group consecutive row indices into one event window.
    current = [indices[0]]

    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            events.append(
                build_spike_event(timestamp, series, z, tag, current)
            )
            current = [idx]

    events.append(
        build_spike_event(timestamp, series, z, tag, current)
    )

    return events


def build_spike_event(
    timestamp: pd.Series,
    series: pd.Series,
    z: pd.Series,
    tag: str,
    indices: list[int],
) -> dict[str, Any]:
    start_idx = indices[0]
    end_idx = indices[-1]

    segment = series.loc[indices]
    z_segment = z.loc[indices]

    peak_idx = z_segment.abs().idxmax()
    peak_value = float(series.loc[peak_idx])
    peak_z = float(z.loc[peak_idx])

    direction = "high_spike" if peak_z > 0 else "low_spike"

    return {
        "event_type": "statistical_spike",
        "sensor_tag": tag,
        "start": json_safe(timestamp.loc[start_idx]),
        "end": json_safe(timestamp.loc[end_idx]),
        "peak_timestamp": json_safe(timestamp.loc[peak_idx]),
        "peak_value": peak_value,
        "peak_robust_z": peak_z,
        "direction": direction,
        "sample_count": len(indices),
        "status": "candidate",
        "interpretation": (
            "Statistical outlier relative to this sensor's historical "
            "distribution; engineering meaning requires validation."
        ),
    }


def detect_change_points(
    timestamp: pd.Series,
    series: pd.Series,
    tag: str,
) -> list[dict[str, Any]]:
    clean = series.copy()

    rolling_mean = clean.rolling(
        ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    ).mean()

    rolling_std = clean.rolling(
        ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    ).std()

    previous_mean = rolling_mean.shift(ROLLING_WINDOW)
    current_std = rolling_std.replace(0, np.nan)

    score = (rolling_mean - previous_mean).abs() / current_std
    candidates = score[score >= CHANGE_SCORE_THRESHOLD]

    if candidates.empty:
        return []

    # Keep the strongest change points and avoid thousands of overlapping
    # windows becoming separate events.
    top = candidates.sort_values(ascending=False).head(20)

    events = []

    for idx, score_value in top.items():
        if pd.isna(idx) or pd.isna(score_value):
            continue

        events.append(
            {
                "event_type": "change_point_candidate",
                "sensor_tag": tag,
                "timestamp": json_safe(timestamp.loc[idx]),
                "change_score": float(score_value),
                "status": "candidate",
                "interpretation": (
                    "A large shift between rolling means was detected. "
                    "This is not proof of a fault or causal event."
                ),
            }
        )

    return events


def sensor_summary(
    timestamp: pd.Series,
    series: pd.Series,
    tag: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    numeric = pd.to_numeric(series, errors="coerce")

    valid = numeric.dropna()

    if valid.empty:
        return (
            {
                "tag": tag,
                "tag_family": infer_tag_family(tag),
                "count": 0,
                "missing_count": int(numeric.isna().sum()),
            },
            [],
        )

    trend = trend_summary(numeric)

    spikes = detect_spike_events(
        timestamp=timestamp,
        series=numeric,
        tag=tag,
    )

    changes = detect_change_points(
        timestamp=timestamp,
        series=numeric,
        tag=tag,
    )

    summary = {
        "tag": tag,
        "tag_family": infer_tag_family(tag),
        "count": int(len(numeric)),
        "valid_count": int(len(valid)),
        "missing_count": int(numeric.isna().sum()),
        "missing_fraction": float(numeric.isna().mean()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "median": float(valid.median()),
        "trend": trend,
        "spike_event_count": len(spikes),
        "change_point_candidate_count": len(changes),
    }

    return summary, spikes + changes


def correlation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        c for c in df.columns
        if c != "Timestamp"
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_columns) < 2:
        return pd.DataFrame(
            columns=[
                "sensor_a",
                "sensor_b",
                "correlation",
                "absolute_correlation",
            ]
        )

    corr = df[numeric_columns].corr()

    rows = []

    for i, sensor_a in enumerate(numeric_columns):
        for sensor_b in numeric_columns[i + 1:]:
            value = corr.loc[sensor_a, sensor_b]

            if pd.isna(value):
                continue

            if abs(float(value)) >= CORRELATION_THRESHOLD:
                rows.append(
                    {
                        "sensor_a": sensor_a,
                        "sensor_b": sensor_b,
                        "correlation": float(value),
                        "absolute_correlation": abs(float(value)),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        "absolute_correlation",
        ascending=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical time-series analysis on RCSD-1YD."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIME-SERIES ANALYSIS")
    print("=" * 60)
    print(f"Input: {input_path}")

    df = pd.read_excel(
        input_path,
        sheet_name="Sheet1",
    )

    if "Timestamp" not in df.columns:
        raise ValueError("Expected a 'Timestamp' column.")

    timestamp = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if timestamp.isna().any():
        raise ValueError(
            f"Timestamp parsing failed for "
            f"{int(timestamp.isna().sum())} rows."
        )

    numeric_columns = [
        c for c in df.columns
        if c != "Timestamp"
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"Rows: {len(df)}")
    print(f"Numeric sensors: {len(numeric_columns)}")

    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for tag in numeric_columns:
        summary, sensor_events = sensor_summary(
            timestamp=timestamp,
            series=df[tag],
            tag=str(tag),
        )

        summaries.append(summary)
        events.extend(sensor_events)

    corr_df = correlation_pairs(df)

    # Sensor summary.
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        output_dir / "sensor_timeseries_summary.csv",
        index=False,
    )

    # Correlation pairs.
    corr_df.to_csv(
        output_dir / "correlated_sensor_pairs.csv",
        index=False,
    )

    profile = {
        "source": {
            "file_name": input_path.name,
            "sheet": "Sheet1",
        },
        "dataset": {
            "rows": len(df),
            "sensor_count": len(numeric_columns),
            "timestamp_column": "Timestamp",
            "start": json_safe(timestamp.min()),
            "end": json_safe(timestamp.max()),
        },
        "analysis_parameters": {
            "rolling_window_samples": ROLLING_WINDOW,
            "trend_window_samples": TREND_WINDOW,
            "robust_z_threshold": ROBUST_Z_THRESHOLD,
            "change_score_threshold": CHANGE_SCORE_THRESHOLD,
            "correlation_threshold": CORRELATION_THRESHOLD,
        },
        "limitations": [
            "Spike detection is statistical, not an engineering alarm.",
            "Change points are candidates requiring domain validation.",
            "Correlation does not establish causation.",
            "Threshold violations are not evaluated because engineering limits "
            "have not yet been linked from technical documents.",
            "Equipment failure is not inferred from statistical anomalies.",
        ],
        "sensor_summaries": summaries,
    }

    with (output_dir / "timeseries_profile.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            profile,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (output_dir / "event_candidates.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "status": "candidate",
                "events": events,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Events detected: {len(events)}")
    print(f"Correlated pairs: {len(corr_df)}")
    print()
    print(f"Created: {output_dir / 'timeseries_profile.json'}")
    print(f"Created: {output_dir / 'sensor_timeseries_summary.csv'}")
    print(f"Created: {output_dir / 'event_candidates.json'}")
    print(f"Created: {output_dir / 'correlated_sensor_pairs.csv'}")
    print()
    print("Next stage: link validated events back to the knowledge graph.")
    print("=" * 60)


if __name__ == "__main__":
    main()
