"""
Profile an industrial Excel workbook and prepare it for the next ingestion stages.

Designed for the RCSD-1YD workbook, but kept generic enough for future
industrial datasets.

Usage:
    uv run python src/ingestion/profile_excel.py

Or, from anywhere:
    uv run python profile_excel.py --input /path/to/RCSD-1YD.xlsx

Outputs:
    data/processed/refinery/
        workbook_profile.json
        sensor_profile.csv
        <non-empty-sheet>.csv
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
DEFAULT_OUTPUT_DIR = Path("data/processed/refinery")


# Common industrial tag prefixes seen in the RCSD-1YD workbook.
# These are only tag-family hints. They are NOT treated as authoritative
# engineering semantics or units.
TAG_FAMILY_HINTS = {
    "ZI": "ZI",
    "PI": "PI",
    "PDI": "PDI",
    "TI": "TI",
    "XI": "XI",
    "SI": "SI",
}


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
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

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (pd.Timedelta,)):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return value


def safe_filename(name: str) -> str:
    """Make a worksheet name safe for a filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return cleaned or "sheet"


def infer_tag_family(column_name: str) -> str | None:
    """
    Extract a simple tag-family hint from an industrial tag.

    Example:
        75PI823.pv -> PI
        75TI821.pv -> TI
    """
    match = re.search(r"\d+(PDI|PI|TI|XI|ZI|SI)", column_name.upper())
    if match:
        return match.group(1)

    # Fallback for tags that start directly with the family.
    upper = column_name.upper()
    for family in TAG_FAMILY_HINTS:
        if upper.startswith(family):
            return family

    return None


def profile_numeric_column(series: pd.Series) -> dict[str, Any]:
    """Return useful statistics for a numeric sensor column."""
    numeric = pd.to_numeric(series, errors="coerce")

    non_null = numeric.dropna()

    result: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_fraction": float(series.isna().mean()),
        "unique_count": int(series.nunique(dropna=True)),
    }

    if non_null.empty:
        result.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "median": None,
                "q01": None,
                "q99": None,
            }
        )
        return result

    result.update(
        {
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "mean": float(non_null.mean()),
            "std": float(non_null.std()),
            "median": float(non_null.median()),
            "q01": float(non_null.quantile(0.01)),
            "q99": float(non_null.quantile(0.99)),
        }
    )

    return result


def profile_timestamp(df: pd.DataFrame, timestamp_column: str) -> dict[str, Any]:
    """Profile timestamp quality and sampling regularity."""
    ts = pd.to_datetime(df[timestamp_column], errors="coerce")

    result: dict[str, Any] = {
        "column": timestamp_column,
        "parse_failures": int(ts.isna().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "min": json_safe(ts.min()) if not ts.dropna().empty else None,
        "max": json_safe(ts.max()) if not ts.dropna().empty else None,
    }

    deltas = ts.sort_values().diff().dropna()

    if deltas.empty:
        result["sampling"] = {}
        return result

    counts = deltas.value_counts()

    result["sampling"] = {
        "most_common_interval": str(counts.index[0]),
        "most_common_interval_count": int(counts.iloc[0]),
        "unique_intervals": int(counts.shape[0]),
        "largest_interval": str(deltas.max()),
        "smallest_interval": str(deltas.min()),
    }

    # Report the five most common intervals for diagnosing irregular data.
    result["sampling"]["top_intervals"] = [
        {
            "interval": str(interval),
            "count": int(count),
        }
        for interval, count in counts.head(5).items()
    ]

    return result


def profile_sheet(df: pd.DataFrame, sheet_name: str) -> dict[str, Any]:
    """Create a structural + statistical profile for one worksheet."""
    profile: dict[str, Any] = {
        "sheet_name": sheet_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "is_empty": bool(df.empty),
        "columns": [],
    }

    if df.empty:
        return profile

    timestamp_candidates = [
        str(col)
        for col in df.columns
        if str(col).strip().lower() in {"timestamp", "time", "datetime", "date"}
    ]

    profile["timestamp_candidates"] = timestamp_candidates

    for column in df.columns:
        name = str(column)
        series = df[column]

        column_info: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "is_object": bool(pd.api.types.is_object_dtype(series)),
            "missing_count": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": [json_safe(v) for v in series.dropna().head(5).tolist()],
        }

        family = infer_tag_family(name)
        if family:
            column_info["industrial_tag_family_hint"] = family

        if column_info["is_numeric"]:
            column_info["statistics"] = profile_numeric_column(series)

        profile["columns"].append(column_info)

    if timestamp_candidates:
        profile["timestamp"] = profile_timestamp(df, timestamp_candidates[0])

    profile["numeric_column_count"] = int(
        sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    )

    profile["non_numeric_column_count"] = int(
        len(df.columns) - profile["numeric_column_count"]
    )

    profile["total_missing_cells"] = int(df.isna().sum().sum())

    return profile


def build_sensor_profile(
    profiles: list[dict[str, Any]],
) -> pd.DataFrame:
    """Flatten numeric industrial-tag columns into a practical sensor table."""
    rows: list[dict[str, Any]] = []

    for sheet in profiles:
        if sheet["is_empty"]:
            continue

        for column in sheet["columns"]:
            if not column["is_numeric"]:
                continue

            stats = column.get("statistics", {})

            rows.append(
                {
                    "sheet": sheet["sheet_name"],
                    "tag": column["name"],
                    "tag_family_hint": column.get("industrial_tag_family_hint"),
                    "dtype": column["dtype"],
                    "missing_count": column["missing_count"],
                    "missing_fraction": column["missing_fraction"],
                    "unique_count": column["unique_count"],
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "median": stats.get("median"),
                    "q01": stats.get("q01"),
                    "q99": stats.get("q99"),
                }
            )

    return pd.DataFrame(rows)


def read_workbook(
    input_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read every worksheet, save non-empty sheets as CSV, and profile them."""
    workbook = pd.ExcelFile(input_path)

    profiles: list[dict[str, Any]] = []

    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(input_path, sheet_name=sheet_name)

        profile = profile_sheet(df, sheet_name)
        profiles.append(profile)

        # Preserve a processed copy for downstream pipeline stages.
        if not df.empty:
            output_csv = output_dir / f"{safe_filename(sheet_name)}.csv"
            df.to_csv(output_csv, index=False)

    workbook_info = {
        "file_name": input_path.name,
        "file_path": str(input_path.resolve()),
        "sheet_count": len(workbook.sheet_names),
        "sheet_names": [str(name) for name in workbook.sheet_names],
        "profiles": profiles,
    }

    return workbook_info, profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile an industrial Excel workbook."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Excel file (default: {DEFAULT_INPUT})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {input_path}\n"
            "Check that RCSD-1YD.xlsx is in data/raw/refinery/."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INDUSTRIAL EXCEL PROFILER")
    print("=" * 60)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print()

    workbook_info, profiles = read_workbook(
        input_path=input_path,
        output_dir=output_dir,
    )

    # Main JSON report.
    profile_json = output_dir / "workbook_profile.json"
    with profile_json.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(workbook_info),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Flat sensor-oriented CSV for the next ingestion stage.
    sensor_df = build_sensor_profile(profiles)
    sensor_csv = output_dir / "sensor_profile.csv"
    sensor_df.to_csv(sensor_csv, index=False)

    print("Sheets:")
    for profile in profiles:
        status = "EMPTY" if profile["is_empty"] else "DATA"
        print(
            f"  {profile['sheet_name']}: "
            f"{profile['rows']} rows x {profile['columns']} columns [{status}]"
        )

    print()
    print(f"Created: {profile_json}")
    print(f"Created: {sensor_csv}")

    non_empty = [p for p in profiles if not p["is_empty"]]
    numeric_columns = sum(p.get("numeric_column_count", 0) for p in non_empty)

    print()
    print(f"Non-empty sheets     : {len(non_empty)}")
    print(f"Numeric columns      : {numeric_columns}")

    timestamp_columns = []
    for profile in non_empty:
        timestamp_columns.extend(profile.get("timestamp_candidates", []))

    print(f"Timestamp candidates  : {timestamp_columns}")

    print()
    print("Next stage: use this profile to build domain discovery and")
    print("sensor/entity mapping. Do not modify the raw Excel file.")
    print("=" * 60)


if __name__ == "__main__":
    main()
