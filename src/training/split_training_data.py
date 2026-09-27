"""
Create reproducible train/validation/test splits for grounded training data.

Default strategy:
  - deterministic hash split by example id
  - keeps duplicate instruction/response pairs in the same split
  - preserves task-type distribution as much as possible
  - writes a manifest with counts and split rules

This first MVP does NOT claim domain generalization because the current
training examples are mostly derived from one refinery dataset. Time/equipment
generalization needs explicit group metadata before it can be enforced.

Inputs:
  data/processed/refinery/training/training_examples.jsonl
  data/processed/refinery/training/training_validation_report.json

Outputs:
  data/processed/refinery/training/splits/
    train.jsonl
    validation.jsonl
    test.jsonl
    split_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "training"
    / "training_examples.jsonl"
)

DEFAULT_VALIDATION_REPORT = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "training"
    / "training_validation_report.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "training"
    / "splits"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_key(item: dict[str, Any]) -> str:
    """
    Keep exact instruction/response duplicates together.

    This also makes the split deterministic if the input file order changes.
    """
    instruction = str(item.get("instruction", "")).strip().lower()
    response = str(item.get("response", "")).strip().lower()
    return hashlib.sha256(
        f"{instruction}\n{response}".encode("utf-8")
    ).hexdigest()


def bucket_from_hash(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(16**16)


def choose_split(
    value: float,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--validation-report",
        default=str(DEFAULT_VALIDATION_REPORT),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--validation", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    args = parser.parse_args()

    if abs((args.train + args.validation + args.test) - 1.0) > 1e-9:
        raise ValueError("train + validation + test must equal 1.0")

    input_path = Path(args.input)
    report_path = Path(args.validation_report)
    output = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing training examples: {input_path}")

    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing training validation report: {report_path}"
        )

    validation_report = load_json(report_path)

    if validation_report.get("invalid_examples", 0) > 0:
        raise RuntimeError(
            "Training data validation contains invalid examples. "
            "Fix them before splitting."
        )

    examples = load_jsonl(input_path)

    if not examples:
        raise RuntimeError("No training examples found.")

    # Group exact duplicates before splitting.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        groups[group_key(item)].append(item)

    split_data: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for group_id, items in sorted(groups.items()):
        bucket = bucket_from_hash(group_id)
        split = choose_split(
            bucket,
            args.train,
            args.validation,
        )
        split_data[split].extend(items)

    output.mkdir(parents=True, exist_ok=True)

    for split_name, items in split_data.items():
        items = sorted(items, key=lambda x: str(x.get("id", "")))
        with (output / f"{split_name}.jsonl").open(
            "w",
            encoding="utf-8",
        ) as f:
            for item in items:
                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def counts(items: list[dict[str, Any]]) -> dict[str, Any]:
        task_counts = Counter(
            str(x.get("task_type", "unknown"))
            for x in items
        )
        difficulty_counts = Counter(
            str(x.get("difficulty", "unknown"))
            for x in items
        )
        return {
            "total": len(items),
            "by_task_type": dict(sorted(task_counts.items())),
            "by_difficulty": dict(sorted(difficulty_counts.items())),
        }

    manifest = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "source": str(input_path),
        "validation_report": str(report_path),
        "method": {
            "type": "deterministic_hash_group_split",
            "hash": "sha256",
            "duplicate_grouping": "normalized instruction + response",
            "train_ratio": args.train,
            "validation_ratio": args.validation,
            "test_ratio": args.test,
            "seed": "none; deterministic from example content",
        },
        "split_counts": {
            "train": counts(split_data["train"]),
            "validation": counts(split_data["validation"]),
            "test": counts(split_data["test"]),
        },
        "limitations": [
            "The current dataset is primarily one refinery-domain corpus.",
            "This split does not prove unseen-equipment generalization.",
            "This split does not prove unseen-domain generalization.",
            "Explicit group metadata should be added before stricter temporal or equipment splits.",
        ],
        "recommendation": (
            "Use this split for the first pipeline smoke test. For research "
            "evaluation, create explicit time/equipment/source groups and split "
            "by those groups rather than relying only on random-like hashing."
        ),
    }

    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 72)
    print(f"Input examples : {len(examples)}")
    print(f"Duplicate groups: {len(groups)}")
    print()
    for name in ("train", "validation", "test"):
        info = manifest["split_counts"][name]
        print(f"{name.upper():12} : {info['total']}")
    print()
    print(f"Train ratio    : {args.train:.1%}")
    print(f"Validation     : {args.validation:.1%}")
    print(f"Test ratio     : {args.test:.1%}")
    print()
    print(f"Created: {output / 'train.jsonl'}")
    print(f"Created: {output / 'validation.jsonl'}")
    print(f"Created: {output / 'test.jsonl'}")
    print(f"Created: {manifest_path}")
    print()
    print("Important: this is a reproducible baseline split, not")
    print("the final research generalization protocol.")
    print("=" * 72)


if __name__ == "__main__":
    main()
