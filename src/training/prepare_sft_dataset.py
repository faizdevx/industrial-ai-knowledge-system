"""
Prepare validated grounded examples for local SFT/LoRA training.

Input:
  data/processed/refinery/training/splits/
    train.jsonl
    validation.jsonl
    test.jsonl
  data/processed/refinery/training/training_validation_report.json

Output:
  data/processed/refinery/training/sft/
    train.jsonl
    validation.jsonl
    test.jsonl
    dataset_manifest.json

Format:
  One JSON object per line with:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {...}
    }

The original source records are preserved inside metadata. The training
prompt includes the grounded context so the model learns evidence-based
reasoning rather than memorizing isolated answers.

This script does NOT call an LLM and does NOT alter the source examples.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING = ROOT / "data" / "processed" / "refinery" / "training"
DEFAULT_SPLITS = DEFAULT_TRAINING / "splits"
DEFAULT_OUTPUT = DEFAULT_TRAINING / "sft"
DEFAULT_VALIDATION = DEFAULT_TRAINING / "training_validation_report.json"


SYSTEM_PROMPT = """
You are an industrial knowledge assistant.

Use only the evidence supplied in the user message.

Rules:
- Do not invent facts, equipment assignments, thresholds, failures, causes,
  or maintenance actions.
- A statistical anomaly is a candidate abnormal event, not a confirmed fault.
- A document mention is not proof of physical asset assignment.
- Correlation is not causation.
- Distinguish documented facts, statistical observations, and hypotheses.
- Preserve uncertainty when evidence is insufficient.
- When source information is available, identify it explicitly.
- Answer in a concise engineering style.
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def render_context(item: dict[str, Any]) -> str:
    context = item.get("context", [])
    source_records = item.get("source_records", [])

    lines = ["EVIDENCE CONTEXT"]

    if context:
        lines.append(compact_json(context))

    if source_records:
        lines.append("")
        lines.append("SOURCE RECORDS")
        lines.append(compact_json(source_records))

    lines.append("")
    lines.append("GROUNDING POLICY")
    lines.append(
        compact_json(
            item.get(
                "grounding_policy",
                {
                    "source_grounded": True,
                    "statistical_event_is_not_confirmed_fault": True,
                    "document_mention_is_not_physical_assignment": True,
                    "correlation_is_not_causation": True,
                },
            )
        )
    )

    return "\n".join(lines)


def transform(item: dict[str, Any], split: str) -> dict[str, Any]:
    instruction = str(item.get("instruction", "")).strip()
    response = str(item.get("response", "")).strip()

    user_content = (
        f"{instruction}\n\n"
        f"{render_context(item)}\n\n"
        "Provide the grounded answer using only this evidence."
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ],
        "metadata": {
            "example_id": item.get("id"),
            "task_type": item.get("task_type"),
            "difficulty": item.get("difficulty"),
            "split": split,
            "source_records": item.get("source_records", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation-report", default=str(DEFAULT_VALIDATION))
    args = parser.parse_args()

    splits = Path(args.splits)
    output = Path(args.output)
    validation_report_path = Path(args.validation_report)

    if validation_report_path.exists():
        report = load_json(validation_report_path)
        if report.get("invalid_examples", 0) > 0:
            raise RuntimeError(
                "Training validation report contains invalid examples. "
                "Fix them before preparing SFT data."
            )

    required = [
        splits / "train.jsonl",
        splits / "validation.jsonl",
        splits / "test.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing split files:")
        for item in missing:
            print(f"  - {item}")
        raise FileNotFoundError(
            "Create the train/validation/test split first."
        )

    output.mkdir(parents=True, exist_ok=True)

    counts = {}

    for split in ("train", "validation", "test"):
        src = splits / f"{split}.jsonl"
        dst = output / f"{split}.jsonl"

        items = load_jsonl(src)

        with dst.open("w", encoding="utf-8") as f:
            for item in items:
                transformed = transform(item, split)
                f.write(
                    json.dumps(
                        transformed,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        counts[split] = len(items)
        print(f"{split:12}: {len(items)} examples")

    manifest = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "format": "messages_jsonl",
        "system_prompt_policy": "grounded_industrial_assistant",
        "source_splits": str(splits),
        "counts": counts,
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
        "notes": [
            "Source records are preserved in metadata.",
            "Evidence context is included in the user message.",
            "No LLM-generated examples are introduced by this conversion.",
            "This format is intended for later local SFT/LoRA tooling.",
            "Human review of examples remains required before model training.",
        ],
    }

    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SFT DATASET PREPARATION")
    print("=" * 72)
    print(f"Output: {output}")
    print()
    print("Created:")
    print(f"  {output / 'train.jsonl'}")
    print(f"  {output / 'validation.jsonl'}")
    print(f"  {output / 'test.jsonl'}")
    print(f"  {manifest_path}")
    print()
    print("Next stage: inspect the formatted examples before training.")
    print("=" * 72)


if __name__ == "__main__":
    main()
