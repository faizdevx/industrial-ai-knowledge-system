"""
Evaluate the LoRA adapter on the held-out SFT test set.

This evaluates the adapted model itself, not the retrieval system.

Inputs:
  data/processed/refinery/training/sft/test.jsonl
  data/models/refinery-lora/
    adapter_config.json
    adapter_model.safetensors
    tokenizer files

The adapter is loaded on top of the same base model used for training.

Outputs:
  data/processed/refinery/evaluation/
    sft_test_results.jsonl
    sft_test_summary.json

Usage:
  uv run python src/evaluation/evaluate_sft.py --limit 5
  uv run python src/evaluation/evaluate_sft.py

For a larger base model:
  --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEST = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "training"
    / "sft"
    / "test.jsonl"
)

DEFAULT_ADAPTER = (
    ROOT
    / "data"
    / "models"
    / "refinery-lora"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "evaluation"
)

DEFAULT_MODEL = os.getenv("SFT_MODEL", "Qwen/Qwen3-0.6B")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_grounding_guard(text: str) -> bool:
    lowered = normalize(text)
    markers = (
        "not confirmed",
        "does not prove",
        "cannot confirm",
        "insufficient",
        "not enough evidence",
        "not causation",
        "does not establish",
        "[doc:",
        "[entity:",
        "[event:",
    )
    return any(marker in lowered for marker in markers)


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    prompt_messages = [
        m for m in messages
        if m.get("role") != "assistant"
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for tokenizers without a chat template.
    blocks = []
    for item in prompt_messages:
        blocks.append(
            f"{item.get('role', 'user').upper()}: "
            f"{item.get('content', '')}"
        )
    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    output = model.generate(**inputs, **kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU loading/inference.",
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    adapter_path = Path(args.adapter)
    output_dir = Path(args.output)

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test set: {test_path}")

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Missing LoRA adapter: {adapter_path}\n"
            "Complete a training run first."
        )

    examples = load_jsonl(test_path)

    if args.limit > 0:
        examples = examples[:args.limit]

    if not examples:
        raise RuntimeError("Test set is empty.")

    print("=" * 72)
    print("LoRA TEST EVALUATION")
    print("=" * 72)
    print(f"Base model : {args.model}")
    print(f"Adapter    : {adapter_path}")
    print(f"Test cases : {len(examples)}")
    print(f"Max tokens : {args.max_new_tokens}")
    print()

    if args.cpu:
        dtype = torch.float32
        device_map = None
    elif torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
    )

    model.eval()

    if device_map is None:
        device = torch.device("cpu")
        model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "sft_test_results.jsonl"

    results = []

    with result_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(examples, start=1):
            example_id = item.get("metadata", {}).get(
                "example_id",
                f"case_{index}",
            )
            task_type = item.get("metadata", {}).get("task_type")

            messages = item.get("messages", [])
            expected = ""

            for message in messages:
                if message.get("role") == "assistant":
                    expected = message.get("content", "")
                    break

            prompt = build_prompt(tokenizer, messages)

            print(f"[{index}/{len(examples)}] {example_id}")

            prediction = generate(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
            )

            result = {
                "example_id": example_id,
                "task_type": task_type,
                "prediction": prediction,
                "reference": expected,
                "automatic_checks": {
                    "non_empty_prediction": bool(prediction.strip()),
                    "grounding_guard_present": contains_grounding_guard(
                        prediction
                    ),
                    "has_answer_section": "Answer:" in prediction,
                    "has_evidence_section": "Evidence:" in prediction,
                    "has_uncertainty_section": "Uncertainty:" in prediction,
                },
                "generated_at": utc_now(),
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                ) + "\n"
            )
            results.append(result)

    total = len(results)

    non_empty = sum(
        r["automatic_checks"]["non_empty_prediction"]
        for r in results
    )

    guard = sum(
        r["automatic_checks"]["grounding_guard_present"]
        for r in results
    )

    structured = sum(
        r["automatic_checks"]["has_answer_section"]
        and r["automatic_checks"]["has_evidence_section"]
        and r["automatic_checks"]["has_uncertainty_section"]
        for r in results
    )

    summary = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "model": args.model,
        "adapter": str(adapter_path),
        "test_examples": total,
        "automatic_metrics": {
            "non_empty_prediction_rate": non_empty / total,
            "grounding_guard_rate": guard / total,
            "required_structure_rate": structured / total,
        },
        "human_metrics_required": {
            "factual_correctness": None,
            "engineering_validity": None,
            "provenance_correctness": None,
            "unsupported_claim_rate": None,
            "usefulness": None,
        },
        "warning": (
            "Exact string matching against generated answers is not a "
            "sufficient measure of engineering correctness. Human review "
            "and comparison against grounded retrieval are required."
        ),
    }

    summary_path = output_dir / "sft_test_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TEST SUMMARY")
    print("=" * 72)
    print(f"Cases                    : {total}")
    print(
        "Non-empty prediction rate: "
        f"{summary['automatic_metrics']['non_empty_prediction_rate']:.2%}"
    )
    print(
        "Grounding guard rate     : "
        f"{summary['automatic_metrics']['grounding_guard_rate']:.2%}"
    )
    print(
        "Required structure rate  : "
        f"{summary['automatic_metrics']['required_structure_rate']:.2%}"
    )
    print(f"Results                  : {result_path}")
    print(f"Summary                  : {summary_path}")
    print()
    print("Human review remains required.")
    print("=" * 72)


if __name__ == "__main__":
    main()