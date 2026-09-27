"""
Compare base model vs LoRA adapter on the same held-out test set.

This is a behavior comparison, not an engineering-truth benchmark.

Outputs:
  data/processed/refinery/evaluation/
    base_vs_lora_results.jsonl
    base_vs_lora_summary.json

Usage:
  uv run python src/evaluation/compare_base_vs_lora.py --limit 5
  uv run python src/evaluation/compare_base_vs_lora.py

Default base model:
  Qwen/Qwen3-0.6B

Adapter:
  data/models/refinery-lora/
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

    blocks = [
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
        for m in prompt_messages
    ]
    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks)


def quality_checks(text: str) -> dict[str, bool]:
    lowered = re.sub(r"\s+", " ", text.lower()).strip()

    grounding_terms = (
        "not confirmed",
        "does not prove",
        "cannot confirm",
        "insufficient",
        "not enough evidence",
        "does not establish",
        "[doc:",
        "[entity:",
        "[event:",
    )

    return {
        "non_empty": bool(text.strip()),
        "has_answer_section": "answer:" in lowered,
        "has_evidence_section": "evidence:" in lowered,
        "has_uncertainty_section": "uncertainty:" in lowered,
        "has_grounding_guard": any(
            term in lowered for term in grounding_terms
        ),
    }


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
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    output = model.generate(**inputs, **kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def load_models(
    model_name: str,
    adapter_path: Path,
    force_cpu: bool,
):
    if force_cpu:
        dtype = torch.float32
        device_map = None
    elif torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if device_map is not None:
        kwargs["device_map"] = device_map

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs,
    )

    if device_map is None:
        base_model.to(torch.device("cpu"))

    print("Loading LoRA adapter...")
    adapted_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
    )
    adapted_model.eval()

    # Keep a standalone base model reference by reloading it. This is
    # intentionally explicit so the comparison is base vs adapted model.
    print("Reloading clean base model...")
    clean_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs,
    )

    if device_map is None:
        clean_base.to(torch.device("cpu"))

    clean_base.eval()

    return tokenizer, clean_base, adapted_model


def extract_reference(item: dict[str, Any]) -> str:
    for message in item.get("messages", []):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    test_path = Path(args.test)
    adapter_path = Path(args.adapter)
    output_dir = Path(args.output)

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test set: {test_path}")

    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")

    cases = load_jsonl(test_path)

    if args.limit > 0:
        cases = cases[:args.limit]

    if not cases:
        raise RuntimeError("Test set is empty.")

    print("=" * 72)
    print("BASE VS LoRA COMPARISON")
    print("=" * 72)
    print(f"Model     : {args.model}")
    print(f"Adapter   : {adapter_path}")
    print(f"Test cases: {len(cases)}")
    print()

    tokenizer, base_model, lora_model = load_models(
        args.model,
        adapter_path,
        args.cpu,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "base_vs_lora_results.jsonl"

    results = []

    with result_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(cases, start=1):
            example_id = item.get("metadata", {}).get(
                "example_id",
                f"case_{index}",
            )
            task_type = item.get("metadata", {}).get("task_type")
            reference = extract_reference(item)

            prompt = build_prompt(
                tokenizer,
                item.get("messages", []),
            )

            print(f"[{index}/{len(cases)}] {example_id}")

            base_answer = generate(
                base_model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
            )

            lora_answer = generate(
                lora_model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
            )

            base_checks = quality_checks(base_answer)
            lora_checks = quality_checks(lora_answer)

            result = {
                "example_id": example_id,
                "task_type": task_type,
                "reference": reference,
                "base_model": {
                    "answer": base_answer,
                    "checks": base_checks,
                },
                "lora_model": {
                    "answer": lora_answer,
                    "checks": lora_checks,
                },
                "human_review": {
                    "base_factual_correctness": None,
                    "lora_factual_correctness": None,
                    "base_engineering_validity": None,
                    "lora_engineering_validity": None,
                    "base_provenance_correctness": None,
                    "lora_provenance_correctness": None,
                    "notes": None,
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

    def rate(model_key: str, check_key: str) -> float:
        if not results:
            return 0.0
        return sum(
            r[model_key]["checks"][check_key]
            for r in results
        ) / len(results)

    summary = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "model": args.model,
        "adapter": str(adapter_path),
        "cases": len(results),
        "automatic_metrics": {
            "base": {
                "non_empty_rate": rate("base_model", "non_empty"),
                "answer_section_rate": rate(
                    "base_model",
                    "has_answer_section",
                ),
                "evidence_section_rate": rate(
                    "base_model",
                    "has_evidence_section",
                ),
                "uncertainty_section_rate": rate(
                    "base_model",
                    "has_uncertainty_section",
                ),
                "grounding_guard_rate": rate(
                    "base_model",
                    "has_grounding_guard",
                ),
            },
            "lora": {
                "non_empty_rate": rate("lora_model", "non_empty"),
                "answer_section_rate": rate(
                    "lora_model",
                    "has_answer_section",
                ),
                "evidence_section_rate": rate(
                    "lora_model",
                    "has_evidence_section",
                ),
                "uncertainty_section_rate": rate(
                    "lora_model",
                    "has_uncertainty_section",
                ),
                "grounding_guard_rate": rate(
                    "lora_model",
                    "has_grounding_guard",
                ),
            },
        },
        "human_metrics": {
            "factual_correctness": "manual review",
            "engineering_validity": "manual review",
            "provenance_correctness": "manual review",
            "unsupported_claim_rate": "manual review",
            "usefulness": "manual review",
        },
        "warning": (
            "This comparison does not establish which model is more "
            "engineering-correct. Manual review against the evidence is "
            "required."
        ),
    }

    summary_path = output_dir / "base_vs_lora_summary.json"
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
    print("COMPARISON SUMMARY")
    print("=" * 72)

    print("Base model")
    print(
        "  non-empty            : "
        f"{summary['automatic_metrics']['base']['non_empty_rate']:.2%}"
    )
    print(
        "  grounding guard      : "
        f"{summary['automatic_metrics']['base']['grounding_guard_rate']:.2%}"
    )
    print(
        "  required sections    : "
        f"{(
            summary['automatic_metrics']['base']['answer_section_rate']
            + summary['automatic_metrics']['base']['evidence_section_rate']
            + summary['automatic_metrics']['base']['uncertainty_section_rate']
        ) / 3:.2%}"
    )

    print()
    print("LoRA model")
    print(
        "  non-empty            : "
        f"{summary['automatic_metrics']['lora']['non_empty_rate']:.2%}"
    )
    print(
        "  grounding guard      : "
        f"{summary['automatic_metrics']['lora']['grounding_guard_rate']:.2%}"
    )
    print(
        "  required sections    : "
        f"{(
            summary['automatic_metrics']['lora']['answer_section_rate']
            + summary['automatic_metrics']['lora']['evidence_section_rate']
            + summary['automatic_metrics']['lora']['uncertainty_section_rate']
        ) / 3:.2%}"
    )

    print()
    print(f"Results : {result_path}")
    print(f"Summary : {summary_path}")
    print()
    print("Manual engineering review is still required.")
    print("=" * 72)


if __name__ == "__main__":
    main()
