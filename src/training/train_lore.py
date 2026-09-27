"""
LoRA SFT training for the Industrial AI Knowledge System.

Default model:
    Qwen/Qwen3-0.6B

Why 0.6B first:
    This is a smoke-test configuration that is practical for local
    experimentation. Move to Qwen/Qwen3-4B or another suitable model only
    after the pipeline and dataset are validated.

Inputs:
    data/processed/refinery/training/sft/train.jsonl
    data/processed/refinery/training/sft/validation.jsonl

Outputs:
    data/models/refinery-lora/
        adapter_config.json
        adapter_model.safetensors
        tokenizer files
        trainer_state.json
        ...

Install once:
    uv add datasets "trl[peft]" accelerate

For QLoRA later:
    add bitsandbytes and a quantization config after hardware is verified.

Usage:
    uv run python src/training/train_lora.py
    uv run python src/training/train_lora.py --model Qwen/Qwen3-4B-Instruct-2507
    uv run python src/training/train_lora.py --epochs 1 --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA = ROOT / "data" / "processed" / "refinery" / "training" / "sft"
DEFAULT_OUTPUT = ROOT / "data" / "models" / "refinery-lora"

# Small first-pass model. Official Qwen3 model card documents Transformers
# loading and chat-template usage for this model family.
DEFAULT_MODEL = os.getenv("SFT_MODEL", "Qwen/Qwen3-0.6B")


def print_hardware() -> None:
    print("=" * 72)
    print("TRAINING HARDWARE")
    print("=" * 72)
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory     : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("GPU            : none detected")
        print(
            "Warning        : CPU LoRA training will be very slow. "
            "Use this only as a smoke test."
        )
    print()


def load_and_check_dataset(data_dir: Path):
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing: {val_path}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )

    for split in ("train", "validation"):
        if len(dataset[split]) == 0:
            raise ValueError(f"{split} dataset is empty.")

        first = dataset[split][0]

        if "messages" not in first:
            raise ValueError(
                f"{split} dataset does not contain 'messages'."
            )

        if not isinstance(first["messages"], list):
            raise ValueError(
                f"{split} messages field is not a list."
            )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 when supported by the GPU.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 GPU training.",
    )
    args = parser.parse_args()

    print_hardware()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_and_check_dataset(data_dir)

    print("=" * 72)
    print("LOCAL LoRA SFT")
    print("=" * 72)
    print(f"Model              : {args.model}")
    print(f"Train examples     : {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    print(f"Max length         : {args.max_length}")
    print(f"LoRA rank          : {args.lora_r}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Epochs             : {args.epochs}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For Qwen3 and similar decoder-only models, use LoRA on attention and
    # MLP projection modules rather than full-weight fine-tuning.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    use_bf16 = args.bf16
    use_fp16 = args.fp16

    if use_bf16 and not torch.cuda.is_available():
        raise RuntimeError("--bf16 requested but CUDA is not available.")

    if use_fp16 and not torch.cuda.is_available():
        raise RuntimeError("--fp16 requested but CUDA is not available.")

    # TRL currently supports conversational message datasets and PEFT-backed
    # SFT. assistant_only_loss keeps the optimization focused on assistant
    # responses when the chat template exposes the required generation masks.
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(train_result.metrics)

    eval_metrics = trainer.evaluate()
    metrics.update(
        {
            f"eval_{key}": value
            for key, value in eval_metrics.items()
        }
    )

    manifest = {
        "schema_version": "1.0",
        "model": args.model,
        "method": "LoRA_SFT",
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bf16": use_bf16,
            "fp16": use_fp16,
        },
        "metrics": metrics,
        "output_dir": str(output_dir),
        "grounding_policy": {
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
        },
        "note": (
            "Training this adapter does not establish that the underlying "
            "examples are engineering ground truth. Evaluate the base model "
            "and adapted model on a held-out test set."
        ),
    }

    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Adapter/model output: {output_dir}")
    print(f"Training metrics     : {metrics}")
    print(f"Manifest             : {manifest_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
