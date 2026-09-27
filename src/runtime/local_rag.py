"""
Local grounded RAG answerer for the refinery MVP.

Pipeline:
    question
      -> semantic document retrieval
      -> Neo4j entity/event evidence
      -> strict evidence prompt
      -> local Ollama model
      -> answer + evidence references + uncertainty

This stage deliberately keeps the model's role constrained:
it reasons over retrieved evidence; it is not asked to invent missing
asset assignments, faults, thresholds, or causal relationships.

Prerequisite:
    Ollama running locally with a model pulled, for example:
      ollama serve
      ollama pull qwen2.5:7b

You can change the model:
    $env:OLLAMA_MODEL="qwen2.5:7b"

Usage:
    uv run python src/runtime/local_rag.py "compressor instrumentation"
    uv run python src/runtime/local_rag.py "What happened around a pressure anomaly?" --top-k 8
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grounded_qa import (
    MODEL_NAME,
    build_bundle,
    fetch_graph_evidence,
    load_metadata,
    semantic_search,
)

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
OUTPUT_DIR = ROOT / "data" / "processed" / "refinery" / "runtime"
EMBEDDINGS = DOC_DIR / "embeddings.npy"
METADATA = DOC_DIR / "chunk_metadata.jsonl"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


SYSTEM_PROMPT = """
You are the reasoning component of a local industrial knowledge system.

Answer ONLY from the supplied evidence bundle.

Rules:
1. Do not invent equipment names, sensor assignments, limits, causes, failures,
   maintenance actions, or facts that are absent from the evidence.
2. A statistical event is a candidate anomaly, not a confirmed fault.
3. A document mentioning a sensor tag does not prove physical asset assignment.
4. Correlation is not causation.
5. Distinguish documented facts, observed statistical behavior, and hypotheses.
6. When evidence is insufficient, explicitly say that it is insufficient.
7. Preserve uncertainty rather than filling gaps.
8. Cite evidence inline using the exact identifiers supplied in the evidence,
   such as [DOC:2.pdf p.454 chunk:...], [ENTITY:...], [EVENT:...].
9. Do not cite evidence that does not support the statement.
10. Prefer a concise engineering-style answer.

Return plain text with these sections:
Answer:
Evidence:
Assessment:
Uncertainty:
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ollama_generate(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{OLLAMA_URL}. Start Ollama and verify the local API is available."
        ) from exc

    if "error" in body:
        raise RuntimeError(f"Ollama returned an error: {body['error']}")

    answer = body.get("response")
    if not answer:
        raise RuntimeError("Ollama returned no response text.")

    return answer.strip()


def evidence_prompt(bundle: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("QUESTION")
    lines.append(bundle["query"])
    lines.append("")
    lines.append("DOCUMENT EVIDENCE")

    for item in bundle.get("document_evidence", []):
        citation = (
            f"[DOC:{item.get('file')} p.{item.get('page')} "
            f"chunk:{item.get('chunk_id')}]"
        )
        lines.append(citation)
        lines.append(item.get("text", "").strip())
        lines.append("")

    lines.append("GRAPH ENTITY EVIDENCE")
    for entity in bundle.get("graph_evidence", {}).get("entities", []):
        lines.append(
            f"[ENTITY:{entity.get('entity_id')}] "
            f"column={entity.get('source_column')} "
            f"class={entity.get('candidate_class')} "
            f"tag_family={entity.get('tag_family')} "
            f"confidence={entity.get('confidence')}"
        )

    lines.append("")
    lines.append("STATISTICAL EVENT EVIDENCE")
    for event in bundle.get("graph_evidence", {}).get("events", []):
        lines.append(
            f"[EVENT:{event.get('event_id')}] "
            f"sensor={event.get('entity_id')} "
            f"type={event.get('event_type')} "
            f"start={event.get('start')} "
            f"end={event.get('end')} "
            f"peak={event.get('peak_value')} "
            f"robust_z={event.get('peak_robust_z')} "
            f"direction={event.get('direction')} "
            f"status={event.get('status')}"
        )
        if event.get("interpretation"):
            lines.append(f"interpretation={event['interpretation']}")

    lines.append("")
    lines.append("Answer the question using only the evidence above.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "industrial-ai"),
    )
    args = parser.parse_args()

    print("=" * 72)
    print("LOCAL GROUNDED RAG")
    print("=" * 72)
    print(f"Query          : {args.query}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"LLM            : {OLLAMA_MODEL}")
    print()

    if not EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing embeddings: {EMBEDDINGS}")

    metadata = load_metadata(METADATA)
    embeddings = np.load(EMBEDDINGS)
    model = SentenceTransformer(MODEL_NAME)

    hits = semantic_search(
        args.query,
        model,
        embeddings,
        metadata,
        max(1, args.top_k),
    )

    chunk_ids = [h["chunk_id"] for h in hits if h.get("chunk_id")]

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    try:
        driver.verify_connectivity()
        graph = fetch_graph_evidence(driver, chunk_ids)
    finally:
        driver.close()

    bundle = build_bundle(args.query, hits, graph)
    prompt = SYSTEM_PROMPT + "\n\n" + evidence_prompt(bundle)

    print(
        f"Retrieved documents : {len(bundle['document_evidence'])}"
    )
    print(
        f"Linked entities     : "
        f"{len(bundle['graph_evidence']['entities'])}"
    )
    print(
        f"Linked events       : "
        f"{len(bundle['graph_evidence']['events'])}"
    )
    print()
    print("Generating locally...")
    print()

    answer = ollama_generate(prompt)

    result = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "query": args.query,
        "model": OLLAMA_MODEL,
        "embedding_model": MODEL_NAME,
        "answer": answer,
        "evidence_bundle": bundle,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in args.query.lower()
    ).strip("_")[:80] or "query"

    output_path = OUTPUT_DIR / f"rag_{safe_name}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print("=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(answer)
    print()
    print(f"Saved result: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
