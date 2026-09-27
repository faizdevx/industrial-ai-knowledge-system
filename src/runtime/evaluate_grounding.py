"""
Run an initial evaluation suite for grounded local RAG.

Automatic checks intentionally measure grounding mechanics, not truth
itself. Human annotation is still required for factual correctness,
engineering validity, and usefulness.

Usage:
    uv run python src/runtime/evaluate_grounding.py
    uv run python src/runtime/evaluate_grounding.py --model qwen2.5:7b

Inputs:
    evaluation_cases.json
    current BGE embeddings + metadata
    current Neo4j graph
    local Ollama

Outputs:
    data/processed/refinery/evaluation/
        evaluation_results.jsonl
        evaluation_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

RUNTIME_DIR = Path(__file__).resolve().parents[0]
ROOT = RUNTIME_DIR.parents[2]

CASES = ROOT / "evaluation_cases.json"
DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
OUTPUT_DIR = ROOT / "data" / "processed" / "refinery" / "evaluation"
EMBEDDINGS = DOC_DIR / "embeddings.npy"
METADATA = DOC_DIR / "chunk_metadata.jsonl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

CITATION_RE = re.compile(
    r"\[(DOC:[^\]]+|ENTITY:[^\]]+|EVENT:[^\]]+)\]"
)

SYSTEM_PROMPT = """
You are evaluating an industrial knowledge-system answer.

Answer ONLY from the supplied evidence.

Rules:
- Do not invent facts.
- Statistical events are candidate anomalies, not confirmed faults.
- A document mention is not proof of physical asset assignment.
- Correlation is not causation.
- Distinguish documented facts, statistical observations, and hypotheses.
- State when evidence is insufficient.
- Cite retrieved evidence using the exact supplied identifiers.

Use these sections:
Answer:
Evidence:
Assessment:
Uncertainty:
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def semantic_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ vector
    indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(indices, start=1):
        meta = metadata[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[int(idx)]),
            "chunk_id": meta.get("chunk_id"),
            "file": meta.get("file_name") or meta.get("file"),
            "page": meta.get("page"),
            "text": meta.get("text", ""),
        })
    return results


def graph_evidence(driver, chunk_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not chunk_ids:
        return {"entities": [], "events": []}

    query = """
    MATCH (d:DocumentChunk)-[:MENTIONS]->(e:Entity)
    WHERE d.id IN $chunk_ids
    OPTIONAL MATCH (ev:Event)-[:DETECTED_BY]->(e)
    RETURN
        d.id AS chunk_id,
        e.id AS entity_id,
        e.source_column AS source_column,
        e.candidate_class AS candidate_class,
        ev.id AS event_id,
        ev.event_type AS event_type,
        ev.status AS event_status,
        ev.start AS event_start,
        ev.end AS event_end,
        ev.peak_value AS peak_value,
        ev.peak_robust_z AS peak_robust_z,
        ev.interpretation AS interpretation
    """

    entities = {}
    events = {}

    with driver.session() as session:
        for row in session.run(query, chunk_ids=chunk_ids):
            if row["entity_id"]:
                key = (row["chunk_id"], row["entity_id"])
                entities[key] = {
                    "chunk_id": row["chunk_id"],
                    "entity_id": row["entity_id"],
                    "source_column": row["source_column"],
                    "candidate_class": row["candidate_class"],
                }

            if row["event_id"]:
                key = (row["entity_id"], row["event_id"])
                events[key] = {
                    "entity_id": row["entity_id"],
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "status": row["event_status"],
                    "start": row["event_start"],
                    "end": row["event_end"],
                    "peak_value": row["peak_value"],
                    "peak_robust_z": row["peak_robust_z"],
                    "interpretation": row["interpretation"],
                }

    return {
        "entities": list(entities.values()),
        "events": list(events.values()),
    }


def build_prompt(
    question: str,
    docs: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
) -> str:
    parts = [SYSTEM_PROMPT, "", "QUESTION", question, "", "DOCUMENT EVIDENCE"]

    for item in docs:
        parts.append(
            f"[DOC:{item['file']} p.{item['page']} "
            f"chunk:{item['chunk_id']}]"
        )
        parts.append(item["text"])
        parts.append("")

    parts.append("GRAPH ENTITY EVIDENCE")
    for item in graph["entities"]:
        parts.append(
            f"[ENTITY:{item['entity_id']}] "
            f"column={item['source_column']} "
            f"class={item['candidate_class']}"
        )

    parts.append("")
    parts.append("STATISTICAL EVENT EVIDENCE")
    for item in graph["events"]:
        parts.append(
            f"[EVENT:{item['event_id']}] "
            f"sensor={item['entity_id']} "
            f"type={item['event_type']} "
            f"status={item['status']} "
            f"start={item['start']} "
            f"end={item['end']} "
            f"peak={item['peak_value']} "
            f"robust_z={item['peak_robust_z']}"
        )
        if item["interpretation"]:
            parts.append(f"interpretation={item['interpretation']}")

    parts.append("")
    parts.append("Generate the grounded answer now.")
    return "\n".join(parts)


def ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}"
        ) from exc

    if body.get("error"):
        raise RuntimeError(body["error"])

    text = body.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty answer.")
    return text


def validate_citations(
    answer: str,
    docs: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    citations = CITATION_RE.findall(answer)

    valid = set()
    for doc in docs:
        valid.add(
            f"DOC:{doc['file']} p.{doc['page']} chunk:{doc['chunk_id']}"
        )

    valid.update(
        f"ENTITY:{x['entity_id']}"
        for x in graph["entities"]
    )
    valid.update(
        f"EVENT:{x['event_id']}"
        for x in graph["events"]
    )

    invalid = [citation for citation in citations if citation not in valid]

    return {
        "citation_count": len(citations),
        "valid_citation_count": len(citations) - len(invalid),
        "invalid_citation_count": len(invalid),
        "invalid_citations": invalid,
        "citation_grounding_pass": len(invalid) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
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

    if not CASES.exists():
        raise FileNotFoundError(f"Missing evaluation cases: {CASES}")
    if not EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing embeddings: {EMBEDDINGS}")
    if not METADATA.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA}")

    cases = load_json(CASES)["cases"]
    if args.limit > 0:
        cases = cases[:args.limit]

    metadata = load_jsonl(METADATA)
    embeddings = np.load(EMBEDDINGS)
    encoder = SentenceTransformer(EMBED_MODEL)

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "evaluation_results.jsonl"

    results = []

    print("=" * 72)
    print("GROUNDED RAG EVALUATION")
    print("=" * 72)
    print(f"Cases         : {len(cases)}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"LLM           : {args.model}")
    print()

    try:
        driver.verify_connectivity()

        with results_path.open("w", encoding="utf-8") as out:
            for n, case in enumerate(cases, start=1):
                question = case["question"]
                print(f"[{n}/{len(cases)}] {question}")

                docs = semantic_search(
                    question,
                    encoder,
                    embeddings,
                    metadata,
                    args.top_k,
                )

                chunk_ids = [
                    d["chunk_id"]
                    for d in docs
                    if d.get("chunk_id")
                ]
                graph = graph_evidence(driver, chunk_ids)

                prompt = build_prompt(question, docs, graph)
                answer = ollama(prompt, args.model)

                citation_metrics = validate_citations(
                    answer,
                    docs,
                    graph,
                )

                result = {
                    "case_id": case["id"],
                    "question": question,
                    "task_type": case["task_type"],
                    "model": args.model,
                    "retrieval": {
                        "top_k": args.top_k,
                        "document_count": len(docs),
                        "entities_linked": len(graph["entities"]),
                        "events_linked": len(graph["events"]),
                        "top_score": docs[0]["score"] if docs else None,
                    },
                    "answer": answer,
                    "automatic_checks": {
                        **citation_metrics,
                        "has_answer_section": "Answer:" in answer,
                        "has_evidence_section": "Evidence:" in answer,
                        "has_uncertainty_section": "Uncertainty:" in answer,
                        "contains_insufficient_language": any(
                            phrase in answer.lower()
                            for phrase in (
                                "insufficient",
                                "not enough evidence",
                                "cannot establish",
                                "cannot confirm",
                                "not confirmed",
                            )
                        ),
                    },
                    "human_review": {
                        "factual_correctness": None,
                        "engineering_validity": None,
                        "provenance_correctness": None,
                        "unsupported_claim_rate": None,
                        "usefulness": None,
                        "review_notes": None,
                    },
                    "generated_at": utc_now(),
                }

                out.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ) + "\n"
                )
                results.append(result)

    finally:
        driver.close()

    total = len(results)
    citation_pass = sum(
        r["automatic_checks"]["citation_grounding_pass"]
        for r in results
    )
    section_pass = sum(
        r["automatic_checks"]["has_answer_section"]
        and r["automatic_checks"]["has_evidence_section"]
        and r["automatic_checks"]["has_uncertainty_section"]
        for r in results
    )

    summary = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "cases": total,
        "automatic_metrics": {
            "citation_grounding_pass_rate": (
                citation_pass / total if total else 0.0
            ),
            "required_section_pass_rate": (
                section_pass / total if total else 0.0
            ),
        },
        "human_metrics": {
            "factual_correctness": "manual annotation required",
            "engineering_validity": "manual annotation required",
            "unsupported_claim_rate": "manual annotation required",
            "provenance_correctness": "manual annotation required",
            "usefulness": "manual annotation required",
        },
        "important": (
            "Automatic checks do not establish engineering truth. "
            "Human review is required before treating an answer as correct."
        ),
    }

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Cases                       : {total}")
    print(
        "Citation grounding pass rate: "
        f"{summary['automatic_metrics']['citation_grounding_pass_rate']:.2%}"
    )
    print(
        "Required sections pass rate : "
        f"{summary['automatic_metrics']['required_section_pass_rate']:.2%}"
    )
    print("Human factual review        : required")
    print(f"Results                     : {results_path}")
    print(f"Summary                     : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
