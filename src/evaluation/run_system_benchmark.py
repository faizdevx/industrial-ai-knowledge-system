"""
End-to-end benchmark for the Industrial AI Knowledge System.

Runs the real Agent V2 pipeline for a fixed benchmark set and records:
  - tools selected
  - document retrieval
  - graph evidence
  - time-series evidence
  - answer
  - basic automatic grounding checks

It intentionally DOES NOT assign a correctness score automatically.
Engineering correctness requires review against the evidence.

Outputs:
  data/processed/refinery/evaluation/system_benchmark_results.jsonl
  data/processed/refinery/evaluation/system_benchmark_summary.json

Usage:
  uv run python src/evaluation/run_system_benchmark.py --limit 2
  uv run python src/evaluation/run_system_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runtime.agent_runtime_v2 import (
    EMBED_MODEL,
    EMBEDDINGS,
    OLLAMA_MODEL,
    choose_tools,
    document_search,
    event_lookup,
    graph_lookup,
    load_metadata,
    ollama_generate,
    render_prompt,
    run_timeseries_tools,
)
from src.runtime.timeseries_tool import available_sensors

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "system_benchmark_cases.json"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "evaluation"
)

METADATA = (
    ROOT
    / "data"
    / "processed"
    / "refinery"
    / "documents"
    / "chunk_metadata.jsonl"
)

DEFAULT_URI = "bolt://localhost:7687"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def grounding_checks(
    answer: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    lowered = re.sub(r"\s+", " ", answer.lower()).strip()

    has_section_structure = all(
        section in lowered
        for section in (
            "answer:",
            "evidence:",
            "assessment:",
            "uncertainty:",
        )
    )

    citations = re.findall(
        r"\[(DOC:[^\]]+|ENTITY:[^\]]+|EVENT:[^\]]+|TS:[^\]]+)\]",
        answer,
    )

    valid_citations = []

    valid_doc_citations = {
        f"DOC:{item['file']} p.{item['page']} chunk:{item['chunk_id']}"
        for item in evidence["documents"]
    }

    valid_entity_citations = {
        f"ENTITY:{item['entity_id']}"
        for item in evidence["graph"]["entities"]
    }

    valid_event_citations = {
        f"EVENT:{item['event_id']}"
        for item in evidence["graph"]["events"]
    }

    for citation in citations:
        if citation in (
            valid_doc_citations
            | valid_entity_citations
            | valid_event_citations
        ):
            valid_citations.append(citation)

    uncertainty_terms = (
        "insufficient",
        "not confirmed",
        "does not prove",
        "cannot confirm",
        "cannot establish",
        "not enough evidence",
    )

    mentions_fault_language = any(
        phrase in lowered
        for phrase in (
            "equipment has failed",
            "confirmed failure",
            "machine failed",
            "bearing failure",
        )
    )

    return {
        "has_required_sections": has_section_structure,
        "citation_count": len(citations),
        "valid_citation_count": len(valid_citations),
        "invalid_citation_count": len(citations) - len(valid_citations),
        "citation_grounding_pass": len(citations) == len(valid_citations),
        "has_uncertainty_language": any(
            term in lowered
            for term in uncertainty_terms
        ),
        "contains_failure_language": mentions_fault_language,
        "has_non_empty_answer": bool(answer.strip()),
    }


def run_case(
    case: dict[str, Any],
    encoder: SentenceTransformer,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    uri: str,
    user: str,
    password: str,
    top_k: int,
) -> dict[str, Any]:
    question = case["question"]
    tools = choose_tools(question)

    documents = document_search(
        question,
        encoder,
        embeddings,
        metadata,
        top_k,
    )

    graph = {
        "entities": [],
        "events": [],
    }

    event_results = []
    timeseries = {
        "identified_sensors": [],
        "summaries": [],
        "recent_values": [],
        "events": [],
        "comparisons": [],
    }

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
    )

    try:
        driver.verify_connectivity()

        if "graph_lookup" in tools:
            graph = graph_lookup(
                driver,
                question,
                limit=20,
            )

        if "event_lookup" in tools:
            event_results = event_lookup(
                driver,
                question,
                limit=20,
            )

    finally:
        driver.close()

    if "timeseries" in tools:
        try:
            timeseries = run_timeseries_tools(
                question,
                available_sensors(),
            )
        except Exception as exc:
            timeseries = {
                "error": str(exc),
            }

    evidence = {
        "schema_version": "2.0",
        "query": question,
        "tools_used": tools,
        "documents": documents,
        "graph": graph,
        "event_results": event_results,
        "timeseries": timeseries,
        "policies": {
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
            "insufficient_evidence_must_be_stated": True,
        },
    }

    answer = ollama_generate(
        render_prompt(evidence)
    )

    automatic = grounding_checks(
        answer,
        evidence,
    )

    return {
        "case_id": case["id"],
        "category": case["category"],
        "question": question,
        "answer": answer,
        "tools_used": tools,
        "retrieval": {
            "document_count": len(documents),
            "top_document_score": (
                documents[0]["score"]
                if documents
                else None
            ),
            "entity_count": len(graph["entities"]),
            "graph_event_count": len(graph["events"]),
            "event_tool_count": len(event_results),
            "timeseries_sensor_count": len(
                timeseries.get(
                    "identified_sensors",
                    [],
                )
            ),
        },
        "automatic_checks": automatic,
        "evidence_trace": evidence,
        "human_review": {
            "factual_correctness": None,
            "engineering_validity": None,
            "provenance_correctness": None,
            "temporal_reasoning_correctness": None,
            "unsupported_claim_rate": None,
            "usefulness": None,
            "review_notes": None,
        },
        "generated_at": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
    )
    parser.add_argument(
        "--password",
        default="industrial-ai",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)

    if not cases_path.exists():
        raise FileNotFoundError(
            f"Missing benchmark cases: {cases_path}"
        )

    if not EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {EMBEDDINGS}"
        )

    if not METADATA.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA}"
        )

    cases = load_json(cases_path)["cases"]

    if args.limit > 0:
        cases = cases[:args.limit]

    metadata = load_metadata()
    embeddings = np.load(EMBEDDINGS)
    encoder = SentenceTransformer(EMBED_MODEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = (
        OUTPUT_DIR / "system_benchmark_results.jsonl"
    )

    results = []

    print("=" * 72)
    print("END-TO-END SYSTEM BENCHMARK")
    print("=" * 72)
    print(f"Cases         : {len(cases)}")
    print(f"Embedding     : {EMBED_MODEL}")
    print(f"LLM           : {OLLAMA_MODEL}")
    print()

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for index, case in enumerate(
            cases,
            start=1,
        ):
            print(
                f"[{index}/{len(cases)}] "
                f"{case['id']} | "
                f"{case['category']}"
            )

            result = run_case(
                case,
                encoder,
                embeddings,
                metadata,
                args.uri,
                args.user,
                args.password,
                args.top_k,
            )

            results.append(result)

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    total = len(results)

    def avg(metric: str) -> float:
        if not total:
            return 0.0

        return sum(
            1
            for result in results
            if result["automatic_checks"].get(metric)
        ) / total

    summary = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "cases": total,
        "automatic_metrics": {
            "non_empty_answer_rate": avg(
                "has_non_empty_answer"
            ),
            "required_structure_rate": avg(
                "has_required_sections"
            ),
            "citation_grounding_pass_rate": avg(
                "citation_grounding_pass"
            ),
            "uncertainty_language_rate": avg(
                "has_uncertainty_language"
            ),
        },
        "retrieval": {
            "average_top_document_score": (
                sum(
                    r["retrieval"]["top_document_score"]
                    for r in results
                    if r["retrieval"]["top_document_score"]
                    is not None
                )
                / max(
                    1,
                    sum(
                        1
                        for r in results
                        if r["retrieval"]["top_document_score"]
                        is not None
                    )
                )
            ),
            "average_documents_per_case": (
                sum(
                    r["retrieval"]["document_count"]
                    for r in results
                ) / max(1, total)
            ),
            "average_entities_per_case": (
                sum(
                    r["retrieval"]["entity_count"]
                    for r in results
                ) / max(1, total)
            ),
        },
        "human_metrics": {
            "factual_correctness": "manual",
            "engineering_validity": "manual",
            "provenance_correctness": "manual",
            "temporal_reasoning_correctness": "manual",
            "unsupported_claim_rate": "manual",
            "usefulness": "manual",
        },
        "interpretation": (
            "Automatic metrics describe system mechanics only. "
            "They do not establish engineering correctness or safety."
        ),
    }

    summary_path = (
        OUTPUT_DIR / "system_benchmark_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"Cases                       : {total}")
    print(
        "Non-empty answer rate      : "
        f"{summary['automatic_metrics']['non_empty_answer_rate']:.2%}"
    )
    print(
        "Required structure rate    : "
        f"{summary['automatic_metrics']['required_structure_rate']:.2%}"
    )
    print(
        "Citation grounding rate    : "
        f"{summary['automatic_metrics']['citation_grounding_pass_rate']:.2%}"
    )
    print(
        "Uncertainty language rate  : "
        f"{summary['automatic_metrics']['uncertainty_language_rate']:.2%}"
    )
    print()
    print(f"Results : {results_path}")
    print(f"Summary : {summary_path}")
    print()
    print(
        "Human review is required for factual and engineering correctness."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
