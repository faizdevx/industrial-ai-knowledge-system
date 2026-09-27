"""
Minimal local agent runtime for the Industrial AI Knowledge System.

The agent does deterministic tool selection first, then asks the local LLM to
reason over the collected evidence.

Tools:
  1. document_search   -> local BGE semantic retrieval
  2. graph_lookup      -> Neo4j entities
  3. event_lookup      -> Neo4j statistical events

This is intentionally a transparent first agent implementation. It records
which tools were used so later evaluation can measure tool selection.

Usage:
    uv run python src/runtime/agent_runtime.py "compressor instrumentation"
    uv run python src/runtime/agent_runtime.py "What pressure anomalies were detected?"
    uv run python src/runtime/agent_runtime.py "Does the evidence prove failure?"

Environment:
    OLLAMA_MODEL=qwen2.5:7b
    OLLAMA_URL=http://localhost:11434/api/generate
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=industrial-ai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
EMBEDDINGS = DOC_DIR / "embeddings.npy"
METADATA = DOC_DIR / "chunk_metadata.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed" / "refinery" / "runtime"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

SYSTEM_PROMPT = """
You are the final reasoning component of a local industrial knowledge agent.

Use only the tool evidence supplied to you.

Rules:
- Do not invent equipment, sensor assignments, thresholds, causes, failures,
  or maintenance actions.
- Statistical events are candidate anomalies, not confirmed faults.
- A document mention does not prove physical asset assignment.
- Correlation does not prove causation.
- Separate documented facts, statistical observations, and hypotheses.
- When evidence is insufficient, say so explicitly.
- Cite evidence using the exact IDs supplied by the tools.
- Do not pretend that a retrieved document is proof of a sensor's physical
  installation or asset relationship.

Respond with:
Answer:
Evidence:
Assessment:
Uncertainty:
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_metadata() -> list[dict[str, Any]]:
    if not METADATA.exists():
        raise FileNotFoundError(f"Missing: {METADATA}")

    with METADATA.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def document_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    vector = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ vector
    indices = np.argsort(-scores)[:top_k]

    results = []

    for rank, idx in enumerate(indices, start=1):
        meta = metadata[int(idx)]
        results.append({
            "rank": rank,
            "score": round(float(scores[int(idx)]), 6),
            "chunk_id": meta.get("chunk_id"),
            "file": meta.get("file_name") or meta.get("file"),
            "page": meta.get("page"),
            "text": meta.get("text", ""),
        })

    return results


def graph_lookup(
    driver,
    query: str,
    limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """
    Search entity IDs/source columns by token overlap with the query.

    This is deliberately simple. A later version can use ontology-aware
    entity resolution and explicit asset aliases.
    """
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.-]+", query)
        if len(token) >= 3
    ]

    cypher = """
    MATCH (e:Entity)
    WHERE any(token IN $tokens
              WHERE toLower(coalesce(e.id, '')) CONTAINS token
                 OR toLower(coalesce(e.source_column, '')) CONTAINS token
                 OR toLower(coalesce(e.tag_family, '')) CONTAINS token
                 OR toLower(coalesce(e.candidate_class, '')) CONTAINS token)
    OPTIONAL MATCH (ev:Event)-[:DETECTED_BY]->(e)
    RETURN
      e.id AS entity_id,
      e.source_column AS source_column,
      e.candidate_class AS candidate_class,
      e.tag_family AS tag_family,
      collect(DISTINCT {
        event_id: ev.id,
        event_type: ev.event_type,
        status: ev.status,
        start: ev.start,
        end: ev.end,
        peak_value: ev.peak_value,
        peak_robust_z: ev.peak_robust_z,
        interpretation: ev.interpretation
      })[..10] AS events
    LIMIT $limit
    """

    entities = []
    events = []

    with driver.session() as session:
        rows = session.run(
            cypher,
            tokens=tokens,
            limit=limit,
        )

        seen_events = set()

        for row in rows:
            entities.append({
                "entity_id": row["entity_id"],
                "source_column": row["source_column"],
                "candidate_class": row["candidate_class"],
                "tag_family": row["tag_family"],
            })

            for event in row["events"] or []:
                event_id = event.get("event_id")
                if event_id and event_id not in seen_events:
                    seen_events.add(event_id)
                    events.append({
                        "entity_id": row["entity_id"],
                        **event,
                    })

    return {
        "entities": entities,
        "events": events,
    }


def event_lookup(driver, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Retrieve recent candidate events, optionally filtered by query tokens.
    """
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.-]+", query)
        if len(token) >= 3
    ]

    cypher = """
    MATCH (ev:Event)-[:DETECTED_BY]->(e:Entity)
    WHERE any(token IN $tokens
              WHERE toLower(coalesce(ev.event_type, '')) CONTAINS token
                 OR toLower(coalesce(e.id, '')) CONTAINS token
                 OR toLower(coalesce(e.source_column, '')) CONTAINS token
                 OR toLower(coalesce(ev.interpretation, '')) CONTAINS token)
       OR $tokens = []
    RETURN
      ev.id AS event_id,
      ev.event_type AS event_type,
      ev.status AS status,
      ev.start AS start,
      ev.end AS end,
      ev.peak_value AS peak_value,
      ev.peak_robust_z AS peak_robust_z,
      ev.direction AS direction,
      ev.interpretation AS interpretation,
      e.id AS entity_id
    ORDER BY ev.start DESC
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            tokens=tokens,
            limit=limit,
        )
        return [
            dict(row)
            for row in rows
        ]


def choose_tools(query: str) -> list[str]:
    """
    Transparent first-pass routing policy.

    Every query gets document retrieval.
    Graph/event tools are added for queries that ask about sensors/entities,
    anomalies/events, or failure/condition reasoning.
    """
    q = query.lower()

    tools = ["document_search"]

    graph_terms = (
        "sensor",
        "tag",
        "entity",
        "compressor",
        "bearing",
        "instrument",
        "pressure",
        "temperature",
        "speed",
        "position",
    )

    event_terms = (
        "anomaly",
        "abnormal",
        "event",
        "spike",
        "change point",
        "changed",
        "trend",
        "failure",
        "fault",
        "what happened",
        "before",
        "after",
    )

    if any(term in q for term in graph_terms):
        tools.append("graph_lookup")

    if any(term in q for term in event_terms):
        tools.append("event_lookup")

    return tools


def build_evidence(
    query: str,
    tools_used: list[str],
    documents: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "query": query,
        "tools_used": tools_used,
        "documents": documents,
        "graph_entities": graph["entities"],
        "graph_events": graph["events"],
        "event_results": events,
        "policies": {
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
            "insufficient_evidence_must_be_stated": True,
        },
    }


def render_prompt(evidence: dict[str, Any]) -> str:
    lines = [
        SYSTEM_PROMPT,
        "",
        "USER QUESTION",
        evidence["query"],
        "",
        "TOOLS USED",
        ", ".join(evidence["tools_used"]),
        "",
        "DOCUMENT TOOL RESULTS",
    ]

    for item in evidence["documents"]:
        lines.extend([
            (
                f"[DOC:{item['file']} p.{item['page']} "
                f"chunk:{item['chunk_id']}] score={item['score']}"
            ),
            item["text"],
            "",
        ])

    lines.append("GRAPH ENTITY RESULTS")

    for item in evidence["graph_entities"]:
        lines.append(
            f"[ENTITY:{item['entity_id']}] "
            f"column={item['source_column']} "
            f"class={item['candidate_class']} "
            f"family={item['tag_family']}"
        )

    lines.append("")
    lines.append("GRAPH-LINKED EVENT RESULTS")

    for item in evidence["graph_events"]:
        lines.append(
            f"[EVENT:{item['event_id']}] "
            f"sensor={item['entity_id']} "
            f"type={item['event_type']} "
            f"status={item['status']} "
            f"start={item['start']} "
            f"end={item['end']} "
            f"peak={item['peak_value']} "
            f"robust_z={item['peak_robust_z']}"
        )

    lines.append("")
    lines.append("EVENT TOOL RESULTS")

    for item in evidence["event_results"]:
        lines.append(
            f"[EVENT:{item.get('event_id')}] "
            f"sensor={item.get('entity_id')} "
            f"type={item.get('event_type')} "
            f"status={item.get('status')} "
            f"start={item.get('start')} "
            f"end={item.get('end')} "
            f"peak={item.get('peak_value')} "
            f"robust_z={item.get('peak_robust_z')}"
        )

    lines.append("")
    lines.append("Answer using only these tool results.")

    return "\n".join(lines)


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
            f"Could not connect to local Ollama at {OLLAMA_URL}. "
            "Start Ollama and verify the model is available."
        ) from exc

    if body.get("error"):
        raise RuntimeError(body["error"])

    answer = str(body.get("response", "")).strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--graph-limit", type=int, default=20)
    parser.add_argument("--event-limit", type=int, default=20)
    parser.add_argument("--save", action="store_true")
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

    if not EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing document embeddings: {EMBEDDINGS}")

    metadata = load_metadata()
    embeddings = np.load(EMBEDDINGS)
    encoder = SentenceTransformer(EMBED_MODEL)

    tools = choose_tools(args.query)

    print("=" * 72)
    print("INDUSTRIAL AI AGENT")
    print("=" * 72)
    print(f"Question    : {args.query}")
    print(f"Tools       : {', '.join(tools)}")
    print(f"LLM         : {OLLAMA_MODEL}")
    print()

    documents = []
    graph = {"entities": [], "events": []}
    events = []

    if "document_search" in tools:
        print("Tool: document_search")
        documents = document_search(
            args.query,
            encoder,
            embeddings,
            metadata,
            args.top_k,
        )

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    try:
        driver.verify_connectivity()

        if "graph_lookup" in tools:
            print("Tool: graph_lookup")
            graph = graph_lookup(
                driver,
                args.query,
                args.graph_limit,
            )

        if "event_lookup" in tools:
            print("Tool: event_lookup")
            events = event_lookup(
                driver,
                args.query,
                args.event_limit,
            )
    finally:
        driver.close()

    evidence = build_evidence(
        args.query,
        tools,
        documents,
        graph,
        events,
    )

    prompt = render_prompt(evidence)

    print()
    print(
        f"Evidence: {len(documents)} docs, "
        f"{len(graph['entities'])} entities, "
        f"{len(graph['events']) + len(events)} event records"
    )
    print("Reasoning locally...")
    print()

    answer = ollama_generate(prompt)

    print("=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(answer)

    result = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "query": args.query,
        "tools_used": tools,
        "model": OLLAMA_MODEL,
        "answer": answer,
        "evidence": evidence,
    }

    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        safe = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            args.query.lower(),
        ).strip("_")[:80] or "query"

        output = OUTPUT_DIR / f"agent_{safe}.json"
        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

        print()
        print(f"Saved: {output}")

    print("=" * 72)


if __name__ == "__main__":
    main()
