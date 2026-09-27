"""
Agent runtime v2 with explicit time-series tools.

This replaces the earlier deterministic router with a richer tool set:

  document_search
  graph_lookup
  event_lookup
  identify_sensors
  sensor_summary
  recent_values
  sensor_events
  compare_sensors

The router remains deterministic and auditable for the MVP. Later, the local
LLM can be allowed to select tools through structured tool calling.

Usage:
  uv run python src/runtime/agent_runtime_v2.py "What pressure anomalies were detected?"
  uv run python src/runtime/agent_runtime_v2.py "Show recent values for 75PI823.pv"
  uv run python src/runtime/agent_runtime_v2.py "Compare 75PI823.pv and 75TI..."
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

from agent_runtime import (
    EMBED_MODEL,
    EMBEDDINGS,
    METADATA,
    document_search,
    event_lookup,
    graph_lookup,
    load_metadata,
)
from timeseries_tool import (
    compare_sensors,
    identify_sensors,
    recent_values,
    sensor_events,
    sensor_summary,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed" / "refinery" / "runtime"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

SYSTEM_PROMPT = """
You are the reasoning component of a local industrial AI agent.

Use only the supplied tool evidence.

Rules:
- Do not invent equipment, sensor assignments, thresholds, causes, failures,
  or maintenance actions.
- Statistical events are candidate anomalies, not confirmed faults.
- Document mention is not proof of physical asset assignment.
- Correlation is not causation.
- Distinguish documented facts, numerical observations, statistical events,
  and hypotheses.
- When evidence is insufficient, say so.
- Cite exact [DOC:...], [ENTITY:...], [EVENT:...], or [TS:...] identifiers
  supplied by the tools.
- Do not claim a causal explanation unless the evidence explicitly supports it.

Return:
Answer:
Evidence:
Assessment:
Uncertainty:
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_tools(query: str) -> list[str]:
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
        "change-point",
        "failure",
        "fault",
        "happened",
        "before",
        "after",
        "detected",
    )

    timeseries_terms = (
        "recent",
        "latest",
        "current value",
        "current reading",
        "trend",
        "over time",
        "time series",
        "timeseries",
        "value",
        "values",
        "measurement",
        "measurements",
        "compare",
        "correlation",
    )

    if any(term in q for term in graph_terms):
        tools.append("graph_lookup")

    if any(term in q for term in event_terms):
        tools.append("event_lookup")

    if any(term in q for term in timeseries_terms):
        tools.append("timeseries")

    # Event/failure questions generally benefit from time-series evidence too.
    if any(term in q for term in event_terms):
        if "timeseries" not in tools:
            tools.append("timeseries")

    return tools


def extract_explicit_sensors(
    query: str,
    all_sensor_names: list[str],
) -> list[str]:
    q = query.lower()

    exact = [
        name for name in all_sensor_names
        if name.lower() in q
    ]

    return exact


def run_timeseries_tools(
    query: str,
    sensor_names: list[str],
) -> dict[str, Any]:
    matched = extract_explicit_sensors(
        query,
        sensor_names,
    )

    if not matched:
        matched = identify_sensors(
            query,
            limit=4,
        )

    output: dict[str, Any] = {
        "identified_sensors": matched,
        "summaries": [],
        "recent_values": [],
        "events": [],
        "comparisons": [],
    }

    # Exact "compare A and B" requests.
    if len(matched) >= 2 and (
        "compare" in query.lower()
        or "correlation" in query.lower()
    ):
        try:
            output["comparisons"].append(
                compare_sensors(
                    matched[0],
                    matched[1],
                    rows=500,
                )
            )
        except Exception as exc:
            output["timeseries_error"] = str(exc)

    for sensor in matched[:3]:
        try:
            output["summaries"].append(
                sensor_summary(sensor)
            )
        except Exception as exc:
            output.setdefault("tool_errors", []).append(
                f"summary {sensor}: {exc}"
            )

        try:
            output["recent_values"].append(
                recent_values(
                    sensor,
                    rows=12,
                )
            )
        except Exception as exc:
            output.setdefault("tool_errors", []).append(
                f"recent {sensor}: {exc}"
            )

        try:
            output["events"].append({
                "sensor": sensor,
                "items": sensor_events(
                    sensor,
                    limit=12,
                ),
            })
        except Exception as exc:
            output.setdefault("tool_errors", []).append(
                f"events {sensor}: {exc}"
            )

    return output


def render_prompt(evidence: dict[str, Any]) -> str:
    parts = [
        SYSTEM_PROMPT,
        "",
        "QUESTION",
        evidence["query"],
        "",
        "TOOLS USED",
        ", ".join(evidence["tools_used"]),
        "",
        "DOCUMENT EVIDENCE",
    ]

    for item in evidence["documents"]:
        parts.extend([
            (
                f"[DOC:{item['file']} p.{item['page']} "
                f"chunk:{item['chunk_id']}] score={item['score']}"
            ),
            item["text"],
            "",
        ])

    parts.append("GRAPH ENTITIES")

    for item in evidence["graph"]["entities"]:
        parts.append(
            f"[ENTITY:{item['entity_id']}] "
            f"column={item['source_column']} "
            f"class={item['candidate_class']} "
            f"family={item['tag_family']}"
        )

    parts.append("")
    parts.append("GRAPH EVENTS")

    for item in evidence["graph"]["events"]:
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

    parts.append("")
    parts.append("EVENT SEARCH RESULTS")

    for item in evidence["event_results"]:
        parts.append(
            f"[EVENT:{item.get('event_id')}] "
            f"sensor={item.get('entity_id')} "
            f"type={item.get('event_type')} "
            f"status={item.get('status')} "
            f"start={item.get('start')} "
            f"end={item.get('end')} "
            f"peak={item.get('peak_value')} "
            f"robust_z={item.get('peak_robust_z')}"
        )

    ts = evidence["timeseries"]

    parts.append("")
    parts.append("TIME-SERIES TOOL RESULTS")

    parts.append(
        "IDENTIFIED SENSORS: "
        + ", ".join(ts.get("identified_sensors", []))
    )

    for item in ts.get("summaries", []):
        sensor = item["sensor"]
        parts.append(
            f"[TS:SUMMARY:{sensor}] "
            f"rows={item['rows']} "
            f"min={item['min']} "
            f"max={item['max']} "
            f"mean={item['mean']} "
            f"median={item['median']} "
            f"std={item['std']} "
            f"range={item['first_timestamp']}..{item['last_timestamp']}"
        )

    for series in ts.get("recent_values", []):
        sensor = series["sensor"]
        parts.append(f"[TS:RECENT:{sensor}]")
        for row in series["rows"]:
            parts.append(
                f"timestamp={row['timestamp']} value={row['value']}"
            )

    for series in ts.get("events", []):
        sensor = series["sensor"]
        for event in series["items"]:
            event_id = (
                event.get("id")
                or event.get("event_id")
                or "unknown"
            )
            parts.append(
                f"[TS:EVENT:{event_id}] "
                f"sensor={sensor} "
                f"type={event.get('event_type')} "
                f"status={event.get('status')} "
                f"start={event.get('start')} "
                f"end={event.get('end')} "
                f"peak={event.get('peak_value')} "
                f"robust_z={event.get('peak_robust_z')}"
            )

    for comparison in ts.get("comparisons", []):
        parts.append(
            f"[TS:COMPARE:{comparison['sensor_a']}:{comparison['sensor_b']}] "
            f"rows={comparison['rows_analyzed']} "
            f"pearson={comparison['pearson_correlation']} "
            f"range={comparison['start']}..{comparison['end']}"
        )

    parts.append("")
    parts.append("Answer strictly from this evidence.")
    return "\n".join(parts)


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
        with urllib.request.urlopen(
            request,
            timeout=300,
        ) as response:
            body = json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}."
        ) from exc

    if body.get("error"):
        raise RuntimeError(body["error"])

    response = str(
        body.get("response", "")
    ).strip()

    if not response:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--uri",
        default=os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        ),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--password",
        default=os.getenv(
            "NEO4J_PASSWORD",
            "industrial-ai",
        ),
    )
    args = parser.parse_args()

    if not EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {EMBEDDINGS}"
        )

    metadata = load_metadata()
    embeddings = np.load(EMBEDDINGS)
    encoder = SentenceTransformer(EMBED_MODEL)

    tools = choose_tools(args.query)

    print("=" * 72)
    print("INDUSTRIAL AI AGENT V2")
    print("=" * 72)
    print(f"Question : {args.query}")
    print(f"Tools    : {', '.join(tools)}")
    print(f"LLM      : {OLLAMA_MODEL}")
    print()

    documents = []
    graph = {
        "entities": [],
        "events": [],
    }
    event_results = []
    ts_results = {
        "identified_sensors": [],
        "summaries": [],
        "recent_values": [],
        "events": [],
        "comparisons": [],
    }

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
        auth=(
            args.user,
            args.password,
        ),
    )

    try:
        driver.verify_connectivity()

        if "graph_lookup" in tools:
            print("Tool: graph_lookup")
            graph = graph_lookup(
                driver,
                args.query,
                limit=20,
            )

        if "event_lookup" in tools:
            print("Tool: event_lookup")
            event_results = event_lookup(
                driver,
                args.query,
                limit=20,
            )
    finally:
        driver.close()

    if "timeseries" in tools:
        print("Tool: timeseries")
        try:
            from timeseries_tool import available_sensors
            all_sensors = available_sensors()
            ts_results = run_timeseries_tools(
                args.query,
                all_sensors,
            )
        except Exception as exc:
            ts_results = {
                "identified_sensors": [],
                "summaries": [],
                "recent_values": [],
                "events": [],
                "comparisons": [],
                "error": str(exc),
            }

    evidence = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "query": args.query,
        "tools_used": tools,
        "documents": documents,
        "graph": graph,
        "event_results": event_results,
        "timeseries": ts_results,
        "policies": {
            "statistical_event_is_not_confirmed_fault": True,
            "document_mention_is_not_physical_assignment": True,
            "correlation_is_not_causation": True,
            "insufficient_evidence_must_be_stated": True,
        },
    }

    print()
    print(
        f"Evidence: {len(documents)} docs, "
        f"{len(graph['entities'])} entities, "
        f"{len(graph['events']) + len(event_results)} event records"
    )

    prompt = render_prompt(evidence)

    print("Reasoning locally...")
    print()

    answer = ollama_generate(prompt)

    print("=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(answer)

    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            args.query.lower(),
        ).strip("_")[:80] or "query"

        path = OUTPUT_DIR / f"agent_v2_{safe}.json"
        path.write_text(
            json.dumps(
                {
                    "evidence": evidence,
                    "answer": answer,
                    "model": OLLAMA_MODEL,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

        print(f"\nSaved: {path}")

    print("=" * 72)


if __name__ == "__main__":
    main()
