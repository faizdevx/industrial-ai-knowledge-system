"""
Local web dashboard for Industrial AI Knowledge System Agent V2.

Start:
    uv run uvicorn src.app.app_v2:app --host 127.0.0.1 --port 8000

Open:
    http://127.0.0.1:8000

Features:
  - Agent V2 chat
  - recent sensor values
  - sensor summary
  - candidate events
  - health status
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.runtime.agent_runtime_v2 import (
    EMBED_MODEL,
    EMBEDDINGS,
    OLLAMA_MODEL,
    build_evidence,
    choose_tools,
    document_search,
    event_lookup,
    graph_lookup,
    load_metadata,
    ollama_generate,
    render_prompt,
)
from src.runtime.timeseries_tool import (
    available_sensors,
    sensor_events,
    sensor_summary,
    recent_values,
)


ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
METADATA = DOC_DIR / "chunk_metadata.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "industrial-ai")

app = FastAPI(
    title="Industrial AI Knowledge System V2",
    version="0.2.0",
)

_encoder = None
_embeddings = None
_metadata = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 6


def get_retrieval_assets():
    global _encoder, _embeddings, _metadata

    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        if not EMBEDDINGS.exists():
            raise RuntimeError(f"Missing embeddings: {EMBEDDINGS}")

        if not METADATA.exists():
            raise RuntimeError(f"Missing metadata: {METADATA}")

        _encoder = SentenceTransformer(EMBED_MODEL)
        _embeddings = np.load(EMBEDDINGS)
        _metadata = load_metadata()

    return _encoder, _embeddings, _metadata


def get_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )


@app.get("/health")
def health():
    neo4j_ok = False
    ollama_ok = False

    driver = get_driver()
    try:
        driver.verify_connectivity()
        neo4j_ok = True
    except Exception:
        pass
    finally:
        driver.close()

    try:
        import urllib.request

        base = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate",
        ).replace("/api/generate", "/api/tags")

        with urllib.request.urlopen(base, timeout=3):
            ollama_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if neo4j_ok and ollama_ok else "degraded",
        "neo4j": neo4j_ok,
        "ollama": ollama_ok,
        "embedding_model": EMBED_MODEL,
        "llm": OLLAMA_MODEL,
        "sensor_count": len(available_sensors()),
    }


@app.get("/api/sensors")
def sensors():
    return {
        "count": len(available_sensors()),
        "sensors": available_sensors(),
    }


@app.get("/api/sensor/{sensor_tag}")
def sensor(
    sensor_tag: str,
    rows: int = Query(default=20, ge=1, le=500),
):
    try:
        return {
            "summary": sensor_summary(sensor_tag),
            "recent": recent_values(sensor_tag, rows),
            "events": sensor_events(sensor_tag, min(rows, 50)),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.get("/api/events")
def events(
    limit: int = Query(default=25, ge=1, le=100),
):
    driver = get_driver()
    try:
        driver.verify_connectivity()
        results = event_lookup(driver, "", limit)
        return {
            "count": len(results),
            "events": results,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {exc}",
        ) from exc
    finally:
        driver.close()


@app.post("/api/chat")
def chat(request: ChatRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query is empty.",
        )

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20.",
        )

    encoder, embeddings, metadata = get_retrieval_assets()

    tools = choose_tools(query)
    documents = []
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

    if "document_search" in tools:
        documents = document_search(
            query,
            encoder,
            embeddings,
            metadata,
            request.top_k,
        )

    driver = get_driver()

    try:
        driver.verify_connectivity()

        if "graph_lookup" in tools:
            graph = graph_lookup(
                driver,
                query,
                limit=20,
            )

        if "event_lookup" in tools:
            event_results = event_lookup(
                driver,
                query,
                limit=20,
            )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {exc}",
        ) from exc
    finally:
        driver.close()

    if "timeseries" in tools:
        try:
            from src.runtime.agent_runtime_v2 import run_timeseries_tools

            timeseries = run_timeseries_tools(
                query,
                available_sensors(),
            )
        except Exception as exc:
            timeseries["error"] = str(exc)

    evidence = {
        "schema_version": "2.0",
        "query": query,
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

    prompt = render_prompt(evidence)
    answer = ollama_generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "tools_used": tools,
        "evidence": evidence,
    }


HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industrial AI Knowledge System</title>
<style>
body {
  font-family: Arial, sans-serif;
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}
h1 { margin-bottom: 4px; }
.muted { color: #666; }
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
}
textarea {
  width: 100%;
  min-height: 110px;
  box-sizing: border-box;
  margin-top: 10px;
}
input {
  width: 100%;
  padding: 9px;
  box-sizing: border-box;
  margin-top: 8px;
}
button {
  padding: 9px 14px;
  margin-top: 10px;
  cursor: pointer;
}
pre {
  white-space: pre-wrap;
  word-break: break-word;
  background: #f6f6f6;
  padding: 12px;
  border-radius: 6px;
  max-height: 420px;
  overflow: auto;
}
.status {
  margin: 10px 0;
  padding: 10px;
  background: #f6f6f6;
  border-radius: 6px;
}
@media (max-width: 800px) {
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<h1>Industrial AI Knowledge System</h1>
<div class="muted">Agent V2 · Documents · Neo4j · Time-Series · Ollama</div>

<div class="status" id="health">Checking system health...</div>

<div class="card">
  <h2>Ask the Agent</h2>
  <textarea id="query"
    placeholder="Example: What pressure anomalies were detected?"></textarea>
  <button onclick="askAgent()">Ask</button>
  <pre id="answer">No answer yet.</pre>
</div>

<div class="grid">

<div class="card">
  <h2>Sensor</h2>
  <input id="sensor"
    placeholder="Example: 75ZI800BA.pv">
  <button onclick="loadSensor()">Inspect Sensor</button>
  <pre id="sensorOutput">No sensor selected.</pre>
</div>

<div class="card">
  <h2>Candidate Events</h2>
  <button onclick="loadEvents()">Refresh Events</button>
  <pre id="eventsOutput">No events loaded.</pre>
</div>

</div>

<div class="card">
  <h2>Retrieved Evidence</h2>
  <pre id="evidence">No evidence retrieved.</pre>
</div>

<script>
async function checkHealth() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    document.getElementById("health").textContent =
      "Status: " + d.status +
      " | Neo4j: " + d.neo4j +
      " | Ollama: " + d.ollama +
      " | Sensors: " + d.sensor_count;
  } catch (e) {
    document.getElementById("health").textContent =
      "Health check failed: " + e;
  }
}

async function askAgent() {
  const query = document.getElementById("query").value.trim();
  if (!query) return;

  document.getElementById("answer").textContent =
    "Running local agent...";

  const r = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: query, top_k: 6})
  });

  const d = await r.json();

  if (!r.ok) {
    document.getElementById("answer").textContent =
      d.detail || "Request failed.";
    return;
  }

  document.getElementById("answer").textContent = d.answer;
  document.getElementById("evidence").textContent =
    JSON.stringify({
      tools_used: d.tools_used,
      documents: d.evidence.documents.map(x => ({
        file: x.file,
        page: x.page,
        chunk_id: x.chunk_id,
        score: x.score
      })),
      graph_entities: d.evidence.graph.entities,
      graph_events: d.evidence.graph.events,
      event_results: d.evidence.event_results,
      timeseries: d.evidence.timeseries
    }, null, 2);
}

async function loadSensor() {
  const sensor = document.getElementById("sensor").value.trim();
  if (!sensor) return;

  const r = await fetch(
    "/api/sensor/" + encodeURIComponent(sensor) + "?rows=20"
  );
  const d = await r.json();

  if (!r.ok) {
    document.getElementById("sensorOutput").textContent =
      d.detail || "Sensor lookup failed.";
    return;
  }

  document.getElementById("sensorOutput").textContent =
    JSON.stringify(d, null, 2);
}

async function loadEvents() {
  const r = await fetch("/api/events?limit=25");
  const d = await r.json();

  if (!r.ok) {
    document.getElementById("eventsOutput").textContent =
      d.detail || "Event lookup failed.";
    return;
  }

  document.getElementById("eventsOutput").textContent =
    JSON.stringify(d, null, 2);
}

checkHealth();
loadEvents();
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HTML)
