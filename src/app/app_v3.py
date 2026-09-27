"""
Industrial AI Knowledge System V3 dashboard.

Adds:
  - Agent V2 chat
  - sensor summary
  - recent sensor values
  - sensor trend chart data
  - candidate anomaly timeline
  - health status

Start:
  uv run uvicorn src.app.app_v3:app --host 127.0.0.1 --port 8000

Open:
  http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.runtime.agent_runtime_v2 import (
    EMBED_MODEL,
    EMBEDDINGS,
    OLLAMA_MODEL,
    document_search,
    event_lookup,
    graph_lookup,
    load_metadata,
    ollama_generate,
    render_prompt,
    choose_tools,
    run_timeseries_tools,
)
from src.runtime.timeseries_tool import (
    available_sensors,
    sensor_events,
    sensor_summary,
    recent_values,
    canonical_sensor,
    load_sensor_data,
)


ROOT = Path(__file__).resolve().parents[2]

DOC_DIR = ROOT / "data" / "processed" / "refinery" / "documents"
METADATA = DOC_DIR / "chunk_metadata.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "industrial-ai")

app = FastAPI(
    title="Industrial AI Knowledge System V3",
    version="0.3.0",
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

        url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate",
        ).replace("/api/generate", "/api/tags")

        with urllib.request.urlopen(url, timeout=3):
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


@app.get("/api/sensor/{sensor_tag}/trend")
def sensor_trend(
    sensor_tag: str,
    rows: int = Query(default=500, ge=10, le=5000),
):
    try:
        df = load_sensor_data()
        sensor_tag = canonical_sensor(df, sensor_tag)

        frame = df[["Timestamp", sensor_tag]].tail(rows).copy()
        frame[sensor_tag] = pd.to_numeric(
            frame[sensor_tag],
            errors="coerce",
        )
        frame = frame.dropna(subset=[sensor_tag])

        return {
            "sensor": sensor_tag,
            "count": len(frame),
            "points": [
                {
                    "timestamp": row["Timestamp"].isoformat(),
                    "value": float(row[sensor_tag]),
                }
                for _, row in frame.iterrows()
            ],
            "source": "Sheet1.csv",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.get("/api/events")
def events(
    limit: int = Query(default=50, ge=1, le=500),
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


@app.get("/api/events/timeline")
def events_timeline(
    limit: int = Query(default=150, ge=1, le=500),
):
    driver = get_driver()

    try:
        driver.verify_connectivity()

        query = """
        MATCH (ev:Event)-[:DETECTED_BY]->(e:Entity)
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
          e.id AS sensor
        ORDER BY ev.start DESC
        LIMIT $limit
        """

        with driver.session() as session:
            rows = list(session.run(query, limit=limit))

        events = [
            {
                "event_id": row["event_id"],
                "sensor": row["sensor"],
                "event_type": row["event_type"],
                "status": row["status"],
                "start": row["start"],
                "end": row["end"],
                "peak_value": row["peak_value"],
                "peak_robust_z": row["peak_robust_z"],
                "direction": row["direction"],
                "interpretation": row["interpretation"],
            }
            for row in rows
        ]

        return {
            "count": len(events),
            "events": events,
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

    answer = ollama_generate(
        render_prompt(evidence)
    )

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
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {
  font-family: Arial, sans-serif;
  max-width: 1250px;
  margin: 0 auto;
  padding: 24px;
}
h1 { margin-bottom: 4px; }
.muted { color: #666; }
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}
.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin-top: 18px;
}
textarea, input, select {
  width: 100%;
  box-sizing: border-box;
  padding: 9px;
  margin-top: 8px;
}
textarea { min-height: 100px; }
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
  max-height: 380px;
  overflow: auto;
}
canvas {
  width: 100% !important;
  height: 340px !important;
}
.status {
  margin-top: 12px;
  padding: 10px;
  background: #f6f6f6;
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  padding: 7px;
  border-bottom: 1px solid #eee;
  text-align: left;
  font-size: 13px;
}
.small {
  font-size: 12px;
  color: #666;
}
@media (max-width: 850px) {
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<h1>Industrial AI Knowledge System</h1>
<div class="muted">
  Agent V2 · Documents · Neo4j · Time-Series · Ollama
</div>

<div class="status" id="health">Checking system...</div>

<div class="card">
  <h2>Ask the Agent</h2>
  <textarea id="query"
    placeholder="Example: What pressure anomalies were detected?"></textarea>
  <button onclick="askAgent()">Ask</button>
  <pre id="answer">No answer yet.</pre>
</div>

<div class="grid">

<div class="card">
  <h2>Sensor Trend</h2>
  <select id="sensorSelect"></select>
  <button onclick="loadTrend()">Load Trend</button>
  <canvas id="trendChart"></canvas>
  <pre id="sensorInfo">Select a sensor.</pre>
</div>

<div class="card">
  <h2>Candidate Anomaly Timeline</h2>
  <button onclick="loadTimeline()">Refresh Timeline</button>
  <div style="overflow:auto; margin-top:12px;">
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Sensor</th>
          <th>Type</th>
          <th>Status</th>
          <th>Robust Z</th>
        </tr>
      </thead>
      <tbody id="timelineBody"></tbody>
    </table>
  </div>
</div>

</div>

<div class="card">
  <h2>Retrieved Evidence</h2>
  <pre id="evidence">No evidence retrieved.</pre>
</div>

<script>
let trendChart = null;

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

async function loadSensors() {
  const r = await fetch("/api/sensors");
  const d = await r.json();

  const select = document.getElementById("sensorSelect");
  select.innerHTML = "";

  d.sensors.forEach(sensor => {
    const option = document.createElement("option");
    option.value = sensor;
    option.textContent = sensor;
    select.appendChild(option);
  });
}

async function loadTrend() {
  const sensor = document.getElementById("sensorSelect").value;
  if (!sensor) return;

  const r = await fetch(
    "/api/sensor/" + encodeURIComponent(sensor) + "/trend?rows=500"
  );
  const d = await r.json();

  if (!r.ok) {
    document.getElementById("sensorInfo").textContent =
      d.detail || "Trend lookup failed.";
    return;
  }

  const labels = d.points.map(p => p.timestamp);
  const values = d.points.map(p => p.value);

  if (trendChart) {
    trendChart.destroy();
  }

  trendChart = new Chart(
    document.getElementById("trendChart"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: d.sensor,
          data: values,
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 10
            }
          }
        }
      }
    }
  );

  const summary = await fetch(
    "/api/sensor/" + encodeURIComponent(sensor) + "?rows=12"
  );
  const summaryData = await summary.json();

  document.getElementById("sensorInfo").textContent =
    JSON.stringify(summaryData, null, 2);
}

async function loadTimeline() {
  const r = await fetch("/api/events/timeline?limit=100");
  const d = await r.json();

  const body = document.getElementById("timelineBody");
  body.innerHTML = "";

  if (!r.ok) {
    body.innerHTML =
      "<tr><td colspan='5'>" +
      (d.detail || "Timeline failed.") +
      "</td></tr>";
    return;
  }

  d.events.forEach(event => {
    const row = document.createElement("tr");

    row.innerHTML =
      "<td>" + (event.start || "") + "</td>" +
      "<td>" + (event.sensor || "") + "</td>" +
      "<td>" + (event.event_type || "") + "</td>" +
      "<td>" + (event.status || "") + "</td>" +
      "<td>" + (event.peak_robust_z ?? "") + "</td>";

    body.appendChild(row);
  });
}

async function askAgent() {
  const query = document.getElementById("query").value.trim();

  if (!query) return;

  document.getElementById("answer").textContent =
    "Running local agent...";

  const r = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      query: query,
      top_k: 6
    })
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

checkHealth();
loadSensors().then(loadTrend);
loadTimeline();
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HTML)
