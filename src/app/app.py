"""
Local web application for the Industrial AI Knowledge System MVP.

Start:
    uv run uvicorn src.app.app:app --host 127.0.0.1 --port 8000

Open:
    http://127.0.0.1:8000

Endpoints:
    GET  /health
    POST /api/chat
    GET  /api/search?q=compressor%20instrumentation
    GET  /api/events

This layer is intentionally thin. The agent, retrieval, Neo4j and Ollama
remain separate components.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from src.runtime.agent_runtime import (
    EMBED_MODEL,
    OLLAMA_MODEL,
    EMBEDDINGS,
    document_search,
    graph_lookup,
    event_lookup,
    load_metadata,
    ollama_generate,
    build_evidence,
    render_prompt,
    choose_tools,
)


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data" / "processed" / "refinery" / "documents" / "chunk_metadata.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "industrial-ai")

app = FastAPI(
    title="Industrial AI Knowledge System",
    version="0.1.0",
)

_encoder: SentenceTransformer | None = None
_embeddings: np.ndarray | None = None
_metadata: list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 6


def get_retrieval_assets():
    global _encoder, _embeddings, _metadata

    if _encoder is None:
        if not EMBEDDINGS.exists():
            raise RuntimeError(f"Missing embeddings: {EMBEDDINGS}")
        if not METADATA.exists():
            raise RuntimeError(f"Missing metadata: {METADATA}")

        _encoder = SentenceTransformer(EMBED_MODEL)
        _embeddings = np.load(EMBEDDINGS)
        _metadata = load_metadata()

    return _encoder, _embeddings, _metadata


def get_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )


@app.get("/health")
def health():
    neo4j_ok = False

    driver = get_driver()
    try:
        driver.verify_connectivity()
        neo4j_ok = True
    except Exception:
        neo4j_ok = False
    finally:
        driver.close()

    ollama_ok = False
    try:
        import urllib.request
        with urllib.request.urlopen(
            os.getenv(
                "OLLAMA_URL",
                "http://localhost:11434/api/generate",
            ).replace("/api/generate", "/api/tags"),
            timeout=3,
        ):
            ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if neo4j_ok and ollama_ok else "degraded",
        "neo4j": neo4j_ok,
        "ollama": ollama_ok,
        "embedding_model": EMBED_MODEL,
        "llm": OLLAMA_MODEL,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is empty.")

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20.",
        )

    encoder, embeddings, metadata = get_retrieval_assets()

    tools = choose_tools(query)
    documents = []
    graph = {"entities": [], "events": []}
    events = []

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
            events = event_lookup(
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

    evidence = build_evidence(
        query,
        tools,
        documents,
        graph,
        events,
    )

    prompt = render_prompt(evidence)
    answer = ollama_generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "tools_used": tools,
        "evidence": {
            "documents": documents,
            "entities": graph["entities"],
            "graph_events": graph["events"],
            "event_results": events,
        },
    }


@app.get("/api/search")
def search(
    q: str = Query(min_length=1),
    top_k: int = Query(default=8, ge=1, le=20),
):
    encoder, embeddings, metadata = get_retrieval_assets()

    results = document_search(
        q,
        encoder,
        embeddings,
        metadata,
        top_k,
    )

    return {
        "query": q,
        "results": results,
    }


@app.get("/api/events")
def events(limit: int = Query(default=20, ge=1, le=100)):
    driver = get_driver()

    try:
        driver.verify_connectivity()
        results = event_lookup(
            driver,
            "",
            limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {exc}",
        ) from exc
    finally:
        driver.close()

    return {
        "count": len(results),
        "events": results,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Industrial AI Knowledge System</title>
<style>
body {
  font-family: Arial, sans-serif;
  max-width: 1100px;
  margin: 40px auto;
  padding: 0 20px;
}
h1 { margin-bottom: 4px; }
.muted { color: #666; }
textarea {
  width: 100%;
  min-height: 110px;
  padding: 10px;
  box-sizing: border-box;
  margin-top: 12px;
}
button {
  padding: 10px 16px;
  margin-top: 10px;
  cursor: pointer;
}
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  background: #f5f5f5;
  padding: 14px;
  border-radius: 6px;
}
.panel {
  margin-top: 20px;
}
</style>
</head>
<body>
<h1>Industrial AI Knowledge System</h1>
<div class="muted">Local retrieval + Neo4j + Ollama</div>

<div class="panel">
<textarea id="query"
placeholder="Ask: What pressure anomalies were detected?"></textarea>
<br>
<button onclick="ask()">Ask</button>
</div>

<div class="panel">
<h2>Answer</h2>
<pre id="answer">Waiting for a question.</pre>
</div>

<div class="panel">
<h2>Evidence</h2>
<pre id="evidence">No evidence retrieved yet.</pre>
</div>

<script>
async function ask() {
  const query = document.getElementById("query").value.trim();
  if (!query) return;

  document.getElementById("answer").textContent = "Running local agent...";
  document.getElementById("evidence").textContent = "";

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: query, top_k: 6})
  });

  const data = await response.json();

  if (!response.ok) {
    document.getElementById("answer").textContent =
      data.detail || "Request failed.";
    return;
  }

  document.getElementById("answer").textContent = data.answer;
  document.getElementById("evidence").textContent =
    JSON.stringify({
      tools_used: data.tools_used,
      documents: data.evidence.documents.map(x => ({
        file: x.file,
        page: x.page,
        chunk_id: x.chunk_id,
        score: x.score
      })),
      entities: data.evidence.entities,
      graph_events: data.evidence.graph_events,
      event_results: data.evidence.event_results
    }, null, 2);
}
</script>
</body>
</html>
"""
    )
