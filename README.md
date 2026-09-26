# industrial-ai-knowledge-system

```
████  █████ █████ ███ █   █ █████ ████  █   █       ███  ███ 
█   █ █     █      █  ██  █ █     █   █  █ █       █   █  █  
████  ████  ████   █  █ █ █ ████  ████    █   ████ █████  █  
█  █  █     █      █  █  ██ █     █  █    █        █   █  █  
█   █ █████ █     ███ █   █ █████ █   █   █        █   █ ███

```
<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Datasets-FFD21E?logo=huggingface&logoColor=black)
![Qwen](https://img.shields.io/badge/Qwen-OpenWeight%20LLM-7C3AED)
![QLoRA](https://img.shields.io/badge/QLoRA-FineTuning-8E44AD)
![LoRA](https://img.shields.io/badge/LoRA-ParameterEfficientFT-8E44AD)
![Docling](https://img.shields.io/badge/Docling-Document%20Parsing-0F766E)
![NVIDIA NeMo](https://img.shields.io/badge/NVIDIA%20NeMo-AI%20Framework-76B900?logo=nvidia&logoColor=white)
![NeMo Curator](https://img.shields.io/badge/NeMo%20Curator-Data%20Curation-76B900?logo=nvidia&logoColor=white)
![NeMo Retriever](https://img.shields.io/badge/NeMo%20Retriever-RAG%20%26%20Retrieval-76B900?logo=nvidia&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-Data%20Processing-CD792C)
![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-4581C3?logo=neo4j&logoColor=white)
![Graphiti](https://img.shields.io/badge/Graphiti-Temporal%20Knowledge%20Graph-FF6B35)
![Parquet](https://img.shields.io/badge/Parquet-Data%20Storage-50ABF1)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-On--Premise%20Deployment-2496ED?logo=docker&logoColor=white)

</p>

An on-premise AI system for turning messy industrial data, documents, sensor data, and maintenance history into structured, validated, evidence-linked knowledge.

The initial implementation focuses on **petroleum refinery / process-industry use cases**, with the goal of building an architecture that can later be adapted to other industrial domains.

## Problem

Industrial knowledge is usually scattered across:

* Technical documents and manuals
* PDF reports
* CSV / Excel files
* Sensor and time-series data
* Alarms and events
* Maintenance records
* P&IDs, diagrams, and images

This makes it difficult to connect information across sources and answer engineering questions using all available evidence.

## Goal

Build a local AI system that can:

```text
                USER / ORGANIZATION
                       │
                       ▼
              ┌───────────────────┐
              │  INDUSTRIAL DATA  │
              │ PDF CSV XLSX P&ID │
              │ Images / Manuals  │
              │ Sensor Time Series│
              └─────────┬─────────┘
                        │
                        ▼
             PHASE 1: INGESTION
                        │
                        ▼
             PHASE 2: UNDERSTANDING
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
    DOCUMENT PIPELINE          DATA PIPELINE
    Docling                    Pandas/Polars
    OCR                        profiling
    tables                     units
    layouts                    timestamps
    diagrams                   statistics
           │                         │
           └────────────┬────────────┘
                        ▼
             PHASE 3: DOMAIN DISCOVERY
                        │
                        ▼
            "What kind of system is this?"
                        │
                        ▼
             PHASE 4: KNOWLEDGE BUILDING
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         ENTITY DISCOVERY   RELATIONSHIP DISCOVERY
              │                   │
              └─────────┬─────────┘
                        ▼
                ONTOLOGY + KG
                        │
                        ▼
            PHASE 5: TIME-SERIES
                  UNDERSTANDING
                        │
                        ▼
              EVENTS / PATTERNS
                        │
                        ▼
             PHASE 6: VALIDATION
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
         source      graph       temporal
         checks      checks      checks
                        │
                        ▼
             PHASE 7: KNOWLEDGE
                   PACKAGE
                        │
           ┌────────────┴─────────────┐
           ▼                          ▼
    retrieval knowledge         training knowledge
           │                          │
           ▼                          ▼
       PHASE 8                 PHASE 9
     DATA / RAG              TRAINING DATA
                                  │
                                  ▼
                          NeMo Curator
                          Data Designer
                                  │
                                  ▼
                             PHASE 10
                           LoRA / SFT
                                  │
                                  ▼
                         DOMAIN-ADAPTED MODEL
                                  │
             ┌────────────────────┴──────────────────┐
             ▼                                       ▼
      KNOWLEDGE / RETRIEVAL                     MODEL
             │                                       │
             └────────────────┬──────────────────────┘
                              ▼
                         PHASE 11
                       AGENT RUNTIME
                              │
                              ▼
          KG + Documents + Sensors + Tools + Model
                              │
                              ▼
                       INDUSTRIAL AI APP
                              │
                              ▼
                         PHASE 12
                 EVALUATION + DEPLOYMENT
                              │
                              ▼
                    NEW INDUSTRIAL DOMAIN
                              │
                              ▼
                     SAME PIPELINE AGAIN
```

## Initial Understanding

**Domain:** Petroleum Refinery / Process Industry

**Input:**

* PDF documents
* CSV / XLSX
* Sensor / time-series data
* Industrial images and diagrams
* Alarms and events
* Maintenance records

**Reasoning tasks:**

* Equipment relationships
* Sensor relationships
* Temporal / event reasoning
* Operating-condition reasoning
* Document retrieval
* Maintenance and historical comparison

## Example Questions

The system should eventually be able to answer questions such as:

* What is happening to compressor C-101?
* Why is bearing temperature increasing?
* Which sensors are related to the lubrication system?
* What happened before the vibration increased?
* Is the current condition outside the documented operating range?
* Which document defines the operating limit?
* Have similar events happened previously?

## Core Requirements

* **On-premise:** confidential industrial data stays local
* **Evidence-grounded:** answers should be supported by source data
* **Structured:** important industrial entities and relationships should be represented explicitly
* **Traceable:** extracted knowledge should retain its source where possible
* **Evaluated:** extraction, retrieval, reasoning, and answer quality should be measurable

## Project Status

🚧 **Early development**

Current focus:

1.Ingestion Layer
2.Domain Specified(currently refinery based only)
3.ontology behind whole knowledge extracted
4.Discover relationships
5.Attach provenance
6.the time-series intelligence
7.Knowledge Graphs
8.Produce the Knowledge Package
9.Generate the training dataset
10.synthetic and hard-negative data
11.Quality control before training(validation)
12.Fine-tune the model
13.Build the runtime intelligence layer
14.adjust the orchestrators around the model and automate whole thing locally
15.Security and on-premise execution
16.Evaluation of the complete system
*17.Test another domain

## Planned Architecture

```text
                    ┌──────────────────┐
                    │ Industrial Data  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    Ingestion     │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    Extraction    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Knowledge Layer  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Retrieval/Search │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Reasoning / AI   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │   AI Interface   │
                    └──────────────────┘
```




## Long-Term Direction

The refinery is the **first pilot**, not the final boundary of the system.

The longer-term objective is to develop a reusable industrial AI architecture that can work across domains such as:

* Refineries
* Chemical plants
* Power plants
* Manufacturing
* Other process industries


# MVP 
Industrial AI Knowledge System

An on-premise industrial AI system for converting messy industrial data, documents, and sensor time-series into structured, validated, evidence-linked knowledge and making that knowledge available to a local AI agent.

The current implementation is a refinery/process-industry vertical slice. The refinery is the first pilot domain, not the final boundary of the architecture.

1. Project Objective

Industrial knowledge is distributed across:

Technical manuals and operating procedures

PDF reports and documents

CSV/XLSX sensor data

Time-series measurements

Events and alarms

Maintenance information

P&IDs, diagrams, and industrial images

The objective is to build a reusable pipeline that converts these heterogeneous sources into:

messy industrial data
        ↓
understanding
        ↓
structured knowledge
        ↓
validation + provenance
        ↓
retrieval
        ↓
reasoning
        ↓
local industrial AI

The broader research direction is:

unknown industrial domain
        ↓
domain discovery
        ↓
ontology induction
        ↓
validated knowledge graph
        ↓
reasoning dataset
        ↓
domain-adapted AI

The current system demonstrates this pipeline first on refinery data.

2. Current MVP Status

The refinery MVP has been built as a connected pipeline covering:

Excel
  ↓
profiling
  ↓
domain discovery
  ↓
ontology discovery
  ↓
entity / relationship extraction
  ↓
provenance validation
  ↓
Neo4j knowledge graph
  ↓
time-series analysis
  ↓
candidate events
  ↓
Neo4j event integration

PDF
  ↓
PyMuPDF page-aware extraction
  ↓
document chunking
  ↓
BGE embeddings
  ↓
semantic retrieval
  ↓
document → entity linking
  ↓
hybrid retrieval

knowledge package
  ↓
grounded evidence
  ↓
local RAG
  ↓
local agent
  ↓
FastAPI web application

training knowledge
  ↓
grounded training examples
  ↓
validation
  ↓
train / validation / test split
  ↓
SFT / LoRA
  ↓
model evaluation

The current MVP is therefore no longer only a preprocessing experiment. It includes a retrieval and runtime path suitable for an end-to-end local prototype.

3. Architecture

The system keeps different kinds of industrial information in different layers.

                          USER
                            │
                            ▼
                     ┌─────────────┐
                     │ Web / API   │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Agent       │
                     │ Runtime     │
                     └──────┬──────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
      Document Search   Knowledge Graph  Time-Series
             │              │              │
             │              │              │
         PyMuPDF          Neo4j       Pandas / NumPy
             │              │              │
             ▼              ▼              ▼
          Chunks         Entities        Values
          Embeddings     Relations       Trends
             │           Events          Anomalies
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                       Evidence Bundle
                            │
                            ▼
                        Local LLM
                            │
                            ▼
                    Answer + Evidence
                    + Uncertainty

Separation of concerns

Knowledge Graph
= semantic meaning, entities, classes, relationships, events

Time-Series Layer
= actual numerical observations and statistical analysis

Document Layer
= manuals, reports, procedures, source passages and provenance

Training Layer
= supervised examples derived from the knowledge package

Runtime Layer
= retrieval + tools + local model + agent orchestration

This separation is fundamental to the architecture.

4. Industrial Data

The repository is structured to allow several industrial domains:

data/
└── raw/
    ├── refinery/
    │   └── RCSD-1YD.xlsx
    ├── aerospace/
    │   └── NASA C-MAPSS/
    ├── manufacturing/
    │   └── UCI AI4I 2020 Predictive Maintenance Data/
    └── gas_turbine/
        └── UCI GAS TURBING/

The current implementation focuses on:

Domain = Petroleum Refinery / Process Industry

The first refinery sensor dataset is a one-year timestamped dataset with 25 numeric measurement columns.

The current refinery document corpus contains four PDF documents.

5. Pipeline

5.1 Excel Profiling

src/ingestion/profile_excel.py

Produces:

data/processed/refinery/
├── workbook_profile.json
├── sensor_profile.csv
└── Sheet1.csv

The profiler records workbook structure, types, missing values, numerical statistics, and timestamp information.

5.2 Domain Discovery

src/ontology/domain_discovery_input.py
src/ontology/domain_discovery.py

Produces:

data/processed/refinery/
├── domain_discovery_input.json
└── domain_hypothesis.json

The domain result is represented as an evidence-backed hypothesis, not an unquestionable truth.

Example:

candidate domain : industrial_process
status           : hypothesis
confidence       : 0.8

5.3 Ontology Discovery

src/ontology/ontology_discovery.py

Produces:

data/processed/refinery/ontology_candidate.json

The current refinery slice identifies candidate sensor classes and relationship types from the available evidence.

5.4 Entity and Relationship Extraction

src/ontology/entity_relationship_extraction.py

Produces:

data/processed/refinery/knowledge_objects.json

The current MVP focuses on evidence-backed entities such as sensors and measurement series.

It intentionally does not assume:

sensor → physical asset assignment
sensor → component assignment
statistical anomaly → confirmed failure
correlation → causality

5.5 Provenance Validation

src/validation/provenance_validation.py

Produces:

data/processed/refinery/
├── validated_knowledge.json
└── validation_report.json

The provenance layer records where extracted knowledge came from and provides the traceability needed for later retrieval, evaluation, and auditing.

6. Knowledge Graph

Neo4j is used as the semantic graph layer.

Start the graph:

docker compose up -d

Load validated knowledge:

uv run python src/graph/neo4j_loader.py

The graph contains semantic entities and classes with relationships such as:

(Entity)-[:INSTANCE_OF]->(Class)
(Entity)-[:HAS_MEASUREMENT]->(Entity)
(Event)-[:DETECTED_BY]->(Entity)
(DocumentChunk)-[:MENTIONS]->(Entity)

The graph is not used as a storage dump for every sensor row.

7. Time-Series Intelligence

src/timeseries/timeseries_analysis.py

Produces:

data/processed/refinery/timeseries/
├── timeseries_profile.json
├── sensor_timeseries_summary.csv
├── event_candidates.json
└── correlated_sensor_pairs.csv

The current analysis detects statistical patterns including:

spikes
trends
change-point candidates
correlated sensor pairs

The existing refinery run produced approximately:

640 candidate statistical events
94 highly correlated sensor pairs

These are candidate statistical events, not confirmed equipment faults.

8. Event Integration

src/graph/event_graph_integration.py

Links statistical events to their source sensors:

(Event)-[:DETECTED_BY]->(Sensor)

Event provenance and candidate status are preserved.

9. Document Pipeline

The current MVP uses PyMuPDF for document extraction.

Docling is not part of the current production path.

PDF
 ↓
PyMuPDF
 ↓
page-aware extraction
 ↓
document_pages.jsonl
 ↓
chunking
 ↓
document_chunks.jsonl
 ↓
embeddings
 ↓
vector retrieval

The current refinery document corpus contains:

PDF files : 4
Pages     : 1079
Empty / scanned pages : 133

9.1 Document Chunking

src/retrieval/document_chunking.py

The known-good run produced:

Chunks : 1529

Output:

data/processed/refinery/documents/document_chunks.jsonl

9.2 Embeddings

src/retrieval/document_embedding_index.py

Embedding model:

BAAI/bge-small-en-v1.5

Known-good index:

1529 × 384

Outputs:

data/processed/refinery/documents/
├── embeddings.npy
├── chunk_metadata.jsonl
└── embedding_index.json

9.3 Semantic Retrieval

src/retrieval/document_search.py

Example:

uv run python src/retrieval/document_search.py "compressor instrumentation" --top-k 10

The retrieval branch returns source document, page, chunk identifier, score, text, and provenance.

10. Document-to-Entity Linking

src/retrieval/document_entity_linking.py

The first implementation uses conservative exact text matching.

It creates:

(DocumentChunk)-[:MENTIONS]->(Entity)

Output:

data/processed/refinery/documents/document_entity_links.jsonl

Important interpretation:

MENTIONS
≠ physical assignment
≠ component connectivity
≠ causality

This conservative rule prevents the document pipeline from inventing engineering relationships.

11. Hybrid Retrieval

src/retrieval/hybrid_retrieval.py

Hybrid retrieval combines:

document semantic retrieval
        +
Neo4j entity evidence
        +
candidate event evidence

Conceptually:

Question
   ↓
┌──────────────┬───────────────┐
│              │               │
▼              ▼               ▼
Documents    Entities        Events
│              │               │
└──────────────┴───────────────┘
               ↓
         Evidence Context

12. Knowledge Package

src/knowledge/knowledge_package_builder.py

The package converts the independent artifacts into one reproducible refinery knowledge package.

Target structure:

data/knowledge_package/refinery/
├── manifest.json
├── README.md
├── metadata/
├── ontology/
├── graph/
├── documents/
├── timeseries/
├── events/
├── provenance/
└── validation/

The package is intended to become the shared input to both:

training
runtime reasoning

This avoids coupling the model-training pipeline directly to the raw source files.

13. Grounded Reasoning

src/runtime/grounded_qa.py

This stage assembles evidence before an LLM is allowed to reason over it.

Question
   ↓
Document retrieval
   +
Graph evidence
   +
Event evidence
   ↓
Evidence Bundle

The evidence bundle preserves:

document source
page
chunk id
entity id
event id
statistical status
provenance

14. Local RAG

src/runtime/local_rag.py

The RAG layer provides the local model with retrieved evidence instead of asking the model to rely entirely on its parameters.

The reasoning policy requires:

No invented facts
No invented asset assignment
No invented thresholds
No unsupported fault diagnosis
Correlation ≠ causation
Statistical anomaly ≠ confirmed failure

15. Time-Series Tools

src/runtime/timeseries_tool.py

Available tools include:

identify_sensors()
sensor_summary()
recent_values()
sensor_events()
compare_sensors()

These expose numerical evidence to the runtime system without putting the entire time-series into the knowledge graph.

16. Agent Runtime

Current runtime:

src/runtime/agent_runtime_v2.py

The agent can use:

document_search
graph_lookup
event_lookup
identify_sensors
sensor_summary
recent_values
sensor_events
compare_sensors

Runtime:

USER QUESTION
      ↓
TOOL ROUTER
      ↓
┌─────┬───────────┬────────────┐
│     │           │            │
PDF  Neo4j    Time-Series    Events
│     │           │            │
└─────┴───────────┴────────────┘
              ↓
         EVIDENCE BUNDLE
              ↓
           LOCAL LLM
              ↓
       ANSWER + EVIDENCE
       + UNCERTAINTY

The current implementation uses a deterministic first-pass router so that tool selection remains inspectable and reproducible.

17. Local LLM

Ollama is used for local model serving.

Example:

ollama serve

Pull a local model:

ollama pull qwen2.5:7b

The runtime can then use the local Ollama endpoint.

Model serving remains local:

Data
 ↓
Retrieval
 ↓
Neo4j / Time-Series
 ↓
Ollama

No external inference API is required by the runtime path.

18. Training Pipeline

The training path is separated from the knowledge graph itself.

Knowledge Package
      +
Documents
      +
Events
      +
Evidence
      ↓
Training Data Generator
      ↓
Grounded Examples
      ↓
Validation
      ↓
Train / Validation / Test
      ↓
SFT / LoRA

Generate training examples

src/training/training_data_generator.py

Output:

data/processed/refinery/training/
├── training_examples.jsonl
└── training_summary.json

The dataset contains tasks such as:

entity classification
provenance
document evidence
time-series event interpretation
uncertainty reasoning
hard negatives

19. Training Data Validation

src/training/validate_training_data.py

Checks include:

schema
source records
document references
entity references
event references
duplicate IDs
duplicate examples
uncertainty safeguards

Output:

data/processed/refinery/training/
├── training_validation_results.jsonl
└── training_validation_report.json

Passing the validator does not mean the examples are engineering ground truth. Human review remains necessary.

20. Train / Validation / Test Split

src/training/split_training_data.py

Current baseline:

80% train
10% validation
10% test

Outputs:

data/processed/refinery/training/splits/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── split_manifest.json

This is a reproducible baseline split.

It is not the final research protocol for unseen-equipment, future-time, or unseen-domain generalization.

21. SFT Dataset Preparation

src/training/prepare_sft_dataset.py

Output:

data/processed/refinery/training/sft/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── dataset_manifest.json

Training format:

system
   ↓
industrial grounding rules

user
   ↓
question + evidence

assistant
   ↓
grounded response

22. LoRA / SFT

src/training/train_lora.py

The current trainer is designed for a first local LoRA smoke test.

The initial configuration uses a small Qwen model before moving to larger models or QLoRA experiments.

Example:

uv run python src/training/train_lora.py

Adapter output:

data/models/refinery-lora/

Training is deliberately separated from retrieval so that changing industrial facts remain in the knowledge/retrieval layer rather than being forced into model weights.

23. Model Evaluation

src/evaluation/evaluate_sft.py

Evaluates the trained adapter on the held-out test set.

TEST SET
   │
   ├── Base Model
   │
   └── LoRA Model
           ↓
        Compare

Automatic checks include:

non-empty response
grounding language
required structure

Engineering correctness still requires human evaluation.

24. Base vs LoRA Comparison

src/evaluation/compare_base_vs_lora.py

The same test examples are run through:

Base model
     vs
LoRA-adapted model

The purpose is to determine whether fine-tuning actually changes the desired behavior rather than simply producing another model checkpoint.

25. End-to-End Benchmark

src/evaluation/run_system_benchmark.py

The benchmark evaluates the complete system across categories such as:

document retrieval
time-series reasoning
cross-modal reasoning
uncertainty
provenance
temporal reasoning
hard negatives

Outputs:

data/processed/refinery/evaluation/
├── system_benchmark_results.jsonl
└── system_benchmark_summary.json

Automatic metrics measure system mechanics.

Human review is required for:

factual correctness
engineering validity
provenance correctness
temporal reasoning correctness
unsupported claim rate
usefulness

26. Web Application

Current application:

src/app/app_v3.py

Start it with:

uv run uvicorn src.app.app_v3:app --host 127.0.0.1 --port 8000

Open:

http://127.0.0.1:8000

The dashboard exposes:

Agent chat
Sensor selector
Sensor trend chart
Candidate anomaly timeline
Retrieved document evidence
Neo4j evidence
Time-series evidence
System health

The application is a thin presentation layer. Retrieval, the agent, Neo4j, time-series tools, and the local model remain separate components.

27. Final MVP Integration Check

Run:

uv run python src/evaluation/final_system_check.py

The integration check verifies:

core artifacts
document artifacts
time-series artifacts
knowledge package
training artifacts
runtime modules
embedding consistency
Neo4j
Ollama

Expected final state:

STATUS: PASSED

28. Repository Structure

industrial-ai-knowledge-system/
│
├── data/
│   ├── raw/
│   │   ├── refinery/
│   │   ├── aerospace/
│   │   ├── manufacturing/
│   │   └── gas_turbine/
│   │
│   ├── processed/
│   │   └── refinery/
│   │       ├── documents/
│   │       ├── timeseries/
│   │       ├── training/
│   │       ├── evaluation/
│   │       └── runtime/
│   │
│   └── knowledge_package/
│       └── refinery/
│
├── src/
│   ├── ingestion/
│   ├── ontology/
│   ├── validation/
│   ├── graph/
│   ├── timeseries/
│   ├── retrieval/
│   ├── knowledge/
│   ├── training/
│   ├── evaluation/
│   ├── runtime/
│   └── app/
│
├── docker-compose.yml
├── pyproject.toml
└── README.md

29. Quick Start

Install dependencies

This project uses uv.

uv sync

Additional dependencies used by the current runtime/training path may include:

uv add pymupdf
uv add sentence-transformers numpy
uv add fastapi uvicorn
uv add datasets "trl[peft]" accelerate

Start Neo4j

docker compose up -d

Start Ollama

ollama serve

Pull the configured local model:

ollama pull qwen2.5:7b

Build the knowledge pipeline

Run the ingestion, ontology, validation, graph, and time-series stages in dependency order.

Then build the document branch:

uv run python src/ingestion/document_ingestion.py
uv run python src/retrieval/document_chunking.py
uv run python src/retrieval/document_embedding_index.py
uv run python src/retrieval/document_search.py "compressor instrumentation" --top-k 10

Then:

uv run python src/retrieval/document_entity_linking.py
uv run python src/retrieval/hybrid_retrieval.py "compressor instrumentation" --top-k 8

Build the package:

uv run python src/knowledge/knowledge_package_builder.py --overwrite

Run grounded reasoning:

uv run python src/runtime/grounded_qa.py "What pressure anomalies were detected?"

Run the local agent:

uv run python src/runtime/agent_runtime_v2.py "What pressure anomalies were detected?"

Start the web application:

uv run uvicorn src.app.app_v3:app --host 127.0.0.1 --port 8000

30. Evidence and Reasoning Policy

The system follows a strict evidence policy.

Statistical anomaly

statistical anomaly
        ≠
confirmed fault

Document mention

document mentions sensor tag
        ≠
physical asset assignment

Correlation

correlation
        ≠
causation

Insufficient evidence

When the evidence is insufficient, the system should say that it is insufficient rather than inventing a conclusion.

This is a core requirement of the architecture, not merely a prompt instruction.

31. Research Evaluation Direction

The system should eventually be evaluated on more than whether the chatbot produces plausible prose.

Relevant evaluation dimensions include:

Data processing
  file processing success
  extraction quality

Knowledge construction
  ontology precision
  entity precision
  relationship precision
  coverage

Provenance
  percentage of claims with source evidence

Temporal reasoning
  event ordering
  pattern detection

Runtime AI
  factual correctness
  unsupported-claim rate
  provenance correctness
  uncertainty handling

Automation
  human corrections
  manual intervention required

The eventual generalization experiment should separate:

future time
equipment
data source
industrial domain

A stronger research test is:

refinery
   ↓
same pipeline
   ↓
new industrial domain
   ↓
measure required adaptation

The important question is not merely whether a model can answer refinery questions. It is how much manual intervention the knowledge-construction and model-customization pipeline requires when the industrial domain changes.

32. Known MVP Limitations

The current system is intentionally a first vertical slice.

1. Asset mapping is incomplete

Sensor tags are currently treated conservatively. Full physical asset/component mapping requires additional evidence and entity-resolution logic.

2. Statistical events are not engineering diagnoses

The current event detector identifies statistical candidates. It does not establish equipment failure.

3. Document linking is conservative

The first document/entity linker uses exact text matching. Semantic entity resolution is a later stage.

4. Time-series reasoning is still basic

Current tools expose summaries, recent values, events, and simple comparisons. More advanced temporal reasoning is still required.

5. Training data is a prototype

The generated examples are grounded candidates. They require human review before being treated as high-quality training data.

6. Cross-domain generalization is not yet demonstrated

The architecture supports multiple domains, but the refinery vertical slice is the current primary implementation.

33. Long-Term Direction

The refinery is the first pilot.

The longer-term system should support domains such as:

Refinery
Chemical plant
Power plant
Manufacturing
Other process industries

The target architecture is:

ANY INDUSTRIAL DOMAIN
        ↓
DOMAIN DISCOVERY
        ↓
ONTOLOGY INDUCTION
        ↓
MULTIMODAL KNOWLEDGE CONSTRUCTION
        ↓
VALIDATION + PROVENANCE
        ↓
KNOWLEDGE PACKAGE
        ↓
TRAINING DATA
        ↓
DOMAIN ADAPTATION
        ↓
RUNTIME AGENT
        ↓
EVALUATION
        ↓
NEW DOMAIN

The research boundary should remain measurable and defensible:

The system should be evaluated by how accurately it constructs and uses industrial knowledge, how well it preserves provenance and uncertainty, and how much manual intervention is required when the same pipeline is applied to a new industrial domain.

34. Current Development Philosophy

The project deliberately follows:

build the data pipeline
        ↓
validate the knowledge
        ↓
connect retrieval
        ↓
connect reasoning
        ↓
evaluate
        ↓
then optimize the model

The model is only one component.

The central system is:

data
  +
knowledge
  +
provenance
  +
time-series
  +
retrieval
  +
tools
  +
model
  +
evaluation

That is the Industrial AI Knowledge System.