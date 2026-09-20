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


