# LLM Quality Gate

## Overview

**LLM Quality Gate** is a production-grade evaluation and regression-testing system for Large Language Model (LLM) applications. It ensures that prompt, model, or configuration changes do not silently degrade output quality before reaching production.

The system is **fully provider-agnostic**. While **Groq** is the default provider (due to its fast, free hosted open‑source models), users can choose **any supported LLM API** — open-source or proprietary — via configuration, without changing evaluation logic.

Supported provider categories include:

* Free hosted open-source APIs (Groq, Hugging Face, OpenRouter)
* Proprietary APIs (OpenAI, Anthropic Claude, Google Gemini)
* Optional local providers (Ollama, LocalAI)

> Think of this as **unit tests + CI/CD for prompts and LLM behavior**, with swappable model backends.

---

## The Problem

Most teams shipping LLM-powered features:

* Change prompts or models without regression testing
* Rely on manual spot checks
* Discover failures only after users complain

A single change can:

* Reduce task success
* Increase hallucinations
* Break downstream workflows

**LLM Quality Gate** answers one critical question:

> *Did this change make the system better or worse — and can we prove it automatically?*

---

## What This Project Builds

LLM Quality Gate is an **end-to-end evaluation pipeline** that:

* Runs automated LLM evaluations on every change
* Computes multiple quality metrics
* Enforces pass/fail quality gates in CI
* Sends alerts when thresholds are violated
* Tracks quality trends over time
* Allows **side-by-side comparison across LLM providers and models**

---

## Core Features

### 1. Evaluation Dataset (Golden Set)

A version-controlled dataset of test cases used to validate prompt behavior.

Each test case includes:

* Input prompt
* Reference / expected output
* Metadata (task type, difficulty, tags)

**Purpose**:

* Enables deterministic regression testing
* Makes prompt and model changes auditable

---

### 2. LLM Inference Layer (Pluggable Providers)

LLM Quality Gate includes a **unified LLM interface** with multiple provider adapters.

#### Supported Providers

**Free / Open-source Hosted**

* Groq (default)
* Hugging Face Inference API
* OpenRouter (free models only)

**Proprietary APIs (User-Provided Keys)**

* OpenAI (GPT‑4.x, GPT‑4o, GPT‑3.5)
* Anthropic Claude (Claude 3 family)
* Google Gemini

**Optional Local Providers**

* Ollama
* LocalAI

#### Key Characteristics

* Provider selected via configuration (not code)
* Unified request/response schema
* Deterministic generation (temperature = 0)
* Separate *Generator* and *Judge* roles
* Easy provider/model swapping for benchmarking

This design avoids vendor lock-in and supports real-world evaluation workflows.

---

### 3. Evaluation Metrics Engine

#### a) Task Success Metric

Measures whether the model successfully completes the task.

Methods:

* Exact match (when applicable)
* Semantic similarity using sentence embeddings

Outputs:

* Per-test score
* Aggregate success rate

---

#### b) Relevance Score

Measures how closely the output relates to the input prompt.

Method:

* Embedding similarity between prompt and output

Purpose:

* Detects off-topic or low-signal responses
* Lightweight and deterministic

---

#### c) Hallucination Risk Detection (Non-Rule-Based)

Hallucination detection is implemented using an **LLM-as-Judge** approach.

How it works:

* A configurable *Judge LLM* evaluates whether the output contains unsupported factual claims
* Uses a fixed evaluation prompt for consistency
* Returns a binary verdict: *Grounded* or *Hallucinated*

Properties:

* No brittle rules or regexes
* Widely used in modern LLM evaluation frameworks
* Works across providers and models

---

#### d) Stability / Consistency Metric

Measures output consistency across multiple runs of the same prompt.

Method:

* Run the prompt multiple times
* Compute pairwise semantic similarity
* Lower consistency indicates higher hallucination risk

This captures non-deterministic failure modes that single-run metrics miss.

---

### 4. Quality Gates (Pass / Fail Logic)

Predefined thresholds enforce minimum quality standards.

Examples:

* Minimum task success rate
* Maximum hallucination rate
* Minimum relevance score

If any threshold is violated:

* CI pipeline fails
* Deployment is blocked

LLM quality becomes **enforceable**, not subjective.

---

### 5. CI Integration (GitHub Actions)

On every:

* Prompt update
* Evaluation dataset change
* Model or provider configuration change

The CI pipeline:

1. Runs evaluations
2. Computes metrics
3. Compares results to baseline thresholds
4. Fails the PR if quality regresses

Prompts and model choices become **testable, reviewable artifacts**.

---

### 6. Alerting System

When evaluations fail:

* Slack notification is sent
* Email notification via GitHub Actions

Alerts include:

* Failed metric
* Current vs baseline score
* Provider and model used
* Commit hash

---

### 7. Historical Tracking & Storage

All evaluation runs are stored in a lightweight database (SQLite or DuckDB).

Each record includes:

* Timestamp
* Commit hash
* Provider and model name
* Metric scores

This enables:

* Trend analysis across prompt changes
* Cross-provider/model comparison
* Regression debugging

---

### 8. Web Dashboard (Tailwind CSS)

A modern dashboard for exploring evaluation results.

Features:

* Metric trends over time
* Pass/fail history
* Provider and model comparisons
* Per-test case drill-down

Tech stack:

* FastAPI backend
* Tailwind CSS frontend
* Chart.js for visualization

---

## System Architecture

**High-level flow**:

Prompt / Model Change → CI Trigger → LLM Evaluation (Provider Adapter) → Metrics Engine → Quality Gate → Alerts + Storage → Dashboard

---

## Design Principles

* Provider-agnostic by default
* Zero-cost capable (using free tiers)
* Deterministic evaluation
* CI-first mindset
* Reproducible and auditable results

---

## Tech Stack

**Backend**

* Python
* FastAPI
* pytest

**LLM & ML**

* Groq (default)
* OpenAI / Claude / Gemini (optional)
* Hugging Face / OpenRouter
* sentence-transformers for embeddings

**Frontend**

* Tailwind CSS
* Chart.js

**Storage**

* SQLite or DuckDB

**CI/CD**

* GitHub Actions

---

## Repository Structure

```
llm-quality-gate/
├── evals/
│   ├── dataset.json
│   ├── metrics.py
│   ├── judge.py
│   └── runner.py
├── llm/
│   ├── base.py          # Abstract LLM interface
│   ├── groq.py          # Groq adapter (default)
│   ├── openai.py        # OpenAI adapter
│   ├── claude.py        # Anthropic Claude adapter
│   ├── gemini.py        # Google Gemini adapter
│   ├── hf.py            # Hugging Face adapter
│   └── openrouter.py    # OpenRouter adapter
├── ci/
│   └── github_actions.yml
├── dashboard/
│   ├── backend/
│   └── frontend/
├── storage/
│   └── results.db
├── config.yaml
├── README.md
└── requirements.txt
```

---

## Who This Is For

* ML Engineers
* AI Engineers
* Platform Engineers
* Startups shipping LLM features
* Teams that need prompt and model regression safety

---

## How to Position This Project

> “LLM Quality Gate is a provider-agnostic evaluation system that enforces LLM output quality using CI-based testing, hallucination detection, and historical tracking — supporting Groq, OpenAI, Claude, Gemini, and open-source models.”

---

## Future Extensions

* RAG-specific groundedness metrics
* Automatic provider fallback on failures
* Cost-aware quality benchmarking
* Canary prompt testing
* Dataset growth automation

---

## License

MIT License

---

## Final Note

LLM Quality Gate is not a demo — it is an **engineering reference pattern**.

It demonstrates how modern teams should treat prompts, models, and providers: **tested, gated, observable, and swappable**.
