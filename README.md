# 🛡️ LLMQ Gate

> You changed one prompt. Summarization improved.  
> Classification silently broke. Nobody noticed for 3 days.  
> **LLMQ Gate makes this a CI problem, not a production incident.**

[![PyPI version](https://img.shields.io/pypi/v/llmq-gate.svg)](https://pypi.org/project/llmq-gate/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Status](https://github.com/Emart29/llm-quality-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions)

---

## The Problem

LLM applications have no test suite. Changes that seem safe — a prompt 
tweak, a model version bump, a temperature adjustment — can silently 
degrade performance on tasks you didn't test manually. A model update 
changes response formats overnight. Nobody notices until users complain 
or metrics tank a week later.

Traditional software has pytest. CI/CD pipelines catch regressions 
before they ship. LLM apps have had nothing equivalent — until now.

**LLMQ Gate brings the same regression-detection discipline to LLM 
applications.** Define test cases, set quality thresholds, run evals 
against any provider, and fail CI builds automatically when quality 
drops below your standards.

---

## How It Works
```
Dataset → LLM Provider → Metrics Engine → Quality Gates → Pass / Fail
```

1. Define test cases in `evals/dataset.json` — inputs, expected 
   outputs, context
2. Run evals against any supported provider (Groq, OpenAI, Gemini, 
   Ollama, and more)
3. Four metrics computed automatically — task success, relevance, 
   hallucination, consistency
4. Quality gates pass or fail against your configured thresholds
5. Results stored for historical tracking and regression detection
6. CI build fails if quality drops — just like a broken unit test

---

## ⚡ Quickstart
```bash
pip install llmq-gate && llmq init && llmq eval --provider groq
```
```
$ llmq eval --provider groq --fail-on-gate

Loading configuration... ✓
Connecting to Groq (llama-3.1-8b-instant)... ✓
Loading dataset (12 test cases)... ✓

Running evaluations:
  ✓ question_answering_1    Task Success: 0.95  Relevance: 0.92  Hallucination: 0.02
  ✓ summarization_1         Task Success: 0.88  Relevance: 0.94  Hallucination: 0.01
  ✓ classification_1        Task Success: 1.00  Relevance: 0.89  Hallucination: 0.00
  ✗ sentiment_analysis_1    Task Success: 0.72  Relevance: 0.85  Hallucination: 0.03
  ✓ code_generation_1       Task Success: 0.91  Relevance: 0.96  Hallucination: 0.01
  ... (7 more)

Metrics Summary:
  Task Success:     0.87 (threshold: 0.80) ✓
  Relevance:        0.91 (threshold: 0.70) ✓
  Hallucination:    0.02 (threshold: 0.10) ✓
  Consistency:      0.94 (threshold: 0.80) ✓

Quality Gate: PASS ✓
```

---

## 📊 Metrics

| Metric | Method | What It Answers |
|--------|--------|-----------------|
| Task Success | Exact match + semantic similarity | Did the model get it right? |
| Relevance | Embedding cosine similarity | Is the response on-topic? |
| Hallucination | LLM-as-judge | Did it fabricate information? |
| Consistency | Multi-run variance | Are outputs stable across runs? |

Each metric maps to a real failure mode teams encounter in production.
Hallucination detection uses an LLM-as-judge pattern — a second model 
evaluates whether the response introduces facts not grounded in the 
provided context.

---

## 🔌 Supported Providers

| Provider | Models | Cost |
|----------|--------|------|
| Groq | Llama 3.1, Mixtral | Free tier |
| OpenAI | GPT-3.5, GPT-4 | Paid |
| Claude | Claude 3 Haiku / Sonnet | Paid |
| Gemini | Gemini 1.5 Flash / Pro | Free tier |
| HuggingFace | Open models | Free |
| OpenRouter | 100+ models | Varies |
| Ollama | Local models | Free |
| LocalAI | Local models | Free |

---

## 🔄 CI/CD Integration

One workflow file. Builds fail automatically when LLM quality drops 
below your thresholds — the same way a broken unit test fails a build.
```yaml
# .github/workflows/llm-quality-gate.yml
name: LLM Quality Gate

on:
  pull_request:
    paths: ['prompts/**', 'llm/**', 'llmq.yaml']

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install LLMQ Gate
        run: pip install llmq-gate

      - name: Run Quality Gate
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: llmq eval --provider groq --fail-on-gate
```

**Exit codes:** `0` pass · `1` fail or runtime error · `2` config error

This means your pipeline knows exactly what went wrong — quality 
regression, runtime failure, or misconfiguration — without parsing 
logs manually.

---

## ⚙️ Configuration

**`llmq.yaml`**
```yaml
llm:
  default_provider: "groq"
  temperature: 0.0
  max_tokens: 1000

providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    model: "llama-3.1-8b-instant"
  openai:
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-3.5-turbo"

quality_gates:
  task_success_threshold: 0.8
  relevance_threshold: 0.7
  hallucination_threshold: 0.1
```

**`evals/dataset.json`**
```json
{
  "test_cases": [
    {
      "id": "example_1",
      "task_type": "question_answering",
      "input": "What is the capital of France?",
      "expected_output": "Paris",
      "context": "Geography question",
      "reference": "Paris is the capital of France."
    }
  ]
}
```

---

## 🖥️ Dashboard
```bash
llmq dashboard
# → http://localhost:8000
```

Historical performance tracking, provider comparisons, quality gate 
trends, and per-test-case drill-down — all in one view. See exactly 
which test cases are degrading and when the regression started.

---

## 🔧 CLI Reference
```bash
llmq init                                   # Initialize project
llmq doctor                                 # Check system health
llmq eval --provider groq                   # Run evaluation
llmq eval --provider openai --fail-on-gate  # CI mode — exits 1 on fail
llmq compare                                # Compare providers side-by-side
llmq providers                              # List provider status
llmq runs --limit 10                        # View recent runs
llmq dashboard                              # Start web dashboard
```

---

## 🌐 REST API
```bash
# Trigger an evaluation programmatically
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq"}'

# View recent runs
curl http://localhost:8000/api/v1/runs

# Compare providers
curl http://localhost:8000/api/v1/compare
```

---

## 📁 Project Structure
```
llm-quality-gate/
├── core/                   # Metrics engine (task success, relevance,
│                           # hallucination, consistency)
├── llm/                    # Provider abstractions (Groq, OpenAI, etc.)
├── evals/                  # Evaluation runner + dataset loader
├── cli/                    # llmq CLI commands
├── dashboard/              # Web dashboard (FastAPI + frontend)
├── storage/                # Run history + result persistence
├── ci/                     # CI integration helpers
├── tests/                  # Pytest test suite
├── .github/workflows/      # GitHub Actions pipeline
├── llmq.yaml               # Project configuration
├── config.example.yaml     # Configuration reference
└── pyproject.toml          # Package metadata (published to PyPI)
```

---

## 📌 Key Engineering Decisions

**Why LLM-as-judge for hallucination detection?**
Rule-based approaches (keyword matching, NLI models) miss subtle 
fabrications. Using a second LLM to evaluate grounding catches 
hallucinations that simpler methods skip — at the cost of one 
extra API call per evaluation.

**Why semantic similarity for relevance instead of exact match?**
Exact match punishes correct answers that use different wording. 
Embedding cosine similarity measures whether the response is 
*semantically* on-topic — which is what relevance actually means 
for generative outputs.

**Why multi-run variance for consistency?**
LLMs are non-deterministic at temperature > 0. A model that gives 
correct answers 70% of the time is not production-ready. Consistency 
scoring surfaces this instability before it reaches users.

**Why fail the CI build instead of just reporting?**
Reporting creates alert fatigue — teams learn to ignore dashboards. 
Failing the build makes quality regression a blocking issue that 
must be resolved before merging, the same discipline applied to 
unit tests.

---

## 📦 Installation & Version Notes
```bash
pip install llmq-gate
```

**v0.1.1 (current)**
- Unified config: `llmq.yaml` everywhere
- Config auto-discovery from current directory upward
- Standalone eval mode — no dashboard dependency
- Standardized exit codes (0/1/2)

*Migrating from ≤0.1.0?* Rename `config.yaml` → `llmq.yaml`. 
That's the only breaking change.

---

## 🗺️ Roadmap

**v1.1**
Custom metric plugins · Slack/Discord webhooks · 
A/B testing · Performance benchmarks

**v1.2**
Multi-language datasets · Regression analysis · 
Cost tracking · Distributed eval

**v2.0**
Visual prompt debugging · Automated prompt optimization · 
Enterprise SSO

---

## Contributing
```bash
git clone https://github.com/Emart29/llm-quality-gate.git
cd llm-quality-gate
python -m venv venv && source venv/bin/activate
pip install -e . && llmq doctor
```

Fork → branch → test (`python -m pytest tests/ -v`) → PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

**Built by [Emmanuel Nwanguma](https://linkedin.com/in/nwangumaemmanuel)**  
[⭐ Star on GitHub](https://github.com/Emart29/llm-quality-gate) ·
[📦 PyPI](https://pypi.org/project/llmq-gate/) ·
[🌐 Website](https://llm-quality-gate.vercel.app)
