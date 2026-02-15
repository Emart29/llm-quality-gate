# LLMQ Gate
**Regression Testing & Quality Gates for LLM Applications**

LLMQ Gate is a pytest-like quality gate system for LLM-powered applications.

[![PyPI version](https://img.shields.io/pypi/v/llmq-gate?style=flat-square&color=blue)](https://pypi.org/project/llmq-gate/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/Emart29/llm-quality-gate/blob/main/LICENSE)
[![CI Status](https://img.shields.io/github/actions/workflow/status/Emart29/llm-quality-gate/ci.yml?label=tests&style=flat-square)](https://github.com/Emart29/llm-quality-gate/actions)

---

## The Problem

LLM applications fail silently. No stack trace when your classifier drifts. No exception when prompts degrade. Model updates break functionality overnight without warning.

**LLMQ Gate catches regressions before production:**
- Prompt changes that improve one task but break another
- Model updates that change response formats
- Provider API changes affecting output quality
- Configuration drift reducing consistency

## Quick Start

```bash
pip install llmq-gate
llmq init
echo "GROQ_API_KEY=your_key_here" >> .env
llmq eval --provider groq
```

## CLI Example

```bash
$ llmq eval --provider groq --fail-on-gate

LLMQ Gate v0.1.1 - LLM Quality Gate System
==========================================

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
Exit code: 0
```

## Dashboard

![LLMQ Gate Dashboard Screenshot](docs/dashboard.png)

Interactive web dashboard with historical performance tracking, provider comparisons, and detailed test case analysis.

```bash
llmq dashboard
# → http://localhost:8000
```

## How It Works

```
Dataset → LLM Provider → Metrics Engine → Quality Gates → Pass/Fail
```

1. **Define test cases** in `evals/dataset.json` with inputs, expected outputs, and context
2. **Run evaluations** against any supported provider  
3. **Metrics are computed** automatically — task success, relevance, hallucination, consistency
4. **Quality gates** pass or fail based on your configured thresholds
5. **Results are stored** for historical tracking and comparison

## Supported Providers

| Provider | Models | API Key | Cost |
|----------|--------|---------|------|
| **Groq** | Llama 3.1, Mixtral | Required | Free tier |
| **OpenAI** | GPT-3.5, GPT-4 | Required | Paid |
| **Claude** | Claude 3 Haiku / Sonnet | Required | Paid |
| **Gemini** | Gemini 1.5 Flash / Pro | Required | Free tier |
| **HuggingFace** | Open models | Required | Free |
| **OpenRouter** | 100+ models | Required | Varies |
| **Ollama** | Local models | — | Free |
| **LocalAI** | Local models | — | Free |

## CI/CD Integration

Add quality gates to your pull request workflow. Builds fail automatically when LLM performance drops below thresholds.

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

## Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| **Task Success** | Exact match + semantic similarity | Did the model produce the correct answer? |
| **Relevance** | Embedding-based cosine similarity | Is the response relevant to the input? |
| **Hallucination** | LLM-as-judge detection | Did the model fabricate information? |
| **Consistency** | Multi-run variance analysis | Are responses stable across runs? |

---

## Dashboard

```bash
llmq dashboard
```

The interactive web dashboard provides historical performance tracking, provider comparison charts, quality gate pass/fail trends, and test case drill-down analysis.

[🎬 Watch the full CLI + Dashboard walkthrough →](https://llm-quality-gate.vercel.app)


## v0.1.1 Highlights

- Unified configuration filename: `llmq.yaml` everywhere.
- Config auto-discovery from current directory upward (similar to `pyproject.toml` lookup).
- `llmq eval` now supports standalone mode by falling back to local engine if dashboard API is unavailable.
- CLI exit codes are standardized:
  - `0`: quality gate passed
  - `1`: quality gate failed or runtime error
  - `2`: configuration error (e.g., missing `llmq.yaml`)

## Configuration

**`llmq.yaml`** — project-level settings:

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

**`evals/dataset.json`** — test cases:

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

## CLI Reference

```bash
# Setup
llmq init                                    # Initialize new project
llmq doctor                                  # Check system health

# Evaluation
llmq eval --provider groq                    # Run evaluation
llmq eval --provider openai --fail-on-gate   # CI mode (exit 1 on gate failure)
llmq compare                                 # Compare providers side-by-side

# Management
llmq providers                               # List provider status
llmq runs --limit 10                         # View recent runs
llmq dashboard                               # Start web dashboard
llmq settings --set '{"quality_gates": {"task_success_threshold": 0.9}}'
```

---

## API

```bash
# Start an evaluation
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq"}'

# Get run history
curl http://localhost:8000/api/v1/runs

# Compare providers
curl http://localhost:8000/api/v1/compare
```

## Contributing

```bash
git clone https://github.com/Emart29/llm-quality-gate.git
cd llm-quality-gate
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
llmq doctor               # Verify setup
```

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`  
3. Make changes and add tests
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

## Roadmap

**v1.1** — Custom metric plugins · Slack/Discord webhooks · A/B testing framework · Performance benchmarking

**v1.2** — Multi-language datasets · Advanced regression analysis · Cost tracking per provider · Distributed evaluation  

**v2.0** — Visual prompt debugging · Automated prompt optimization · Enterprise SSO · Advanced analytics

---

## License

MIT — see [LICENSE](LICENSE) for details.
