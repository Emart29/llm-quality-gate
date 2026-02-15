# LLMQ Gate

**Catch silent LLM failures before they reach production.**

[![PyPI version](https://img.shields.io/pypi/v/llmq-gate.svg)](https://pypi.org/project/llmq-gate/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://pypi.org/project/llmq-gate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions)

LLMQ Gate is a pytest-like quality gate for LLM applications. Define test cases, run evals against any provider, and fail CI builds when quality regresses.

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

## Why

Your LLM app has no test suite. A prompt tweak improves summarization but silently breaks classification. A model update changes response formats overnight. Nobody notices until users complain.

LLMQ Gate makes this a CI problem, not a production incident.

---

## How It Works

```
Dataset → LLM Provider → Metrics Engine → Quality Gates → Pass / Fail
```

1. Define test cases in `evals/dataset.json` — inputs, expected outputs, context
2. Run evals against any supported provider
3. Four metrics computed automatically — task success, relevance, hallucination, consistency
4. Quality gates pass or fail against your thresholds
5. Results stored for historical tracking and regression detection

---

## Dashboard

![LLMQ Gate Dashboard Screenshot](docs/dashboard.png)

Historical performance tracking, provider comparisons, quality gate trends, and test case drill-down.

```bash
llmq dashboard
# → http://localhost:8000
```

[🎬 CLI + Dashboard walkthrough →](https://llm-quality-gate.vercel.app)

---

## Providers

| Provider | Models | Cost |
|---|---|---|
| **Groq** | Llama 3.1, Mixtral | Free tier |
| **OpenAI** | GPT-3.5, GPT-4 | Paid |
| **Claude** | Claude 3 Haiku / Sonnet | Paid |
| **Gemini** | Gemini 1.5 Flash / Pro | Free tier |
| **HuggingFace** | Open models | Free |
| **OpenRouter** | 100+ models | Varies |
| **Ollama** | Local models | Free |
| **LocalAI** | Local models | Free |

---

## Metrics

| Metric | Method | What it answers |
|---|---|---|
| **Task Success** | Exact match + semantic similarity | Did the model get it right? |
| **Relevance** | Embedding cosine similarity | Is the response on-topic? |
| **Hallucination** | LLM-as-judge | Did it fabricate information? |
| **Consistency** | Multi-run variance | Are outputs stable across runs? |

---

## CI/CD Integration

One workflow file. Builds fail when LLM quality drops below your thresholds.

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

Exit codes: `0` pass · `1` fail or runtime error · `2` config error

---

## Configuration

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

## CLI Reference

```bash
llmq init                                    # Initialize project
llmq doctor                                  # Check system health
llmq eval --provider groq                    # Run evaluation
llmq eval --provider openai --fail-on-gate   # CI mode
llmq compare                                 # Compare providers side-by-side
llmq providers                               # List provider status
llmq runs --limit 10                         # View recent runs
llmq dashboard                               # Start web dashboard
```

## API

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq"}'

curl http://localhost:8000/api/v1/runs
curl http://localhost:8000/api/v1/compare
```

---

## v0.1.1

- Unified config: `llmq.yaml` everywhere
- Config auto-discovery from current directory upward
- Standalone eval mode — no dashboard dependency
- Standardized exit codes (`0`/`1`/`2`)

**Migrating from ≤0.1.0?** Rename `config.yaml` → `llmq.yaml`. That's the main change.

---

## Roadmap

**v1.1** — Custom metric plugins · Slack/Discord webhooks · A/B testing · Performance benchmarks

**v1.2** — Multi-language datasets · Regression analysis · Cost tracking · Distributed eval

**v2.0** — Visual prompt debugging · Automated prompt optimization · Enterprise SSO

---

## Contributing

```bash
git clone https://github.com/Emart29/llm-quality-gate.git
cd llm-quality-gate
python -m venv venv && source venv/bin/activate
pip install -e . && llmq doctor
```

Fork → branch → test (`python -m pytest tests/ -v`) → PR.

---

## License

MIT — see [LICENSE](LICENSE).

---

[⭐ Star on GitHub](https://github.com/Emart29/llm-quality-gate) · [📦 PyPI](https://pypi.org/project/llmq-gate/) · [🌐 Website](https://llm-quality-gate.vercel.app)
