# LLMQ

[![Tests](https://github.com/Emart29/llm-quality-gate/actions/workflows/tests.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions/workflows/tests.yml)
[![LLM Quality Gate](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/Emart29/llm-quality-gate)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Emart29/llm-quality-gate/pulls)

**An open-source LLM regression testing & CI quality gate framework.**

Prevent prompt and model regressions before they reach production with automated testing across 8 LLM providers.

## Why LLMQ?

LLM applications fail silently. A prompt change that works in development can degrade performance in production. Model updates can break existing functionality. Without systematic testing, these regressions go undetected until users complain.

**Common LLM Regression Examples:**
- Prompt optimization improves one task but breaks another
- Model updates change response format, breaking downstream parsing
- Provider API changes affect response quality
- Temperature adjustments reduce consistency
- Context length changes truncate important information

LLMQ catches these issues before deployment with automated regression testing and quality gates.

## Quick Start

Get running in 5 minutes:

```bash
# 1. Install
pip install -e .

# 2. Initialize project
llmq init

# 3. Set API key (copy .env.example to .env)
echo "GROQ_API_KEY=your_key_here" >> .env

# 4. Run evaluation
llmq eval --provider groq
```

View results at `http://localhost:8000` after running `llmq dashboard`.

## CI Integration

Add to `.github/workflows/llm-quality-gate.yml`:

```yaml
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
      
      - name: Install LLMQ
        run: pip install -e .
      
      - name: Run Quality Gate
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          llmq eval --provider groq --fail-on-gate
```

## Architecture

![LLMQ Architecture](docs/architecture.svg)

**Flow:** Dataset → Provider → Metrics → Quality Gates → Results

## Supported Providers

| Provider | Models | API Key Required | Cost |
|----------|--------|------------------|------|
| **Groq** | Llama 3.1, Mixtral | ✅ | Free tier |
| **OpenAI** | GPT-3.5, GPT-4 | ✅ | Paid |
| **Claude** | Claude 3 Haiku/Sonnet | ✅ | Paid |
| **Gemini** | Gemini 1.5 Flash/Pro | ✅ | Free tier |
| **HuggingFace** | Open models | ✅ | Free |
| **OpenRouter** | 100+ models | ✅ | Varies |
| **Ollama** | Local models | ❌ | Free |
| **LocalAI** | Local models | ❌ | Free |

## CLI Commands

```bash
# Project setup
llmq init                           # Initialize new project
llmq doctor                         # Check system health

# Evaluation
llmq eval --provider groq           # Run evaluation
llmq eval --provider openai --fail-on-gate  # CI mode
llmq compare                        # Compare providers

# Management
llmq providers                      # List provider status
llmq runs --limit 10               # View recent runs
llmq dashboard                      # Start web interface
llmq settings --set '{"quality_gates": {"task_success_threshold": 0.9}}'
```

## Dashboard

![Dashboard Overview](docs/images/evaluation-summary.svg)

> 🎬 **[Interactive Demo](docs/demo.html)** — See the full CLI + Dashboard walkthrough.

**Features:**
- Historical performance tracking
- Provider comparison charts
- Quality gate pass/fail trends
- Test case drill-down analysis

## Configuration

`llmq.yaml`:
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

`evals/dataset.json`:
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

## Metrics

- **Task Success**: Exact match + semantic similarity
- **Relevance**: Embedding-based cosine similarity  
- **Hallucination**: LLM-as-judge detection
- **Consistency**: Multi-run variance analysis

## API

```bash
# Start evaluation
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq"}'

# Get results
curl http://localhost:8000/api/v1/runs

# Provider comparison
curl http://localhost:8000/api/v1/compare
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

**Development setup:**
```bash

git clone https://github.com/Emart29/llm-quality-gate.git
cd llm-quality-gate
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
llmq doctor  # Verify setup
```

## Roadmap

**v1.1**
- Custom metric plugins
- Slack/Discord webhooks
- A/B testing framework
- Performance benchmarking

**v1.2**
- Multi-language datasets
- Advanced regression analysis
- Cost tracking per provider
- Distributed evaluation

**v2.0**
- Visual prompt debugging
- Automated prompt optimization
- Enterprise SSO integration
- Advanced analytics

---

**License:** MIT | **Python:** 3.8+ | **Status:** Production Ready
