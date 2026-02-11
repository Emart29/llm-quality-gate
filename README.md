# LLMQ - LLM Quality Gate

A provider-agnostic evaluation framework for LLM-powered applications. Automatically test, measure, and enforce quality standards across any LLM provider.

## Features

- **8 LLM Providers**: Groq, OpenAI, Anthropic Claude, Google Gemini, HuggingFace, OpenRouter, Ollama, LocalAI
- **4 Core Metrics**: Task Success, Relevance, Hallucination Detection (LLM-as-Judge), Consistency
- **Quality Gates**: Configurable thresholds with automatic pass/fail enforcement
- **CI/CD Integration**: GitHub Actions workflow with PR comments and regression detection
- **Web Dashboard**: Dark-themed UI with Chart.js visualizations and Tailwind CSS
- **Web + API First**: FastAPI `api/v1` endpoints drive dashboard and automation
- **CLI Integration**: optional `llmq` client that calls the API
- **Historical Tracking**: DuckDB storage with trend analysis and provider benchmarking

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Launch dashboard + API
python -m integrations.cli.main dashboard

# 4. Trigger evaluation through API client
python -m integrations.cli.main eval --provider groq
```

## Project Structure

```
llm-quality-gate/
├── llm/                    # Provider infrastructure
│   ├── base.py            # Abstract LLMProvider, LLMRequest, LLMResponse
│   ├── factory.py         # Provider factory with role-based creation
│   ├── groq.py            # Groq (default free provider)
│   ├── openai.py          # OpenAI GPT family
│   ├── claude.py          # Anthropic Claude
│   ├── gemini.py          # Google Gemini
│   ├── huggingface.py     # HuggingFace Inference
│   ├── openrouter.py      # OpenRouter (multi-model)
│   ├── ollama.py          # Ollama (local)
│   └── localai.py         # LocalAI (local)
├── evals/                  # Evaluation framework
│   ├── dataset.py         # Dataset schema & loader
│   ├── dataset.json       # Golden test dataset
│   ├── runner.py          # Test execution engine
│   ├── metrics.py         # MetricsEngine, QualityGate
│   ├── comprehensive_runner.py  # Full eval + scoring pipeline
│   ├── validator.py       # Dataset validation
│   └── maintenance.py     # Versioning & growth automation
├── storage/                # Persistence layer
│   ├── database.py        # DuckDB schema & connection
│   ├── models.py          # Data models
│   └── repository.py      # CRUD operations
├── dashboard/              # Web UI + API
│   ├── app.py             # FastAPI app + static hosting
│   ├── api.py             # Central /api/v1 router
│   └── static/index.html  # Tailwind + Chart.js frontend
├── integrations/          # Optional external integrations
│   └── cli/main.py        # API client for llmq commands
├── cli/                   # Backward-compatible shim
│   └── main.py
├── ci/                     # CI/CD integration
│   └── runner.py          # CI evaluation runner
├── tests/                  # Test suite
├── config.yaml            # Centralized configuration
└── .github/workflows/     # GitHub Actions
```

## API Endpoints (Web-First)

```bash
# Trigger evaluation
POST /api/v1/evaluate

# List providers & models
GET /api/v1/providers

# Retrieve historical runs
GET /api/v1/runs

# Compare provider metrics
GET /api/v1/compare

# Configure quality gates/settings
GET/POST /api/v1/settings

# Canary flow (run subset then optionally promote)
POST /api/v1/evaluate/canary

# Mark a run as baseline for regression detection
POST /api/v1/runs/{run_id}/baseline
```

## CLI Integration (Optional)

```bash
# Run evaluation via API
python -m integrations.cli.main eval --provider groq

# Compare providers via API
python -m integrations.cli.main compare

# List providers
python -m integrations.cli.main providers
```

## Configuration

All settings are in `config.yaml`:

```yaml
llm:
  default_provider: "groq"
  temperature: 0.0          # Deterministic by default

providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    model: "llama3-8b-8192"
  openai:
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-3.5-turbo"
  # ... all 8 providers

roles:
  generator:
    provider: "groq"         # Fast, free for generation
  judge:
    provider: "openai"       # Reliable for evaluation

quality_gates:
  task_success_threshold: 0.8
  relevance_threshold: 0.7
  hallucination_threshold: 0.1
  consistency_threshold: 0.8
```

## Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| Task Success | Exact match + semantic similarity | Does the output match expected answer? |
| Relevance | Embedding cosine similarity | Is the output relevant to the prompt? |
| Hallucination | LLM-as-Judge + heuristic fallback | Does the output contain fabricated info? |
| Consistency | Pairwise similarity across runs | Are multiple runs producing similar outputs? |

## CI/CD Integration

The GitHub Actions workflow automatically:
1. Runs unit tests
2. Boots the FastAPI service and executes API-first evaluations on PRs
3. Enforces metric thresholds and quality gates via API response checks
4. Posts detailed metrics, provider info, commit hash, and regression info as PR comments
5. Fails PRs when thresholds or quality gates are breached

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
