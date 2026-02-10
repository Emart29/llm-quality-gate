# LLMQ - LLM Quality Gate

A provider-agnostic evaluation framework for LLM-powered applications. Automatically test, measure, and enforce quality standards across any LLM provider.

## Features

- **8 LLM Providers**: Groq, OpenAI, Anthropic Claude, Google Gemini, HuggingFace, OpenRouter, Ollama, LocalAI
- **4 Core Metrics**: Task Success, Relevance, Hallucination Detection (LLM-as-Judge), Consistency
- **Quality Gates**: Configurable thresholds with automatic pass/fail enforcement
- **CI/CD Integration**: GitHub Actions workflow with PR comments and regression detection
- **Web Dashboard**: Dark-themed UI with Chart.js visualizations and Tailwind CSS
- **CLI Tool**: `llmq` commands for evaluation, comparison, and dashboard
- **Historical Tracking**: DuckDB storage with trend analysis and provider benchmarking

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Run evaluation
python -m cli.main eval --provider groq

# 4. Launch dashboard
python -m cli.main dashboard
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
├── dashboard/              # Web UI
│   ├── app.py             # FastAPI backend
│   └── static/index.html  # Tailwind + Chart.js frontend
├── cli/                    # Command-line interface
│   └── main.py            # llmq CLI tool
├── ci/                     # CI/CD integration
│   └── runner.py          # CI evaluation runner
├── tests/                  # Test suite
├── config.yaml            # Centralized configuration
└── .github/workflows/     # GitHub Actions
```

## CLI Commands

```bash
# Initialize project
python -m cli.main init

# Run evaluation
python -m cli.main eval --provider groq --model llama3-8b-8192

# Launch dashboard
python -m cli.main dashboard --port 8000

# Compare providers
python -m cli.main compare

# Validate dataset
python -m cli.main validate --suggestions
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
2. Executes LLM evaluation on PRs
3. Posts detailed metrics as PR comments
4. Fails PRs when quality gates are breached

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
