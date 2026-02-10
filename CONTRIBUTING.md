# Contributing to LLMQ

Thank you for your interest in contributing to the LLM Quality Gate project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/llm-quality-gate.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Workflow

1. Make your changes
2. Run the test suite: `python -m pytest tests/ -v`
3. Ensure all tests pass
4. Commit with a descriptive message
5. Push and open a pull request

## Project Structure

- `llm/` - Provider adapters and factory
- `evals/` - Evaluation dataset, runner, metrics
- `storage/` - Database and repository layer
- `dashboard/` - Web UI (FastAPI + Tailwind/Chart.js)
- `cli/` - Command-line interface
- `tests/` - Test suite

## Adding a New Provider

1. Create `llm/your_provider.py` extending `LLMProvider`
2. Implement the `async generate(request: LLMRequest) -> LLMResponse` method
3. Register in `llm/factory.py` `PROVIDERS` dict
4. Add configuration to `config.yaml`
5. Add tests in `tests/test_llm_providers.py`

## Adding Test Cases

Edit `evals/dataset.json` or use the dataset management API:

```python
from evals.dataset import TestCase, TaskType
tc = TestCase(
    input_prompt="Your prompt here",
    expected_output="Expected answer",
    task_type=TaskType.QUESTION_ANSWERING,
)
```

## Code Style

- Follow PEP 8
- Use type hints
- Keep functions focused and small
- Write tests for new functionality

## Reporting Issues

Open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS)
