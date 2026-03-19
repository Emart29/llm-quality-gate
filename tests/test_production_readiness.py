"""Production readiness tests — validates the complete user workflow.

These tests verify that the framework works end-to-end as a user would
experience it: config loading, dataset validation, metrics computation,
quality gate evaluation, database persistence, and dashboard startup.
"""

import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("CI", "true")


# ── Config ────────────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_resolve_config_path_finds_llmq_yaml(self):
        from core.config import resolve_config_path
        path = resolve_config_path()
        assert path.name == "llmq.yaml"
        assert path.exists()

    def test_config_has_required_sections(self):
        from core.config import resolve_config_path
        with open(resolve_config_path()) as f:
            config = yaml.safe_load(f)
        assert "llm" in config
        assert "providers" in config
        assert "quality_gates" in config
        assert config["llm"]["default_provider"] == "groq"

    def test_all_eight_providers_configured(self):
        from core.config import resolve_config_path
        with open(resolve_config_path()) as f:
            config = yaml.safe_load(f)
        expected = {"groq", "openai", "claude", "gemini", "huggingface",
                    "openrouter", "ollama", "localai"}
        assert set(config["providers"].keys()) == expected


# ── Dataset ───────────────────────────────────────────────────────────────

class TestDataset:
    def test_load_golden_dataset(self):
        from evals.dataset import DatasetLoader
        ds = DatasetLoader.load_from_file("evals/dataset.json")
        assert ds.name
        assert ds.version
        assert len(ds.test_cases) >= 10

    def test_all_test_cases_have_required_fields(self):
        from evals.dataset import DatasetLoader
        ds = DatasetLoader.load_from_file("evals/dataset.json")
        for tc in ds.test_cases:
            assert tc.id, "Test case missing id"
            assert tc.input_prompt, f"{tc.id} missing input_prompt"
            assert tc.task_type, f"{tc.id} missing task_type"


# ── LLM Factory ───────────────────────────────────────────────────────────

class TestLLMFactory:
    def test_all_providers_registered(self):
        from llm.factory import LLMFactory
        assert len(LLMFactory.PROVIDERS) == 8
        assert "groq" in LLMFactory.PROVIDERS
        assert "gemini" in LLMFactory.PROVIDERS

    def test_factory_loads_config(self):
        from llm.factory import LLMFactory
        factory = LLMFactory("llmq.yaml")
        assert factory.config is not None
        assert "providers" in factory.config


# ── Provider Imports ──────────────────────────────────────────────────────

class TestProviderImports:
    """All providers should import without crashing, even without API keys."""

    def test_groq_import(self):
        from llm.groq import GroqProvider  # noqa: F401

    def test_openai_import(self):
        from llm.openai import OpenAIProvider  # noqa: F401

    def test_claude_import(self):
        from llm.claude import ClaudeProvider  # noqa: F401

    def test_gemini_lazy_import(self):
        from llm.gemini import GeminiProvider  # noqa: F401

    def test_huggingface_import(self):
        from llm.huggingface import HuggingFaceProvider  # noqa: F401

    def test_openrouter_import(self):
        from llm.openrouter import OpenRouterProvider  # noqa: F401

    def test_ollama_import(self):
        from llm.ollama import OllamaProvider  # noqa: F401

    def test_localai_import(self):
        from llm.localai import LocalAIProvider  # noqa: F401


# ── Pydantic Models ──────────────────────────────────────────────────────

class TestModels:
    def test_llm_request_metadata_isolation(self):
        from llm.base import LLMRequest
        a = LLMRequest(prompt="a")
        b = LLMRequest(prompt="b")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata

    def test_llm_response_metadata_isolation(self):
        from llm.base import LLMResponse
        a = LLMResponse(content="a", model="m", provider="p")
        b = LLMResponse(content="b", model="m", provider="p")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata

    def test_llm_response_usage_isolation(self):
        from llm.base import LLMResponse
        a = LLMResponse(content="a", model="m", provider="p")
        b = LLMResponse(content="b", model="m", provider="p")
        a.usage["prompt_tokens"] = 100
        assert "prompt_tokens" not in b.usage


# ── Metrics Engine ────────────────────────────────────────────────────────

class TestMetricsEngine:
    def test_evaluate_single_returns_metric_results(self):
        from evals.metrics import MetricsEngine
        engine = MetricsEngine()
        result = engine.evaluate_single(
            prompt="What is 2+2?",
            generated="The answer is 4.",
            expected="4",
        )
        assert "task_success" in result
        assert "relevance" in result
        assert hasattr(result["task_success"], "score")
        assert 0 <= result["task_success"].score <= 1

    def test_quality_gate_pass(self):
        from evals.metrics import QualityGate, MetricResult
        gate = QualityGate(thresholds={
            "task_success": 0.8, "relevance": 0.7,
            "hallucination": 0.1, "consistency": 0.8,
        })
        metrics = {
            "task_success": MetricResult("task_success", 0.85, True, 0.8),
            "relevance": MetricResult("relevance", 0.75, True, 0.7),
            "hallucination": MetricResult("hallucination", 0.05, True, 0.1),
            "consistency": MetricResult("consistency", 0.9, True, 0.8),
        }
        result = gate.evaluate(metrics)
        assert result.passed is True

    def test_quality_gate_fail_on_hallucination(self):
        from evals.metrics import QualityGate, MetricResult
        gate = QualityGate(thresholds={
            "task_success": 0.8, "hallucination": 0.1,
        })
        metrics = {
            "task_success": MetricResult("task_success", 0.85, True, 0.8),
            "hallucination": MetricResult("hallucination", 0.3, False, 0.1),
        }
        result = gate.evaluate(metrics)
        assert result.passed is False
        assert "hallucination" in result.failed_metrics


# ── Database ──────────────────────────────────────────────────────────────

class TestDatabase:
    def test_schema_creates_all_tables(self):
        from storage.database import Database
        with tempfile.TemporaryDirectory() as td:
            db = Database(os.path.join(td, "test.db"))
            try:
                tables = db.conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main'"
                ).fetchall()
                names = {t[0] for t in tables}
                assert {"evaluation_runs", "test_case_results",
                        "quality_gate_history", "provider_comparisons"} <= names
            finally:
                db.close()

    def test_insert_and_query_evaluation_run(self):
        from storage.database import Database
        with tempfile.TemporaryDirectory() as td:
            db = Database(os.path.join(td, "test.db"))
            try:
                db.conn.execute(
                    "INSERT INTO evaluation_runs "
                    "(id, provider_name, model_name, dataset_name, dataset_version, "
                    "total_test_cases, successful_executions, failed_executions, "
                    "overall_score, total_execution_time, success_rate, "
                    "quality_gate_passed, created_at) "
                    "VALUES ('r1','groq','llama','ds','1.0',"
                    "10,8,2,0.85,30.5,0.8,true,NOW())"
                )
                rows = db.conn.execute("SELECT * FROM evaluation_runs").fetchall()
                assert len(rows) == 1
            finally:
                db.close()


# ── Storage Models ────────────────────────────────────────────────────────

class TestStorageModels:
    def test_evaluation_run_default_factory_isolation(self):
        from storage.models import EvaluationRun
        a = EvaluationRun(
            id="a", dataset_name="d", dataset_version="1", provider_name="p",
            model_name="m", total_test_cases=1, successful_executions=1,
            failed_executions=0, total_execution_time=1.0, success_rate=1.0,
        )
        b = EvaluationRun(
            id="b", dataset_name="d", dataset_version="1", provider_name="p",
            model_name="m", total_test_cases=1, successful_executions=1,
            failed_executions=0, total_execution_time=1.0, success_rate=1.0,
        )
        a.configuration["k"] = "v"
        assert "k" not in b.configuration


# ── CI Quality Gate ───────────────────────────────────────────────────────

class TestCIQualityGate:
    def test_enforce_thresholds_pass(self):
        from ci.api_quality_gate import enforce_thresholds
        result = {"result": {"aggregated_metrics": {
            "task_success": {"score": 0.85, "threshold": 0.8},
            "hallucination": {"score": 0.05, "threshold": 0.1},
        }}}
        passed, failures = enforce_thresholds(result)
        assert passed is True
        assert failures == []

    def test_enforce_thresholds_fail(self):
        from ci.api_quality_gate import enforce_thresholds
        result = {"result": {"aggregated_metrics": {
            "task_success": {"score": 0.5, "threshold": 0.8},
        }}}
        passed, failures = enforce_thresholds(result)
        assert passed is False
        assert len(failures) == 1

    def test_hallucination_inverted_threshold(self):
        from ci.api_quality_gate import enforce_thresholds
        result = {"result": {"aggregated_metrics": {
            "hallucination": {"score": 0.5, "threshold": 0.1},
        }}}
        passed, _ = enforce_thresholds(result)
        assert passed is False


# ── EvaluationService ─────────────────────────────────────────────────────

class TestEvaluationService:
    def test_service_initializes(self):
        from evals.service import EvaluationService
        svc = EvaluationService()
        assert isinstance(svc._jobs, dict)


# ── Dashboard ─────────────────────────────────────────────────────────────

class TestDashboard:
    def test_app_has_required_routes(self):
        from dashboard.app import app
        routes = {r.path for r in app.routes}
        required = {"/api/v1/evaluate", "/api/v1/providers",
                    "/api/v1/runs", "/api/v1/settings"}
        assert required <= routes

    def test_no_duplicate_api_runs_route(self):
        from dashboard.app import app
        routes = [r.path for r in app.routes]
        assert "/api/runs" not in routes, "Duplicate /api/runs should be removed"


# ── CLI ───────────────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_app_imports(self):
        from cli.main import app
        assert app is not None
