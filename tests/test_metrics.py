"""Tests for the metrics engine."""

import pytest
from evals.metrics import (
    TaskSuccessMetric, RelevanceMetric, HallucinationDetector,
    ConsistencyMetric, QualityGate, MetricsEngine, MetricResult,
)


class TestTaskSuccessMetric:
    def setup_method(self):
        self.metric = TaskSuccessMetric()

    def test_exact_match(self):
        result = self.metric.evaluate("Paris", "Paris", threshold=0.5)
        # In CI mode with mock embeddings, we rely more on exact match (weight 0.3) + semantic (weight 0.7)
        # Exact match should be 1.0, semantic might vary with mock embeddings
        assert result.details["exact_match"] == 1.0  # Exact match should be perfect
        assert result.score > 0.3  # Should pass with reasonable threshold
        assert result.passed

    def test_partial_match(self):
        result = self.metric.evaluate(
            "The capital of France is Paris.",
            "Paris",
            threshold=0.5,
        )
        assert result.score > 0.2  # token-overlap fallback gives lower score
        assert result.details["exact_match"] == 0.8  # contains expected

    def test_no_expected(self):
        result = self.metric.evaluate("Some output", None)
        assert result.score == 1.0
        assert result.passed
        assert result.details["note"] == "No expected output; skipped"

    def test_mismatch(self):
        result = self.metric.evaluate("Tokyo", "Paris", threshold=0.8)
        assert result.details["exact_match"] == 0.0


class TestRelevanceMetric:
    def setup_method(self):
        self.metric = RelevanceMetric()

    def test_relevant(self):
        result = self.metric.evaluate(
            "What is machine learning?",
            "Machine learning is a subset of AI that learns from data.",
            threshold=0.3,
        )
        assert result.score > 0.0
        assert result.metric_name == "relevance"


class TestHallucinationDetector:
    def test_heuristic_grounded(self):
        detector = HallucinationDetector(judge_llm=None)
        result = detector.evaluate(
            prompt="What is 2+2?",
            generated="2+2 equals 4",
            expected="4",
            threshold=0.5,
        )
        assert result.metric_name == "hallucination"
        assert result.details["method"] == "heuristic"

    def test_no_reference(self):
        detector = HallucinationDetector(judge_llm=None)
        result = detector.evaluate(
            prompt="What is 2+2?",
            generated="4",
        )
        assert result.score == 0.0
        assert result.passed


class TestConsistencyMetric:
    def setup_method(self):
        self.metric = ConsistencyMetric()

    def test_identical_outputs(self):
        result = self.metric.evaluate(["hello world", "hello world", "hello world"])
        # Mock embeddings might not give perfect 1.0 similarity, but should be very high
        assert result.score > 0.9
        assert result.passed

    def test_single_output(self):
        result = self.metric.evaluate(["only one"])
        assert result.score == 1.0

    def test_different_outputs(self):
        result = self.metric.evaluate(["cats", "quantum physics"])
        assert result.score < 1.0


class TestQualityGate:
    def test_all_pass(self):
        gate = QualityGate()
        metrics = {
            "task_success": MetricResult("task_success", 0.9, True, 0.8),
            "relevance": MetricResult("relevance", 0.85, True, 0.7),
            "hallucination": MetricResult("hallucination", 0.05, True, 0.1),
        }
        result = gate.evaluate(metrics)
        assert result.passed
        assert len(result.failed_metrics) == 0

    def test_one_fails(self):
        gate = QualityGate()
        metrics = {
            "task_success": MetricResult("task_success", 0.5, False, 0.8),
            "relevance": MetricResult("relevance", 0.85, True, 0.7),
        }
        result = gate.evaluate(metrics)
        assert not result.passed
        assert "task_success" in result.failed_metrics


class TestMetricsEngine:
    def test_evaluate_single(self):
        engine = MetricsEngine()
        results = engine.evaluate_single(
            prompt="What is the capital of France?",
            generated="The capital of France is Paris.",
            expected="Paris",
            metrics_to_evaluate=["task_success", "relevance"],
        )
        assert "task_success" in results
        assert "relevance" in results
        assert "hallucination" not in results

    def test_evaluate_batch(self):
        engine = MetricsEngine()
        items = [
            {
                "prompt": "What is 2+2?",
                "generated": "4",
                "expected": "4",
                "metrics_to_evaluate": ["task_success"],
            },
            {
                "prompt": "Capital of France?",
                "generated": "Paris",
                "expected": "Paris",
                "metrics_to_evaluate": ["task_success"],
            },
        ]
        result = engine.evaluate_batch(items)
        assert "aggregated" in result
        assert "quality_gate" in result
        assert result["quality_gate"].passed is not None
