"""Comprehensive evaluation runner that combines test execution with metrics scoring."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .dataset import EvaluationDataset, TestCase, EvaluationMetric
from .runner import EvaluationRunner, EvaluationResult, TestCaseResult
from .metrics import MetricsEngine, MetricResult, QualityGateResult

logger = logging.getLogger(__name__)


@dataclass
class ScoredTestCaseResult:
    """A test case result enriched with metric scores."""
    test_case_id: str
    test_case: TestCase
    generated_output: str
    execution_time: float
    success: bool
    error: Optional[str] = None
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        if not self.success:
            return False
        return all(m.passed for m in self.metrics.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "input_prompt": self.test_case.input_prompt,
            "expected_output": self.test_case.expected_output,
            "generated_output": self.generated_output,
            "execution_time": self.execution_time,
            "success": self.success,
            "passed": self.passed,
            "error": self.error,
            "task_type": self.test_case.task_type.value,
            "difficulty": self.test_case.difficulty.value,
            "tags": self.test_case.tags,
            "metrics": {
                name: {
                    "score": m.score,
                    "passed": m.passed,
                    "threshold": m.threshold,
                    "details": m.details,
                }
                for name, m in self.metrics.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class ComprehensiveEvaluationResult:
    """Full evaluation result with metrics, quality gate, and per-case scores."""
    dataset_name: str
    dataset_version: str
    provider_name: str
    model_name: str
    execution_timestamp: datetime
    total_test_cases: int
    successful_executions: int
    failed_executions: int
    total_execution_time: float
    scored_results: List[ScoredTestCaseResult]
    aggregated_metrics: Dict[str, MetricResult]
    quality_gate: QualityGateResult
    configuration: Dict[str, Any]

    @property
    def pass_rate(self) -> float:
        if not self.scored_results:
            return 0.0
        return sum(1 for r in self.scored_results if r.passed) / len(self.scored_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "total_test_cases": self.total_test_cases,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "total_execution_time": self.total_execution_time,
            "pass_rate": self.pass_rate,
            "quality_gate": {
                "passed": self.quality_gate.passed,
                "overall_score": self.quality_gate.overall_score,
                "failed_metrics": self.quality_gate.failed_metrics,
            },
            "aggregated_metrics": {
                name: {
                    "score": m.score,
                    "passed": m.passed,
                    "threshold": m.threshold,
                    "details": m.details,
                }
                for name, m in self.aggregated_metrics.items()
            },
            "scored_results": [r.to_dict() for r in self.scored_results],
            "configuration": self.configuration,
        }


class ComprehensiveEvaluationRunner:
    """Runs evaluation and scores results with the metrics engine."""

    def __init__(
        self,
        evaluation_runner: EvaluationRunner,
        metrics_engine: MetricsEngine,
    ):
        self.runner = evaluation_runner
        self.metrics = metrics_engine

    async def run(
        self,
        dataset: EvaluationDataset,
        provider_name: str,
        model_name: str,
        **runner_kwargs,
    ) -> ComprehensiveEvaluationResult:
        """Execute tests and score them."""
        # 1. Run generation
        eval_result: EvaluationResult = await self.runner.run_evaluation(
            dataset=dataset,
            provider_name=provider_name,
            model_name=model_name,
            **runner_kwargs,
        )

        # 2. Score each test case
        scored: List[ScoredTestCaseResult] = []
        batch_items: List[Dict[str, Any]] = []

        for tcr in eval_result.test_case_results:
            tc = tcr.test_case
            metrics_list = [m.value for m in tc.metrics_to_evaluate]

            if tcr.success:
                case_metrics = self.metrics.evaluate_single(
                    prompt=tc.input_prompt,
                    generated=tcr.generated_output,
                    expected=tc.expected_output,
                    context=tc.context,
                    metrics_to_evaluate=metrics_list,
                    custom_thresholds=tc.custom_thresholds,
                )
                batch_items.append({
                    "prompt": tc.input_prompt,
                    "generated": tcr.generated_output,
                    "expected": tc.expected_output,
                    "context": tc.context,
                    "metrics_to_evaluate": metrics_list,
                    "custom_thresholds": tc.custom_thresholds,
                })
            else:
                case_metrics = {}

            scored.append(ScoredTestCaseResult(
                test_case_id=tcr.test_case_id,
                test_case=tc,
                generated_output=tcr.generated_output,
                execution_time=tcr.execution_time,
                success=tcr.success,
                error=tcr.error,
                metrics=case_metrics,
                metadata=tcr.metadata or {},
            ))

        # 3. Aggregate and gate
        if batch_items:
            batch_result = self.metrics.evaluate_batch(batch_items)
            aggregated = batch_result["aggregated"]
            gate = batch_result["quality_gate"]
        else:
            aggregated = {}
            gate = QualityGateResult(passed=False, failed_metrics=["no_successful_runs"])

        return ComprehensiveEvaluationResult(
            dataset_name=eval_result.dataset_name,
            dataset_version=eval_result.dataset_version,
            provider_name=eval_result.provider_name,
            model_name=eval_result.model_name,
            execution_timestamp=eval_result.execution_timestamp,
            total_test_cases=eval_result.total_test_cases,
            successful_executions=eval_result.successful_executions,
            failed_executions=eval_result.failed_executions,
            total_execution_time=eval_result.total_execution_time,
            scored_results=scored,
            aggregated_metrics=aggregated,
            quality_gate=gate,
            configuration=eval_result.configuration,
        )
