"""Metrics engine for LLM evaluation: task success, relevance, hallucination, consistency."""

import logging
import re
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ── Lightweight embedding helpers (lazy-loaded) ───────────────────────────

_embedding_model = None
_is_ci_mode = None


def _is_ci_environment() -> bool:
    """Check if running in CI environment."""
    global _is_ci_mode
    if _is_ci_mode is None:
        _is_ci_mode = os.environ.get('CI', '').lower() in ('true', '1', 'yes')
    return _is_ci_mode


class MockEmbeddingModel:
    """Mock embedding model for CI environments to avoid network calls and model downloads."""
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return deterministic mock embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Create a simple deterministic embedding based on text characteristics
            text_lower = text.lower()
            embedding = [
                len(text) / 100.0,  # Length feature
                text_lower.count('the') / 10.0,  # Common word feature
                text_lower.count('a') / 10.0,  # Vowel feature
                len(set(text_lower.split())) / 50.0,  # Unique words feature
                1.0 if any(word in text_lower for word in ['good', 'great', 'excellent']) else 0.0,  # Positive sentiment
                1.0 if any(word in text_lower for word in ['bad', 'poor', 'terrible']) else 0.0,  # Negative sentiment
            ]
            # Pad or truncate to consistent size
            while len(embedding) < 10:
                embedding.append(0.0)
            embeddings.append(embedding[:10])
        return embeddings


def _get_embedding_model():
    """Lazy-load the sentence-transformers model or return mock in CI."""
    global _embedding_model
    if _embedding_model is None:
        if _is_ci_environment():
            logger.info("CI environment detected, using mock embedding model for fast tests")
            _embedding_model = MockEmbeddingModel()
        else:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed; embedding metrics will fall back to token overlap")
    return _embedding_model


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _token_overlap(text_a: str, text_b: str) -> float:
    """Simple token-overlap similarity (Jaccard) as fallback."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    metric_name: str
    score: float
    passed: bool
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """Aggregate quality gate result for an evaluation run."""
    passed: bool
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    overall_score: float = 0.0
    failed_metrics: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Quality Gate: {status} (score={self.overall_score:.2f})"]
        for name, m in self.metrics.items():
            mark = "PASS" if m.passed else "FAIL"
            lines.append(f"  [{mark}] {name}: {m.score:.3f} (threshold {m.threshold:.2f})")
        return "\n".join(lines)


# ── Individual metric implementations ─────────────────────────────────────

class TaskSuccessMetric:
    """Evaluates task success via exact match and semantic similarity."""

    def __init__(self, exact_match_weight: float = 0.3, semantic_weight: float = 0.7):
        self.exact_match_weight = exact_match_weight
        self.semantic_weight = semantic_weight

    def evaluate(
        self,
        generated: str,
        expected: Optional[str],
        threshold: float = 0.8,
    ) -> MetricResult:
        if not expected:
            return MetricResult(
                metric_name="task_success",
                score=1.0,
                passed=True,
                threshold=threshold,
                details={"note": "No expected output; skipped"},
            )

        exact = self._exact_match_score(generated, expected)
        semantic = self._semantic_similarity(generated, expected)
        score = self.exact_match_weight * exact + self.semantic_weight * semantic

        return MetricResult(
            metric_name="task_success",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"exact_match": exact, "semantic_similarity": semantic},
        )

    @staticmethod
    def _exact_match_score(generated: str, expected: str) -> float:
        gen_norm = re.sub(r"\s+", " ", generated.strip().lower())
        exp_norm = re.sub(r"\s+", " ", expected.strip().lower())
        if gen_norm == exp_norm:
            return 1.0
        if exp_norm in gen_norm:
            return 0.8
        return 0.0

    @staticmethod
    def _semantic_similarity(generated: str, expected: str) -> float:
        model = _get_embedding_model()
        if model is not None:
            embeddings = model.encode([generated, expected])
            return _cosine_similarity(embeddings[0], embeddings[1])
        return _token_overlap(generated, expected)


class RelevanceMetric:
    """Evaluates relevance of generated output to the input prompt using embeddings."""

    def evaluate(
        self,
        prompt: str,
        generated: str,
        threshold: float = 0.7,
    ) -> MetricResult:
        model = _get_embedding_model()
        if model is not None:
            embeddings = model.encode([prompt, generated])
            score = _cosine_similarity(embeddings[0], embeddings[1])
        else:
            score = _token_overlap(prompt, generated)

        return MetricResult(
            metric_name="relevance",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"method": "embedding" if model else "token_overlap"},
        )


class HallucinationDetector:
    """Detects hallucination using an LLM-as-Judge approach.

    The judge receives the prompt, expected context / ground truth, and the
    generated output, and returns a binary verdict: Grounded or Hallucinated.
    """

    JUDGE_SYSTEM_PROMPT = (
        "You are an expert fact-checker. Your job is to determine whether an AI-generated "
        "response is grounded in the provided context/expected answer, or whether it contains "
        "hallucinated information (facts not supported by the context).\n\n"
        "Respond with EXACTLY one of:\n"
        "VERDICT: GROUNDED\n"
        "VERDICT: HALLUCINATED\n\n"
        "Then provide a brief explanation."
    )

    JUDGE_USER_TEMPLATE = (
        "## Input Prompt\n{prompt}\n\n"
        "## Expected/Reference Answer\n{expected}\n\n"
        "## AI-Generated Response\n{generated}\n\n"
        "Determine: is the AI-generated response grounded in the reference, "
        "or does it contain hallucinated claims?"
    )

    def __init__(self, judge_llm=None):
        """
        Args:
            judge_llm: A BaseLLM (or compatible) instance used for judging.
                       If None, falls back to token-overlap heuristic.
        """
        self.judge_llm = judge_llm

    def evaluate(
        self,
        prompt: str,
        generated: str,
        expected: Optional[str] = None,
        context: Optional[str] = None,
        threshold: float = 0.1,
    ) -> MetricResult:
        """Evaluate hallucination risk (lower score = less hallucination = better).

        Args:
            threshold: Maximum acceptable hallucination score (default 0.1 = 10%).
        """
        reference = expected or context or ""
        if not reference:
            return MetricResult(
                metric_name="hallucination",
                score=0.0,
                passed=True,
                threshold=threshold,
                details={"note": "No reference provided; cannot assess hallucination"},
            )

        if self.judge_llm is not None:
            return self._judge_based_evaluation(prompt, generated, reference, threshold)
        return self._heuristic_evaluation(generated, reference, threshold)

    def _judge_based_evaluation(
        self, prompt: str, generated: str, reference: str, threshold: float
    ) -> MetricResult:
        user_msg = self.JUDGE_USER_TEMPLATE.format(
            prompt=prompt, expected=reference, generated=generated
        )

        try:
            response = self.judge_llm.generate(
                messages=[
                    {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            verdict_text = response.content.upper()
            is_hallucinated = "HALLUCINATED" in verdict_text
            score = 1.0 if is_hallucinated else 0.0

            return MetricResult(
                metric_name="hallucination",
                score=score,
                passed=score <= threshold,
                threshold=threshold,
                details={
                    "method": "llm_judge",
                    "verdict": "HALLUCINATED" if is_hallucinated else "GROUNDED",
                    "judge_response": response.content[:500],
                },
            )
        except Exception as e:
            logger.error(f"Judge-based hallucination evaluation failed: {e}")
            return self._heuristic_evaluation(generated, reference, threshold)

    def _heuristic_evaluation(
        self, generated: str, reference: str, threshold: float
    ) -> MetricResult:
        """Heuristic fallback: high overlap with reference → low hallucination."""
        model = _get_embedding_model()
        if model is not None:
            emb = model.encode([generated, reference])
            similarity = _cosine_similarity(emb[0], emb[1])
        else:
            similarity = _token_overlap(generated, reference)

        # Invert: high similarity → low hallucination score
        score = max(0.0, 1.0 - similarity)

        return MetricResult(
            metric_name="hallucination",
            score=score,
            passed=score <= threshold,
            threshold=threshold,
            details={"method": "heuristic", "reference_similarity": similarity},
        )


class ConsistencyMetric:
    """Evaluates consistency across multiple runs of the same prompt."""

    def evaluate(
        self,
        outputs: List[str],
        threshold: float = 0.8,
    ) -> MetricResult:
        if len(outputs) < 2:
            return MetricResult(
                metric_name="consistency",
                score=1.0,
                passed=True,
                threshold=threshold,
                details={"note": "Fewer than 2 outputs; consistency trivially 1.0"},
            )

        pairs = []
        model = _get_embedding_model()
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if model is not None:
                    emb = model.encode([outputs[i], outputs[j]])
                    sim = _cosine_similarity(emb[0], emb[1])
                else:
                    sim = _token_overlap(outputs[i], outputs[j])
                pairs.append(sim)

        avg_similarity = sum(pairs) / len(pairs) if pairs else 0.0

        return MetricResult(
            metric_name="consistency",
            score=avg_similarity,
            passed=avg_similarity >= threshold,
            threshold=threshold,
            details={
                "pairwise_similarities": pairs,
                "num_outputs": len(outputs),
                "min_similarity": min(pairs) if pairs else 0.0,
                "max_similarity": max(pairs) if pairs else 0.0,
            },
        )


# ── Quality Gate ──────────────────────────────────────────────────────────

class QualityGate:
    """Enforces pass/fail thresholds across all metrics."""

    DEFAULT_THRESHOLDS = {
        "task_success": 0.8,
        "relevance": 0.7,
        "hallucination": 0.1,
        "consistency": 0.8,
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def evaluate(self, metrics: Dict[str, MetricResult]) -> QualityGateResult:
        failed = []
        for name, result in metrics.items():
            if not result.passed:
                failed.append(name)

        scores = [m.score for m in metrics.values()]
        overall = sum(scores) / len(scores) if scores else 0.0

        return QualityGateResult(
            passed=len(failed) == 0,
            metrics=metrics,
            overall_score=overall,
            failed_metrics=failed,
        )


# ── Metrics Engine (orchestrator) ─────────────────────────────────────────

class MetricsEngine:
    """Orchestrates all metric evaluations for a test case or batch."""

    def __init__(
        self,
        judge_llm=None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.task_success = TaskSuccessMetric()
        self.relevance = RelevanceMetric()
        self.hallucination = HallucinationDetector(judge_llm=judge_llm)
        self.consistency = ConsistencyMetric()
        self.quality_gate = QualityGate(thresholds=thresholds)
        self.thresholds = {**QualityGate.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def evaluate_single(
        self,
        prompt: str,
        generated: str,
        expected: Optional[str] = None,
        context: Optional[str] = None,
        metrics_to_evaluate: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, MetricResult]:
        """Evaluate a single generated output against all requested metrics."""
        thresholds = {**self.thresholds, **(custom_thresholds or {})}
        evaluate_all = metrics_to_evaluate is None or "all" in metrics_to_evaluate

        results: Dict[str, MetricResult] = {}

        if evaluate_all or "task_success" in metrics_to_evaluate:
            results["task_success"] = self.task_success.evaluate(
                generated, expected, threshold=thresholds.get("task_success", 0.8)
            )

        if evaluate_all or "relevance" in metrics_to_evaluate:
            results["relevance"] = self.relevance.evaluate(
                prompt, generated, threshold=thresholds.get("relevance", 0.7)
            )

        if evaluate_all or "hallucination" in metrics_to_evaluate:
            results["hallucination"] = self.hallucination.evaluate(
                prompt, generated, expected=expected, context=context,
                threshold=thresholds.get("hallucination", 0.1),
            )

        return results

    def evaluate_consistency(
        self,
        outputs: List[str],
        threshold: Optional[float] = None,
    ) -> MetricResult:
        """Evaluate consistency across multiple outputs."""
        t = threshold or self.thresholds.get("consistency", 0.8)
        return self.consistency.evaluate(outputs, threshold=t)

    def evaluate_batch(
        self,
        test_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a batch of test case results and compute aggregate metrics.

        Each item in test_results should have:
            prompt, generated, expected (optional), context (optional),
            metrics_to_evaluate (optional), custom_thresholds (optional)
        """
        per_case: List[Dict[str, MetricResult]] = []
        for item in test_results:
            case_metrics = self.evaluate_single(
                prompt=item["prompt"],
                generated=item["generated"],
                expected=item.get("expected"),
                context=item.get("context"),
                metrics_to_evaluate=item.get("metrics_to_evaluate"),
                custom_thresholds=item.get("custom_thresholds"),
            )
            per_case.append(case_metrics)

        aggregated = self._aggregate_metrics(per_case)
        gate_result = self.quality_gate.evaluate(aggregated)

        return {
            "per_case": per_case,
            "aggregated": aggregated,
            "quality_gate": gate_result,
        }

    def _aggregate_metrics(
        self, per_case: List[Dict[str, MetricResult]]
    ) -> Dict[str, MetricResult]:
        """Aggregate per-case metrics into summary metrics."""
        metric_scores: Dict[str, List[float]] = {}
        metric_thresholds: Dict[str, float] = {}

        for case_metrics in per_case:
            for name, result in case_metrics.items():
                metric_scores.setdefault(name, []).append(result.score)
                metric_thresholds[name] = result.threshold

        aggregated: Dict[str, MetricResult] = {}
        for name, scores in metric_scores.items():
            avg_score = sum(scores) / len(scores)
            threshold = metric_thresholds[name]

            # For hallucination, pass means avg score <= threshold
            if name == "hallucination":
                passed = avg_score <= threshold
            else:
                passed = avg_score >= threshold

            aggregated[name] = MetricResult(
                metric_name=name,
                score=avg_score,
                passed=passed,
                threshold=threshold,
                details={
                    "count": len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "scores": scores,
                },
            )

        return aggregated
