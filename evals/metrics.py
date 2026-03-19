"""Metrics engine for LLM evaluation: task success, relevance, hallucination, consistency."""

import logging
import re
import os
import random
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


def _should_use_lightweight_mode() -> bool:
    """Check if lightweight mode should be used (CI environment or config flag)."""
    return _is_ci_environment() or os.environ.get('LLMQ_LIGHTWEIGHT_MODE', '').lower() in ('true', '1', 'yes')


class MockEmbeddingModel:
    """Mock embedding model for CI environments to avoid network calls and model downloads.
    
    Returns deterministic embeddings based on text content for consistent test results.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with a seed for deterministic results."""
        self.seed = seed
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return deterministic mock embeddings based on text content."""
        embeddings = []
        for i, text in enumerate(texts):
            # Create deterministic embedding using text hash and position
            text_hash = hash(text + str(self.seed)) % (2**31)
            random.seed(text_hash + i)
            
            # Generate features based on text characteristics
            text_lower = text.lower()
            words = text_lower.split()
            
            # More sophisticated features for better similarity
            base_features = [
                len(text) / 100.0,  # Length feature
                len(words) / 20.0,  # Word count
                len(set(text_lower)) / 50.0,  # Character diversity
                len(set(words)) / 30.0,  # Unique word count
                text_lower.count('the') / max(len(words), 1),  # Common word density
                sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
                sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
                1.0 if any(word in text_lower for word in ['good', 'great', 'excellent', 'correct', 'right', 'yes']) else 0.0,
                1.0 if any(word in text_lower for word in ['bad', 'poor', 'terrible', 'wrong', 'incorrect', 'no']) else 0.0,
                # Add word overlap features for better similarity
                len(words) / 100.0 if words else 0.0,
            ]
            
            # Add deterministic "semantic" components based on word content
            word_hash_sum = sum(hash(word) % 1000 for word in words) / max(len(words) * 1000, 1)
            semantic_features = [
                word_hash_sum,
                (word_hash_sum * 2) % 1.0,
                (word_hash_sum * 3) % 1.0,
            ]
            
            embedding = base_features + semantic_features
            
            # Normalize to unit vector for cosine similarity
            norm = sum(x*x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            else:
                # Fallback for zero vectors
                embedding = [1.0] + [0.0] * (len(embedding) - 1)
                norm = 1.0
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings


def _get_embedding_model():
    """Lazy-load the sentence-transformers model or return mock in CI."""
    global _embedding_model
    if _embedding_model is None:
        if _should_use_lightweight_mode():
            logger.info("Lightweight mode detected, using mock embedding model for fast execution")
            _embedding_model = MockEmbeddingModel()
        else:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Using real sentence-transformers model for embeddings")
            except ImportError:
                logger.warning("sentence-transformers not installed; embedding metrics will fall back to token overlap")
    return _embedding_model


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        import numpy as np
        a, b = np.asarray(a), np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except ImportError:
        # Pure-Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


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
    skipped: bool = False


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
    """Evaluates task success via LLM judge (preferred) or semantic similarity fallback."""

    JUDGE_SYSTEM_PROMPT = (
        "You are an expert evaluator. Your job is to score how well an AI-generated response "
        "completes the task described in the input prompt, compared to the expected answer.\n\n"
        "Respond with EXACTLY this format:\n"
        "SCORE: <float between 0.0 and 1.0>\n"
        "REASON: <brief explanation>\n\n"
        "Scoring guide:\n"
        "1.0 = Perfect: fully correct, complete, matches expected answer\n"
        "0.8 = Good: mostly correct with minor differences\n"
        "0.5 = Partial: some correct elements but missing key parts\n"
        "0.2 = Poor: mostly wrong but shows some understanding\n"
        "0.0 = Wrong: completely incorrect or irrelevant"
    )

    JUDGE_USER_TEMPLATE = (
        "## Input Prompt\n{prompt}\n\n"
        "## Expected Answer\n{expected}\n\n"
        "## AI-Generated Response\n{generated}\n\n"
        "Score how well the generated response completes the task."
    )

    def __init__(self, judge_llm=None):
        self.judge_llm = judge_llm

    def evaluate(self, prompt: str, generated: str, expected: Optional[str], threshold: float = 0.7) -> MetricResult:
        if not expected:
            return MetricResult(
                metric_name="task_success",
                score=1.0,
                passed=True,
                threshold=threshold,
                details={"note": "No expected output; skipped"},
                skipped=True,
            )

        if self.judge_llm is not None:
            return self._judge_based_evaluation(prompt, generated, expected, threshold)
        return self._semantic_evaluation(generated, expected, threshold)

    def _judge_based_evaluation(self, prompt: str, generated: str, expected: str, threshold: float) -> MetricResult:
        import asyncio
        user_msg = self.JUDGE_USER_TEMPLATE.format(prompt=prompt, expected=expected, generated=generated)
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    response = pool.submit(
                        asyncio.run,
                        self.judge_llm.generate(
                            messages=[
                                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=0.0,
                            max_tokens=200,
                        )
                    ).result()
            else:
                response = asyncio.run(
                    self.judge_llm.generate(
                        messages=[
                            {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=200,
                    )
                )

            text = response.content
            import re as _re
            match = _re.search(r'SCORE:\s*([\d.]+)', text)
            score = float(match.group(1)) if match else 0.5
            score = max(0.0, min(1.0, score))

            return MetricResult(
                metric_name="task_success",
                score=score,
                passed=score >= threshold,
                threshold=threshold,
                details={"method": "llm_judge", "judge_response": text[:300]},
            )
        except Exception as e:
            logger.error(f"Judge-based task success evaluation failed: {e}")
            return self._semantic_evaluation(generated, expected, threshold)

    @staticmethod
    def _semantic_evaluation(generated: str, expected: str, threshold: float) -> MetricResult:
        model = _get_embedding_model()
        if model is not None:
            embeddings = model.encode([generated, expected])
            score = _cosine_similarity(embeddings[0], embeddings[1])
        else:
            score = _token_overlap(generated, expected)

        return MetricResult(
            metric_name="task_success",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"method": "semantic_similarity"},
        )


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
        import asyncio
        user_msg = self.JUDGE_USER_TEMPLATE.format(
            prompt=prompt, expected=reference, generated=generated
        )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    response = pool.submit(
                        asyncio.run,
                        self.judge_llm.generate(
                            messages=[
                                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=0.0,
                            max_tokens=300,
                        )
                    ).result()
            else:
                response = asyncio.run(
                    self.judge_llm.generate(
                        messages=[
                            {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=300,
                    )
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
        deterministic: bool = False,
    ) -> MetricResult:
        if deterministic:
            return MetricResult(
                metric_name="consistency",
                score=1.0,
                passed=True,
                threshold=threshold,
                details={"note": "Skipped: temperature=0 (deterministic mode produces identical outputs)"},
                skipped=True,
            )
        if len(outputs) < 2:
            return MetricResult(
                metric_name="consistency",
                score=1.0,
                passed=True,
                threshold=threshold,
                details={"note": "Fewer than 2 outputs; consistency trivially 1.0"},
            )

        # In lightweight mode, skip expensive consistency computation
        if _should_use_lightweight_mode():
            # Return a deterministic score based on output similarity
            avg_length = sum(len(output) for output in outputs) / len(outputs)
            length_variance = sum((len(output) - avg_length) ** 2 for output in outputs) / len(outputs)
            # Lower variance = higher consistency
            score = max(0.0, 1.0 - (length_variance / (avg_length + 1)))
            
            return MetricResult(
                metric_name="consistency",
                score=score,
                passed=score >= threshold,
                threshold=threshold,
                details={
                    "method": "lightweight_heuristic",
                    "num_outputs": len(outputs),
                    "avg_length": avg_length,
                    "length_variance": length_variance,
                    "note": "Lightweight mode: using length-based heuristic"
                },
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
            if not result.skipped and not result.passed:
                failed.append(name)

        scores = [m.score for m in metrics.values() if not m.skipped]
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
        self.task_success = TaskSuccessMetric(judge_llm=judge_llm)
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
                prompt, generated, expected, threshold=thresholds.get("task_success", 0.7)
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
