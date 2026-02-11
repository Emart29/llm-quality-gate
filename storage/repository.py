"""Repository pattern for CRUD operations on evaluation data."""

import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .database import Database
from .models import EvaluationRun, TestCaseRecord, QualityGateRecord, ProviderComparison, BaselineMetric

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """CRUD operations for evaluation results."""

    def __init__(self, db: Database):
        self.db = db

    # ── Evaluation Runs ───────────────────────────────────────────────

    def save_run(self, run: EvaluationRun) -> str:
        """Insert an evaluation run. Returns the run id."""
        self.db.execute(
            """INSERT INTO evaluation_runs (
                id, dataset_name, dataset_version, provider_name, model_name,
                commit_hash, branch, trigger,
                total_test_cases, successful_executions, failed_executions,
                total_execution_time, success_rate, pass_rate,
                quality_gate_passed, overall_score,
                task_success_score, relevance_score, hallucination_score, consistency_score, regression_detected, regression_summary,
                configuration, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                run.id, run.dataset_name, run.dataset_version,
                run.provider_name, run.model_name,
                run.commit_hash, run.branch, run.trigger,
                run.total_test_cases, run.successful_executions, run.failed_executions,
                run.total_execution_time, run.success_rate, run.pass_rate,
                run.quality_gate_passed, run.overall_score,
                run.task_success_score, run.relevance_score,
                run.hallucination_score, run.consistency_score,
                run.regression_detected, json.dumps(run.regression_summary),
                json.dumps(run.configuration), run.created_at,
            ],
        )
        return run.id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.db.fetchone("SELECT * FROM evaluation_runs WHERE id = ?", [run_id])

    def list_runs(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dataset_version: Optional[str] = None,
        commit_hash: Optional[str] = None,
        quality_gate_passed: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM evaluation_runs WHERE 1=1"
        params: list = []
        if provider:
            query += " AND provider_name = ?"
            params.append(provider)
        if model:
            query += " AND model_name = ?"
            params.append(model)
        if dataset_version:
            query += " AND dataset_version = ?"
            params.append(dataset_version)
        if commit_hash:
            query += " AND commit_hash = ?"
            params.append(commit_hash)
        if quality_gate_passed is not None:
            query += " AND quality_gate_passed = ?"
            params.append(quality_gate_passed)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self.db.fetchall(query, params)

    def list_run_filters(self) -> Dict[str, List[Any]]:
        """Get distinct values for dashboard filters."""
        providers = self.db.fetchall(
            "SELECT DISTINCT provider_name FROM evaluation_runs ORDER BY provider_name"
        )
        models = self.db.fetchall(
            "SELECT DISTINCT model_name FROM evaluation_runs ORDER BY model_name"
        )
        dataset_versions = self.db.fetchall(
            "SELECT DISTINCT dataset_version FROM evaluation_runs ORDER BY dataset_version"
        )
        commits = self.db.fetchall(
            "SELECT DISTINCT commit_hash FROM evaluation_runs WHERE commit_hash IS NOT NULL ORDER BY created_at DESC LIMIT 100"
        )

        return {
            "providers": [row["provider_name"] for row in providers if row.get("provider_name")],
            "models": [row["model_name"] for row in models if row.get("model_name")],
            "dataset_versions": [row["dataset_version"] for row in dataset_versions if row.get("dataset_version")],
            "commit_hashes": [row["commit_hash"] for row in commits if row.get("commit_hash")],
        }

    def get_run_history(self, provider: str, model: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Get historical runs for trend analysis."""
        return self.db.fetchall(
            """SELECT id, created_at, success_rate, pass_rate, overall_score,
                      task_success_score, relevance_score, hallucination_score, consistency_score,
                      quality_gate_passed, commit_hash
               FROM evaluation_runs
               WHERE provider_name = ? AND model_name = ?
               ORDER BY created_at DESC LIMIT ?""",
            [provider, model, limit],
        )

    # ── Baseline Metrics ───────────────────────────────────────────────

    def get_baseline(self, provider: str, model: str, dataset_version: str) -> Optional[Dict[str, Any]]:
        return self.db.fetchone(
            """SELECT * FROM baseline_metrics
               WHERE provider_name = ? AND model_name = ? AND dataset_version = ?""",
            [provider, model, dataset_version],
        )

    def upsert_baseline(self, baseline: BaselineMetric) -> str:
        existing = self.get_baseline(
            baseline.provider_name,
            baseline.model_name,
            baseline.dataset_version,
        )
        if existing:
            self.db.execute(
                """UPDATE baseline_metrics
                   SET overall_score = ?, task_success_score = ?, relevance_score = ?,
                       hallucination_score = ?, consistency_score = ?, source_run_id = ?,
                       commit_hash = ?, updated_at = ?
                   WHERE id = ?""",
                [
                    baseline.overall_score,
                    baseline.task_success_score,
                    baseline.relevance_score,
                    baseline.hallucination_score,
                    baseline.consistency_score,
                    baseline.source_run_id,
                    baseline.commit_hash,
                    datetime.utcnow(),
                    existing["id"],
                ],
            )
            return existing["id"]

        self.db.execute(
            """INSERT INTO baseline_metrics (
                id, provider_name, model_name, dataset_version,
                overall_score, task_success_score, relevance_score,
                hallucination_score, consistency_score, source_run_id,
                commit_hash, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                baseline.id,
                baseline.provider_name,
                baseline.model_name,
                baseline.dataset_version,
                baseline.overall_score,
                baseline.task_success_score,
                baseline.relevance_score,
                baseline.hallucination_score,
                baseline.consistency_score,
                baseline.source_run_id,
                baseline.commit_hash,
                baseline.created_at,
                baseline.created_at,
            ],
        )
        return baseline.id

    # ── Test Case Results ─────────────────────────────────────────────

    def save_test_case_results(self, records: List[TestCaseRecord]) -> int:
        """Bulk insert test case results. Returns count inserted."""
        for r in records:
            self.db.execute(
                """INSERT INTO test_case_results (
                    id, run_id, test_case_id, input_prompt, expected_output,
                    generated_output, task_type, difficulty, execution_time,
                    success, passed, error,
                    task_success_score, relevance_score, hallucination_score, consistency_score,
                    tags, metadata, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    r.id, r.run_id, r.test_case_id, r.input_prompt, r.expected_output,
                    r.generated_output, r.task_type, r.difficulty, r.execution_time,
                    r.success, r.passed, r.error,
                    r.task_success_score, r.relevance_score,
                    r.hallucination_score, r.consistency_score,
                    json.dumps(r.tags), json.dumps(r.metadata), r.created_at,
                ],
            )
        return len(records)

    def get_test_case_results(self, run_id: str) -> List[Dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM test_case_results WHERE run_id = ? ORDER BY created_at", [run_id]
        )

    # ── Quality Gate History ──────────────────────────────────────────

    def save_quality_gate(self, record: QualityGateRecord) -> str:
        self.db.execute(
            """INSERT INTO quality_gate_history (
                id, run_id, passed, overall_score, failed_metrics, thresholds, commit_hash, created_at
            ) VALUES (?,?,?,?,?,?,?,?)""",
            [
                record.id, record.run_id, record.passed, record.overall_score,
                json.dumps(record.failed_metrics), json.dumps(record.thresholds),
                record.commit_hash, record.created_at,
            ],
        )
        return record.id

    def get_quality_gate_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM quality_gate_history ORDER BY created_at DESC LIMIT ?", [limit]
        )

    # ── Provider Comparisons ──────────────────────────────────────────

    def save_comparison(self, comparison: ProviderComparison) -> str:
        self.db.execute(
            """INSERT INTO provider_comparisons (
                id, comparison_name, dataset_name, dataset_version, providers, results, winner, created_at
            ) VALUES (?,?,?,?,?,?,?,?)""",
            [
                comparison.id, comparison.comparison_name,
                comparison.dataset_name, comparison.dataset_version,
                json.dumps(comparison.providers), json.dumps(comparison.results),
                comparison.winner, comparison.created_at,
            ],
        )
        return comparison.id

    def get_comparisons(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM provider_comparisons ORDER BY created_at DESC LIMIT ?", [limit]
        )

    # ── Utilities ─────────────────────────────────────────────────────

    def save_comprehensive_result(
        self,
        result_dict: Dict[str, Any],
        commit_hash: Optional[str] = None,
        branch: Optional[str] = None,
        regression: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience: persist a ComprehensiveEvaluationResult.to_dict() in one call."""
        run_id = str(uuid.uuid4())

        agg = result_dict.get("aggregated_metrics", {})
        gate = result_dict.get("quality_gate", {})

        run = EvaluationRun(
            id=run_id,
            dataset_name=result_dict["dataset_name"],
            dataset_version=result_dict["dataset_version"],
            provider_name=result_dict["provider_name"],
            model_name=result_dict["model_name"],
            total_test_cases=result_dict["total_test_cases"],
            successful_executions=result_dict["successful_executions"],
            failed_executions=result_dict["failed_executions"],
            total_execution_time=result_dict["total_execution_time"],
            success_rate=result_dict.get("successful_executions", 0) / max(result_dict.get("total_test_cases", 1), 1),
            pass_rate=result_dict.get("pass_rate"),
            quality_gate_passed=gate.get("passed"),
            overall_score=gate.get("overall_score"),
            task_success_score=agg.get("task_success", {}).get("score"),
            relevance_score=agg.get("relevance", {}).get("score"),
            hallucination_score=agg.get("hallucination", {}).get("score"),
            consistency_score=agg.get("consistency", {}).get("score"),
            regression_detected=(regression or {}).get("regression_detected", False),
            regression_summary=regression or {},
            commit_hash=commit_hash,
            branch=branch,
            configuration=result_dict.get("configuration", {}),
        )
        self.save_run(run)

        # Save test case records
        records = []
        for sr in result_dict.get("scored_results", []):
            metrics = sr.get("metrics", {})
            records.append(TestCaseRecord(
                id=str(uuid.uuid4()),
                run_id=run_id,
                test_case_id=sr["test_case_id"],
                input_prompt=sr.get("input_prompt", ""),
                expected_output=sr.get("expected_output"),
                generated_output=sr.get("generated_output", ""),
                task_type=sr.get("task_type", ""),
                difficulty=sr.get("difficulty", ""),
                execution_time=sr.get("execution_time", 0.0),
                success=sr.get("success", False),
                passed=sr.get("passed"),
                error=sr.get("error"),
                task_success_score=metrics.get("task_success", {}).get("score"),
                relevance_score=metrics.get("relevance", {}).get("score"),
                hallucination_score=metrics.get("hallucination", {}).get("score"),
                consistency_score=metrics.get("consistency", {}).get("score"),
                tags=sr.get("tags", []),
                metadata=sr.get("metadata", {}),
            ))
        if records:
            self.save_test_case_results(records)

        # Save quality gate
        self.save_quality_gate(QualityGateRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            passed=gate.get("passed", False),
            overall_score=gate.get("overall_score"),
            failed_metrics=gate.get("failed_metrics", []),
            commit_hash=commit_hash,
        ))

        return run_id

    def export_run(self, run_id: str) -> Dict[str, Any]:
        """Export a full run with all details for portability."""
        run = self.get_run(run_id)
        test_cases = self.get_test_case_results(run_id)
        return {"run": run, "test_case_results": test_cases}
