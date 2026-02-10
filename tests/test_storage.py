"""Tests for the storage layer."""

import pytest
import uuid
import os
from datetime import datetime

from storage.database import Database
from storage.models import EvaluationRun, TestCaseRecord, QualityGateRecord
from storage.repository import EvaluationRepository


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.duckdb")
    database = Database(db_path)
    yield database
    database.close()


@pytest.fixture
def repo(db):
    return EvaluationRepository(db)


class TestDatabase:
    def test_schema_creation(self, db):
        tables = db.fetchall("SHOW TABLES")
        table_names = {t["name"] for t in tables}
        assert "evaluation_runs" in table_names
        assert "test_case_results" in table_names
        assert "quality_gate_history" in table_names
        assert "provider_comparisons" in table_names


class TestEvaluationRepository:
    def test_save_and_get_run(self, repo):
        run = EvaluationRun(
            id=str(uuid.uuid4()),
            dataset_name="test-dataset",
            dataset_version="1.0",
            provider_name="groq",
            model_name="llama3-8b",
            total_test_cases=10,
            successful_executions=8,
            failed_executions=2,
            total_execution_time=5.5,
            success_rate=0.8,
            quality_gate_passed=True,
            overall_score=0.85,
            task_success_score=0.9,
            relevance_score=0.8,
            hallucination_score=0.05,
        )
        repo.save_run(run)

        fetched = repo.get_run(run.id)
        assert fetched is not None
        assert fetched["provider_name"] == "groq"
        assert fetched["success_rate"] == 0.8

    def test_list_runs(self, repo):
        for i in range(3):
            run = EvaluationRun(
                id=str(uuid.uuid4()),
                dataset_name="ds",
                dataset_version="1.0",
                provider_name="groq" if i < 2 else "openai",
                model_name="model",
                total_test_cases=5,
                successful_executions=5,
                failed_executions=0,
                total_execution_time=1.0,
                success_rate=1.0,
            )
            repo.save_run(run)

        all_runs = repo.list_runs()
        assert len(all_runs) == 3

        groq_runs = repo.list_runs(provider="groq")
        assert len(groq_runs) == 2

    def test_save_test_case_results(self, repo):
        run_id = str(uuid.uuid4())
        run = EvaluationRun(
            id=run_id, dataset_name="ds", dataset_version="1.0",
            provider_name="groq", model_name="m",
            total_test_cases=1, successful_executions=1, failed_executions=0,
            total_execution_time=1.0, success_rate=1.0,
        )
        repo.save_run(run)

        records = [TestCaseRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            test_case_id="tc-1",
            input_prompt="test prompt",
            generated_output="test output",
            task_type="question_answering",
            success=True,
            passed=True,
            task_success_score=0.95,
        )]
        count = repo.save_test_case_results(records)
        assert count == 1

        results = repo.get_test_case_results(run_id)
        assert len(results) == 1
        assert results[0]["test_case_id"] == "tc-1"

    def test_quality_gate_history(self, repo):
        run_id = str(uuid.uuid4())
        run = EvaluationRun(
            id=run_id, dataset_name="ds", dataset_version="1.0",
            provider_name="groq", model_name="m",
            total_test_cases=1, successful_executions=1, failed_executions=0,
            total_execution_time=1.0, success_rate=1.0,
        )
        repo.save_run(run)

        gate = QualityGateRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            passed=True,
            overall_score=0.9,
            failed_metrics=[],
        )
        repo.save_quality_gate(gate)

        history = repo.get_quality_gate_history()
        assert len(history) >= 1
        assert history[0]["passed"] is True
