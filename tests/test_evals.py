"""Tests for the evaluation framework."""

import pytest
from evals.dataset import (
    EvaluationDataset, TestCase, DatasetLoader, TaskType,
    DifficultyLevel, EvaluationMetric,
)
from evals.validator import DatasetValidator


class TestDataset:
    def test_create_dataset(self):
        dataset = EvaluationDataset(
            name="Test",
            description="Test dataset",
            version="1.0",
        )
        assert len(dataset.test_cases) == 0

    def test_add_test_case(self):
        dataset = EvaluationDataset(name="Test", description="d", version="1.0")
        tc = TestCase(
            input_prompt="What is 2+2?",
            expected_output="4",
            task_type=TaskType.MATH,
        )
        dataset.add_test_case(tc)
        assert len(dataset.test_cases) == 1

    def test_filter_by_task_type(self):
        dataset = EvaluationDataset(name="Test", description="d", version="1.0")
        dataset.add_test_case(TestCase(
            input_prompt="What is 2+2?", task_type=TaskType.MATH,
        ))
        dataset.add_test_case(TestCase(
            input_prompt="Capital of France?", task_type=TaskType.QUESTION_ANSWERING,
        ))

        math_cases = dataset.filter_by_task_type(TaskType.MATH)
        assert len(math_cases) == 1

    def test_sample_dataset(self):
        dataset = DatasetLoader.create_sample_dataset()
        assert len(dataset.test_cases) > 0
        assert dataset.name is not None


class TestDatasetLoader:
    def test_save_and_load(self, tmp_path):
        dataset = DatasetLoader.create_sample_dataset()
        path = str(tmp_path / "test_dataset.json")
        DatasetLoader.save_to_file(dataset, path)

        loaded = DatasetLoader.load_from_file(path)
        assert loaded.name == dataset.name
        assert len(loaded.test_cases) == len(dataset.test_cases)


class TestValidator:
    def test_valid_dataset(self):
        dataset = DatasetLoader.create_sample_dataset()
        validator = DatasetValidator()
        result = validator.validate_dataset(dataset)
        assert result.is_valid

    def test_empty_dataset(self):
        dataset = EvaluationDataset(name="Empty", description="empty", version="1.0")
        validator = DatasetValidator()
        result = validator.validate_dataset(dataset)
        assert not result.is_valid
        assert any("at least one" in e for e in result.errors)

    def test_validate_single_test_case(self):
        validator = DatasetValidator()
        tc = TestCase(
            input_prompt="What is the meaning of life?",
            task_type=TaskType.QUESTION_ANSWERING,
        )
        result = validator.validate_test_case(tc)
        assert result.is_valid
