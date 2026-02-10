"""Dataset validation utilities."""

from typing import List, Dict, Any, Tuple, Set
from .dataset import EvaluationDataset, TestCase, TaskType, DifficultyLevel, EvaluationMetric
import re
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for dataset validation errors."""
    pass


class ValidationResult:
    """Result of dataset validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid: bool = True
    
    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)
    
    def __str__(self) -> str:
        """String representation of validation result."""
        result = f"Validation {'PASSED' if self.is_valid else 'FAILED'}\n"
        
        if self.errors:
            result += f"\nErrors ({len(self.errors)}):\n"
            for i, error in enumerate(self.errors, 1):
                result += f"  {i}. {error}\n"
        
        if self.warnings:
            result += f"\nWarnings ({len(self.warnings)}):\n"
            for i, warning in enumerate(self.warnings, 1):
                result += f"  {i}. {warning}\n"
        
        return result


class DatasetValidator:
    """Validator for evaluation datasets and test cases."""
    
    def __init__(self):
        self.min_prompt_length = 10
        self.max_prompt_length = 10000
        self.min_expected_output_length = 1
        self.max_expected_output_length = 5000
        self.required_task_types = {TaskType.QUESTION_ANSWERING, TaskType.REASONING}
        self.min_test_cases_per_difficulty = 2
    
    def validate_dataset(self, dataset: EvaluationDataset) -> ValidationResult:
        """
        Validate an entire evaluation dataset.
        
        Args:
            dataset: The dataset to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        # Basic dataset validation
        self._validate_dataset_metadata(dataset, result)
        self._validate_dataset_structure(dataset, result)
        self._validate_dataset_distribution(dataset, result)
        
        # Validate individual test cases
        for test_case in dataset.test_cases:
            self._validate_test_case(test_case, result)
        
        # Cross-test case validation
        self._validate_test_case_uniqueness(dataset, result)
        self._validate_dataset_coverage(dataset, result)
        
        return result
    
    def validate_test_case(self, test_case: TestCase) -> ValidationResult:
        """
        Validate a single test case.
        
        Args:
            test_case: The test case to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        self._validate_test_case(test_case, result)
        return result
    
    def _validate_dataset_metadata(self, dataset: EvaluationDataset, result: ValidationResult) -> None:
        """Validate dataset metadata."""
        if not dataset.name or not dataset.name.strip():
            result.add_error("Dataset name is required and cannot be empty")
        
        if not dataset.description or not dataset.description.strip():
            result.add_error("Dataset description is required and cannot be empty")
        
        if not dataset.version or not dataset.version.strip():
            result.add_error("Dataset version is required and cannot be empty")
        
        # Validate version format (semantic versioning)
        if dataset.version and not re.match(r'^\d+\.\d+(\.\d+)?$', dataset.version):
            result.add_warning("Dataset version should follow semantic versioning (e.g., 1.0.0)")
    
    def _validate_dataset_structure(self, dataset: EvaluationDataset, result: ValidationResult) -> None:
        """Validate dataset structure."""
        if not dataset.test_cases:
            result.add_error("Dataset must contain at least one test case")
            return
        
        if len(dataset.test_cases) < 5:
            result.add_warning("Dataset has fewer than 5 test cases, consider adding more for robust evaluation")
        
        # Validate global thresholds
        required_thresholds = {
            "task_success_threshold",
            "relevance_threshold", 
            "hallucination_threshold",
            "consistency_threshold"
        }
        
        missing_thresholds = required_thresholds - set(dataset.global_thresholds.keys())
        if missing_thresholds:
            result.add_error(f"Missing required global thresholds: {missing_thresholds}")
        
        # Validate threshold values
        for threshold_name, threshold_value in dataset.global_thresholds.items():
            if not isinstance(threshold_value, (int, float)):
                result.add_error(f"Threshold '{threshold_name}' must be a number")
            elif not 0.0 <= threshold_value <= 1.0:
                result.add_error(f"Threshold '{threshold_name}' must be between 0.0 and 1.0")
    
    def _validate_dataset_distribution(self, dataset: EvaluationDataset, result: ValidationResult) -> None:
        """Validate dataset distribution across task types and difficulties."""
        if not dataset.test_cases:
            return
        
        # Check task type distribution
        task_type_counts = {}
        difficulty_counts = {}
        
        for test_case in dataset.test_cases:
            task_type_counts[test_case.task_type] = task_type_counts.get(test_case.task_type, 0) + 1
            difficulty_counts[test_case.difficulty] = difficulty_counts.get(test_case.difficulty, 0) + 1
        
        # Check for required task types
        missing_required_types = self.required_task_types - set(task_type_counts.keys())
        if missing_required_types:
            result.add_warning(f"Dataset missing recommended task types: {[t.value for t in missing_required_types]}")
        
        # Check difficulty distribution
        for difficulty in DifficultyLevel:
            count = difficulty_counts.get(difficulty, 0)
            if count < self.min_test_cases_per_difficulty:
                result.add_warning(f"Only {count} test cases for difficulty '{difficulty.value}', recommend at least {self.min_test_cases_per_difficulty}")
        
        # Check for balanced distribution
        total_cases = len(dataset.test_cases)
        for task_type, count in task_type_counts.items():
            percentage = (count / total_cases) * 100
            if percentage > 50:
                result.add_warning(f"Task type '{task_type.value}' represents {percentage:.1f}% of dataset, consider more balance")
    
    def _validate_test_case(self, test_case: TestCase, result: ValidationResult) -> None:
        """Validate a single test case."""
        # Validate required fields
        if not test_case.input_prompt or not test_case.input_prompt.strip():
            result.add_error(f"Test case {test_case.id}: input_prompt is required and cannot be empty")
        
        # Validate prompt length
        if test_case.input_prompt:
            prompt_length = len(test_case.input_prompt)
            if prompt_length < self.min_prompt_length:
                result.add_warning(f"Test case {test_case.id}: input_prompt is very short ({prompt_length} chars)")
            elif prompt_length > self.max_prompt_length:
                result.add_error(f"Test case {test_case.id}: input_prompt is too long ({prompt_length} chars, max {self.max_prompt_length})")
        
        # Validate expected output if present
        if test_case.expected_output:
            output_length = len(test_case.expected_output)
            if output_length < self.min_expected_output_length:
                result.add_warning(f"Test case {test_case.id}: expected_output is very short")
            elif output_length > self.max_expected_output_length:
                result.add_warning(f"Test case {test_case.id}: expected_output is very long ({output_length} chars)")
        
        # Validate system prompt if present
        if test_case.system_prompt and len(test_case.system_prompt.strip()) == 0:
            result.add_warning(f"Test case {test_case.id}: system_prompt is empty, consider removing it")
        
        # Validate tags
        if not test_case.tags:
            result.add_warning(f"Test case {test_case.id}: no tags specified, consider adding tags for better organization")
        
        for tag in test_case.tags:
            if not re.match(r'^[a-z0-9_]+$', tag):
                result.add_error(f"Test case {test_case.id}: tag '{tag}' should only contain lowercase letters, numbers, and underscores")
        
        # Validate metrics to evaluate
        if not test_case.metrics_to_evaluate:
            result.add_error(f"Test case {test_case.id}: must specify at least one metric to evaluate")
        
        # Validate custom thresholds if present
        if test_case.custom_thresholds:
            for threshold_name, threshold_value in test_case.custom_thresholds.items():
                if not isinstance(threshold_value, (int, float)):
                    result.add_error(f"Test case {test_case.id}: custom threshold '{threshold_name}' must be a number")
                elif not 0.0 <= threshold_value <= 1.0:
                    result.add_error(f"Test case {test_case.id}: custom threshold '{threshold_name}' must be between 0.0 and 1.0")
        
        # Validate constraints
        for constraint in test_case.constraints:
            if not constraint.strip():
                result.add_warning(f"Test case {test_case.id}: empty constraint found")
        
        # Task-specific validation
        self._validate_task_specific_requirements(test_case, result)
    
    def _validate_task_specific_requirements(self, test_case: TestCase, result: ValidationResult) -> None:
        """Validate task-specific requirements."""
        task_type = test_case.task_type
        
        # Question answering should have expected output for exact match evaluation
        if task_type == TaskType.QUESTION_ANSWERING:
            if not test_case.expected_output and EvaluationMetric.TASK_SUCCESS in test_case.metrics_to_evaluate:
                result.add_warning(f"Test case {test_case.id}: Question answering task should have expected_output for task success evaluation")
        
        # Math tasks should have expected output
        if task_type == TaskType.MATH:
            if not test_case.expected_output:
                result.add_warning(f"Test case {test_case.id}: Math task should have expected_output")
        
        # Hallucination detection tests should have specific patterns
        if "hallucination_test" in test_case.tags:
            if EvaluationMetric.HALLUCINATION not in test_case.metrics_to_evaluate:
                result.add_warning(f"Test case {test_case.id}: Hallucination test should evaluate hallucination metric")
        
        # Consistency tests should evaluate consistency
        if "consistency_test" in test_case.tags:
            if EvaluationMetric.CONSISTENCY not in test_case.metrics_to_evaluate:
                result.add_warning(f"Test case {test_case.id}: Consistency test should evaluate consistency metric")
        
        # Creative writing tasks should have constraints
        if task_type == TaskType.CREATIVE_WRITING:
            if not test_case.constraints:
                result.add_warning(f"Test case {test_case.id}: Creative writing task should have constraints")
    
    def _validate_test_case_uniqueness(self, dataset: EvaluationDataset, result: ValidationResult) -> None:
        """Validate that test cases are unique."""
        seen_prompts = set()
        seen_ids = set()
        
        for test_case in dataset.test_cases:
            # Check for duplicate IDs
            if test_case.id in seen_ids:
                result.add_error(f"Duplicate test case ID found: {test_case.id}")
            seen_ids.add(test_case.id)
            
            # Check for duplicate prompts (case-insensitive)
            prompt_key = test_case.input_prompt.lower().strip()
            if prompt_key in seen_prompts:
                result.add_warning(f"Test case {test_case.id}: Similar input prompt already exists in dataset")
            seen_prompts.add(prompt_key)
    
    def _validate_dataset_coverage(self, dataset: EvaluationDataset, result: ValidationResult) -> None:
        """Validate dataset coverage across different dimensions."""
        if not dataset.test_cases:
            return
        
        # Check metric coverage
        all_metrics = set()
        for test_case in dataset.test_cases:
            all_metrics.update(test_case.metrics_to_evaluate)
        
        recommended_metrics = {
            EvaluationMetric.TASK_SUCCESS,
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.HALLUCINATION
        }
        
        missing_metrics = recommended_metrics - all_metrics
        if missing_metrics:
            result.add_warning(f"Dataset doesn't cover recommended metrics: {[m.value for m in missing_metrics]}")
        
        # Check for test cases with expected outputs
        cases_with_expected = sum(1 for tc in dataset.test_cases if tc.expected_output)
        if cases_with_expected == 0:
            result.add_warning("No test cases have expected outputs, limiting evaluation capabilities")
        elif cases_with_expected < len(dataset.test_cases) * 0.3:
            result.add_warning("Less than 30% of test cases have expected outputs")
        
        # Check for system prompts usage
        cases_with_system_prompt = sum(1 for tc in dataset.test_cases if tc.system_prompt)
        if cases_with_system_prompt == 0:
            result.add_warning("No test cases use system prompts, consider adding some for comprehensive testing")
    
    def suggest_improvements(self, dataset: EvaluationDataset) -> List[str]:
        """Suggest improvements for the dataset."""
        suggestions = []
        
        if not dataset.test_cases:
            return ["Add test cases to the dataset"]
        
        # Analyze current distribution
        task_type_counts = {}
        difficulty_counts = {}
        tag_counts = {}
        
        for test_case in dataset.test_cases:
            task_type_counts[test_case.task_type] = task_type_counts.get(test_case.task_type, 0) + 1
            difficulty_counts[test_case.difficulty] = difficulty_counts.get(test_case.difficulty, 0) + 1
            for tag in test_case.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Suggest missing task types
        all_task_types = set(TaskType)
        missing_types = all_task_types - set(task_type_counts.keys())
        if missing_types:
            suggestions.append(f"Consider adding test cases for: {[t.value for t in missing_types]}")
        
        # Suggest balancing difficulties
        total_cases = len(dataset.test_cases)
        for difficulty in DifficultyLevel:
            count = difficulty_counts.get(difficulty, 0)
            percentage = (count / total_cases) * 100
            if percentage < 10:
                suggestions.append(f"Consider adding more {difficulty.value} difficulty test cases (currently {percentage:.1f}%)")
        
        # Suggest adding expected outputs
        cases_without_expected = sum(1 for tc in dataset.test_cases if not tc.expected_output)
        if cases_without_expected > total_cases * 0.5:
            suggestions.append("Consider adding expected outputs to more test cases for better evaluation")
        
        # Suggest adding hallucination tests
        hallucination_tests = sum(1 for tc in dataset.test_cases if "hallucination_test" in tc.tags)
        if hallucination_tests < 3:
            suggestions.append("Consider adding more hallucination detection test cases")
        
        # Suggest adding consistency tests
        consistency_tests = sum(1 for tc in dataset.test_cases if "consistency_test" in tc.tags)
        if consistency_tests < 2:
            suggestions.append("Consider adding consistency test cases")
        
        return suggestions