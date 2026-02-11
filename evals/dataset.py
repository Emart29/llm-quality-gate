"""Evaluation dataset management and schema."""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import json
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks for evaluation."""
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    TRANSLATION = "translation"
    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    CREATIVE_WRITING = "creative_writing"
    FACTUAL_RECALL = "factual_recall"


class DifficultyLevel(str, Enum):
    """Difficulty levels for test cases."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class EvaluationMetric(str, Enum):
    """Metrics that can be evaluated for each test case."""
    TASK_SUCCESS = "task_success"
    RELEVANCE = "relevance"
    HALLUCINATION = "hallucination"
    CONSISTENCY = "consistency"
    ALL = "all"


class TestCase(BaseModel):
    """Individual test case in the evaluation dataset."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_prompt: str = Field(..., description="The input prompt to test")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    expected_output: Optional[str] = Field(None, description="Expected/reference output")
    
    # Metadata
    task_type: TaskType = Field(..., description="Type of task being evaluated")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    # Evaluation configuration
    metrics_to_evaluate: List[EvaluationMetric] = Field(
        default_factory=lambda: [EvaluationMetric.ALL],
        description="Which metrics to evaluate for this test case"
    )
    
    # Test case specific thresholds (override global ones)
    custom_thresholds: Optional[Dict[str, float]] = Field(
        None, 
        description="Custom thresholds for this test case"
    )
    
    # Additional context
    context: Optional[str] = Field(None, description="Additional context for the task")
    constraints: List[str] = Field(default_factory=list, description="Constraints for the response")
    
    # Metadata for tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Creator of the test case")
    version: str = Field(default="1.0", description="Version of the test case")
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Ensure tags are lowercase and alphanumeric."""
        return [tag.lower().replace(' ', '_') for tag in v if tag.strip()]
    
    @field_validator('input_prompt')
    @classmethod
    def validate_input_prompt(cls, v):
        """Ensure input prompt is not empty."""
        if not v.strip():
            raise ValueError("Input prompt cannot be empty")
        return v.strip()


class EvaluationDataset(BaseModel):
    """Complete evaluation dataset with metadata."""
    
    name: str = Field(..., description="Name of the dataset")
    description: str = Field(..., description="Description of the dataset")
    version: str = Field(default="1.0", description="Dataset version")
    
    test_cases: List[TestCase] = Field(default_factory=list)
    
    # Dataset metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None)
    
    # Global configuration
    global_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "task_success_threshold": 0.8,
            "relevance_threshold": 0.7,
            "hallucination_threshold": 0.1,
            "consistency_threshold": 0.8
        }
    )
    
    # Statistics (computed)
    stats: Optional[Dict[str, Any]] = Field(None, description="Dataset statistics")
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the dataset."""
        self.test_cases.append(test_case)
        self.updated_at = datetime.utcnow()
        self._update_stats()
    
    def remove_test_case(self, test_case_id: str) -> bool:
        """Remove a test case by ID."""
        original_length = len(self.test_cases)
        self.test_cases = [tc for tc in self.test_cases if tc.id != test_case_id]
        
        if len(self.test_cases) < original_length:
            self.updated_at = datetime.utcnow()
            self._update_stats()
            return True
        return False
    
    def get_test_case(self, test_case_id: str) -> Optional[TestCase]:
        """Get a test case by ID."""
        for test_case in self.test_cases:
            if test_case.id == test_case_id:
                return test_case
        return None
    
    def filter_by_task_type(self, task_type: TaskType) -> List[TestCase]:
        """Filter test cases by task type."""
        return [tc for tc in self.test_cases if tc.task_type == task_type]
    
    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> List[TestCase]:
        """Filter test cases by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def filter_by_tags(self, tags: List[str]) -> List[TestCase]:
        """Filter test cases that have any of the specified tags."""
        tag_set = set(tag.lower() for tag in tags)
        return [tc for tc in self.test_cases if any(tag in tag_set for tag in tc.tags)]
    
    def _update_stats(self) -> None:
        """Update dataset statistics."""
        if not self.test_cases:
            self.stats = {
                "total_test_cases": 0,
                "task_types": {},
                "difficulty_distribution": {},
                "tag_distribution": {}
            }
            return
        
        # Count by task type
        task_type_counts = {}
        for tc in self.test_cases:
            task_type_counts[tc.task_type.value] = task_type_counts.get(tc.task_type.value, 0) + 1
        
        # Count by difficulty
        difficulty_counts = {}
        for tc in self.test_cases:
            difficulty_counts[tc.difficulty.value] = difficulty_counts.get(tc.difficulty.value, 0) + 1
        
        # Count tags
        tag_counts = {}
        for tc in self.test_cases:
            for tag in tc.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        self.stats = {
            "total_test_cases": len(self.test_cases),
            "task_types": task_type_counts,
            "difficulty_distribution": difficulty_counts,
            "tag_distribution": tag_counts,
            "avg_prompt_length": sum(len(tc.input_prompt) for tc in self.test_cases) / len(self.test_cases),
            "cases_with_expected_output": sum(1 for tc in self.test_cases if tc.expected_output),
            "cases_with_system_prompt": sum(1 for tc in self.test_cases if tc.system_prompt)
        }


class DatasetLoader:
    """Utility class for loading and saving evaluation datasets."""
    
    @staticmethod
    def load_from_file(file_path: str) -> EvaluationDataset:
        """Load dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset = EvaluationDataset(**data)
            dataset._update_stats()
            logger.info(f"Loaded dataset '{dataset.name}' with {len(dataset.test_cases)} test cases")
            return dataset
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    @staticmethod
    def save_to_file(dataset: EvaluationDataset, file_path: str) -> None:
        """Save dataset to JSON file."""
        try:
            dataset._update_stats()  # Update stats before saving
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    dataset.model_dump(), 
                    f, 
                    indent=2, 
                    default=str,  # Handle datetime serialization
                    ensure_ascii=False
                )
            
            logger.info(f"Saved dataset '{dataset.name}' to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    @staticmethod
    def create_sample_dataset() -> EvaluationDataset:
        """Create a sample dataset for testing and demonstration."""
        dataset = EvaluationDataset(
            name="Sample LLM Quality Gate Dataset",
            description="A sample dataset demonstrating various task types and evaluation scenarios",
            version="1.0",
            created_by="LLM Quality Gate System"
        )
        
        # Sample test cases covering different scenarios
        sample_cases = [
            # Question Answering - Easy
            TestCase(
                input_prompt="What is the capital of France?",
                expected_output="Paris",
                task_type=TaskType.QUESTION_ANSWERING,
                difficulty=DifficultyLevel.EASY,
                tags=["geography", "factual", "simple"],
                metrics_to_evaluate=[EvaluationMetric.TASK_SUCCESS, EvaluationMetric.HALLUCINATION]
            ),
            
            # Reasoning - Medium
            TestCase(
                input_prompt="If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
                expected_output="150 miles",
                task_type=TaskType.REASONING,
                difficulty=DifficultyLevel.MEDIUM,
                tags=["math", "calculation", "reasoning"],
                constraints=["Must show calculation", "Answer must include units"]
            ),
            
            # Summarization - Medium
            TestCase(
                input_prompt="Summarize the following text in 2 sentences: [Long article about climate change...]",
                system_prompt="You are a helpful assistant that creates concise, accurate summaries.",
                task_type=TaskType.SUMMARIZATION,
                difficulty=DifficultyLevel.MEDIUM,
                tags=["summarization", "comprehension"],
                metrics_to_evaluate=[EvaluationMetric.RELEVANCE, EvaluationMetric.CONSISTENCY]
            ),
            
            # Creative Writing - Hard
            TestCase(
                input_prompt="Write a short story about a robot learning to paint, in exactly 100 words.",
                task_type=TaskType.CREATIVE_WRITING,
                difficulty=DifficultyLevel.HARD,
                tags=["creative", "storytelling", "constrained"],
                constraints=["Exactly 100 words", "Must include robot character", "Theme: learning to paint"],
                metrics_to_evaluate=[EvaluationMetric.TASK_SUCCESS, EvaluationMetric.RELEVANCE]
            ),
            
            # Hallucination Detection Test
            TestCase(
                input_prompt="What did Albert Einstein say about quantum mechanics in his 1955 Nobel Prize speech?",
                expected_output="Einstein never gave a Nobel Prize speech in 1955. He won the Nobel Prize in Physics in 1921, and he died in 1955.",
                task_type=TaskType.FACTUAL_RECALL,
                difficulty=DifficultyLevel.HARD,
                tags=["factual", "hallucination_test", "historical"],
                metrics_to_evaluate=[EvaluationMetric.HALLUCINATION, EvaluationMetric.TASK_SUCCESS]
            ),
            
            # Consistency Test Case
            TestCase(
                input_prompt="Explain the concept of machine learning in simple terms.",
                system_prompt="You are an expert teacher explaining complex topics to beginners.",
                task_type=TaskType.GENERATION,
                difficulty=DifficultyLevel.MEDIUM,
                tags=["explanation", "consistency_test", "technical"],
                metrics_to_evaluate=[EvaluationMetric.CONSISTENCY, EvaluationMetric.RELEVANCE]
            )
        ]
        
        for test_case in sample_cases:
            dataset.add_test_case(test_case)
        
        return dataset