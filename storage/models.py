"""Data models for the storage layer."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class EvaluationRun:
    """Represents a complete evaluation run."""
    id: str
    dataset_name: str
    dataset_version: str
    provider_name: str
    model_name: str
    total_test_cases: int
    successful_executions: int
    failed_executions: int
    total_execution_time: float
    success_rate: float
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    trigger: Optional[str] = None
    pass_rate: Optional[float] = None
    quality_gate_passed: Optional[bool] = None
    overall_score: Optional[float] = None
    task_success_score: Optional[float] = None
    relevance_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    consistency_score: Optional[float] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestCaseRecord:
    """Represents a single test case result stored in the database."""
    id: str
    run_id: str
    test_case_id: str
    input_prompt: str = ""
    expected_output: Optional[str] = None
    generated_output: str = ""
    task_type: str = ""
    difficulty: str = ""
    execution_time: float = 0.0
    success: bool = False
    passed: Optional[bool] = None
    error: Optional[str] = None
    task_success_score: Optional[float] = None
    relevance_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    consistency_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityGateRecord:
    """Represents a quality gate pass/fail event."""
    id: str
    run_id: str
    passed: bool
    overall_score: Optional[float] = None
    failed_metrics: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    commit_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProviderComparison:
    """Represents a comparison between providers/models."""
    id: str
    dataset_name: str
    dataset_version: str
    comparison_name: Optional[str] = None
    providers: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
