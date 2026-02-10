"""Test case execution runner with batch processing and parallel execution."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import time
import json
from pathlib import Path

from .dataset import EvaluationDataset, TestCase, TaskType, EvaluationMetric
from ..llm.base import BaseLLM
from ..llm.factory import LLMFactory

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Result of executing a single test case."""
    test_case_id: str
    test_case: TestCase
    generated_output: str
    execution_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Result of running an evaluation on a dataset."""
    dataset_name: str
    dataset_version: str
    provider_name: str
    model_name: str
    execution_timestamp: datetime
    total_test_cases: int
    successful_executions: int
    failed_executions: int
    total_execution_time: float
    test_case_results: List[TestCaseResult]
    configuration: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of test case executions."""
        if self.total_test_cases == 0:
            return 0.0
        return self.successful_executions / self.total_test_cases
    
    def get_results_by_task_type(self) -> Dict[TaskType, List[TestCaseResult]]:
        """Group results by task type."""
        results_by_type = {}
        for result in self.test_case_results:
            task_type = result.test_case.task_type
            if task_type not in results_by_type:
                results_by_type[task_type] = []
            results_by_type[task_type].append(result)
        return results_by_type
    
    def get_failed_results(self) -> List[TestCaseResult]:
        """Get all failed test case results."""
        return [result for result in self.test_case_results if not result.success]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            "success_rate": self.success_rate,
            "configuration": self.configuration,
            "test_case_results": [
                {
                    "test_case_id": result.test_case_id,
                    "generated_output": result.generated_output,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "error": result.error,
                    "metadata": result.metadata,
                    "task_type": result.test_case.task_type.value,
                    "difficulty": result.test_case.difficulty.value,
                    "tags": result.test_case.tags
                }
                for result in self.test_case_results
            ]
        }


class EvaluationRunner:
    """Runner for executing test cases against LLM providers."""
    
    def __init__(
        self,
        llm_factory: LLMFactory,
        max_workers: int = 5,
        timeout_seconds: int = 30,
        retry_attempts: int = 3,
        deterministic: bool = True
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            llm_factory: Factory for creating LLM instances
            max_workers: Maximum number of parallel workers
            timeout_seconds: Timeout for individual test case execution
            retry_attempts: Number of retry attempts for failed executions
            deterministic: Whether to use deterministic generation (temperature=0)
        """
        self.llm_factory = llm_factory
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.deterministic = deterministic
        
    async def run_evaluation(
        self,
        dataset: EvaluationDataset,
        provider_name: str,
        model_name: str,
        filter_func: Optional[Callable[[TestCase], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvaluationResult:
        """
        Run evaluation on a dataset using specified provider and model.
        
        Args:
            dataset: The evaluation dataset
            provider_name: Name of the LLM provider
            model_name: Name of the model to use
            filter_func: Optional function to filter test cases
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvaluationResult containing all execution results
        """
        start_time = time.time()
        
        # Filter test cases if filter function provided
        test_cases = dataset.test_cases
        if filter_func:
            test_cases = [tc for tc in test_cases if filter_func(tc)]
        
        logger.info(f"Starting evaluation with {len(test_cases)} test cases using {provider_name}/{model_name}")
        
        # Create LLM instance
        try:
            llm = self.llm_factory.create_llm(provider_name, model_name)
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {e}")
            raise
        
        # Configure for deterministic generation if requested
        generation_config = {}
        if self.deterministic:
            generation_config["temperature"] = 0.0
            generation_config["top_p"] = 1.0
        
        # Execute test cases in parallel
        test_case_results = []
        successful_executions = 0
        failed_executions = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test cases
            future_to_test_case = {
                executor.submit(
                    self._execute_test_case,
                    llm,
                    test_case,
                    generation_config
                ): test_case
                for test_case in test_cases
            }
            
            # Process completed futures
            completed = 0
            for future in as_completed(future_to_test_case):
                test_case = future_to_test_case[future]
                try:
                    result = future.result()
                    test_case_results.append(result)
                    
                    if result.success:
                        successful_executions += 1
                    else:
                        failed_executions += 1
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing test case {test_case.id}: {e}")
                    # Create failed result
                    failed_result = TestCaseResult(
                        test_case_id=test_case.id,
                        test_case=test_case,
                        generated_output="",
                        execution_time=0.0,
                        success=False,
                        error=f"Unexpected error: {str(e)}"
                    )
                    test_case_results.append(failed_result)
                    failed_executions += 1
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(test_cases))
        
        total_execution_time = time.time() - start_time
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            provider_name=provider_name,
            model_name=model_name,
            execution_timestamp=datetime.utcnow(),
            total_test_cases=len(test_cases),
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            total_execution_time=total_execution_time,
            test_case_results=test_case_results,
            configuration={
                "max_workers": self.max_workers,
                "timeout_seconds": self.timeout_seconds,
                "retry_attempts": self.retry_attempts,
                "deterministic": self.deterministic,
                "generation_config": generation_config
            }
        )
        
        logger.info(f"Evaluation completed: {successful_executions}/{len(test_cases)} successful")
        return evaluation_result
    
    def _execute_test_case(
        self,
        llm: BaseLLM,
        test_case: TestCase,
        generation_config: Dict[str, Any]
    ) -> TestCaseResult:
        """Execute a single test case with retry logic."""
        start_time = time.time()
        
        for attempt in range(self.retry_attempts):
            try:
                # Prepare the prompt
                messages = []
                if test_case.system_prompt:
                    messages.append({"role": "system", "content": test_case.system_prompt})
                messages.append({"role": "user", "content": test_case.input_prompt})
                
                # Generate response
                response = llm.generate(
                    messages=messages,
                    **generation_config,
                    timeout=self.timeout_seconds
                )
                
                execution_time = time.time() - start_time
                
                return TestCaseResult(
                    test_case_id=test_case.id,
                    test_case=test_case,
                    generated_output=response.content,
                    execution_time=execution_time,
                    success=True,
                    metadata={
                        "attempt": attempt + 1,
                        "token_count": getattr(response, 'token_count', None),
                        "model_used": getattr(response, 'model', None)
                    }
                )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for test case {test_case.id}: {e}")
                if attempt == self.retry_attempts - 1:
                    # Final attempt failed
                    execution_time = time.time() - start_time
                    return TestCaseResult(
                        test_case_id=test_case.id,
                        test_case=test_case,
                        generated_output="",
                        execution_time=execution_time,
                        success=False,
                        error=str(e),
                        metadata={"failed_attempts": self.retry_attempts}
                    )
                
                # Wait before retry
                time.sleep(1.0 * (attempt + 1))  # Exponential backoff
    
    def run_consistency_test(
        self,
        test_case: TestCase,
        provider_name: str,
        model_name: str,
        num_runs: int = 5
    ) -> List[TestCaseResult]:
        """
        Run the same test case multiple times to test consistency.
        
        Args:
            test_case: The test case to run multiple times
            provider_name: Name of the LLM provider
            model_name: Name of the model to use
            num_runs: Number of times to run the test case
            
        Returns:
            List of TestCaseResult for each run
        """
        logger.info(f"Running consistency test for test case {test_case.id} ({num_runs} runs)")
        
        # Create LLM instance
        llm = self.llm_factory.create_llm(provider_name, model_name)
        
        # Use non-deterministic generation for consistency testing
        generation_config = {"temperature": 0.7, "top_p": 0.9}
        
        results = []
        for run_num in range(num_runs):
            result = self._execute_test_case(llm, test_case, generation_config)
            result.metadata = result.metadata or {}
            result.metadata["consistency_run"] = run_num + 1
            results.append(result)
        
        return results
    
    def save_results(self, result: EvaluationResult, output_path: str) -> None:
        """Save evaluation results to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise


class BatchEvaluationRunner:
    """Runner for batch evaluations across multiple providers and models."""
    
    def __init__(self, evaluation_runner: EvaluationRunner):
        self.evaluation_runner = evaluation_runner
    
    async def run_batch_evaluation(
        self,
        dataset: EvaluationDataset,
        provider_model_configs: List[Dict[str, str]],
        output_dir: str = "evaluation_results",
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[EvaluationResult]:
        """
        Run evaluation across multiple provider/model combinations.
        
        Args:
            dataset: The evaluation dataset
            provider_model_configs: List of {"provider": "name", "model": "name"} configs
            output_dir: Directory to save results
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of EvaluationResult for each provider/model combination
        """
        results = []
        
        for i, config in enumerate(provider_model_configs):
            provider_name = config["provider"]
            model_name = config["model"]
            
            logger.info(f"Running evaluation {i+1}/{len(provider_model_configs)}: {provider_name}/{model_name}")
            
            try:
                result = await self.evaluation_runner.run_evaluation(
                    dataset=dataset,
                    provider_name=provider_name,
                    model_name=model_name,
                    progress_callback=lambda completed, total: progress_callback(
                        f"{provider_name}/{model_name}", completed, total
                    ) if progress_callback else None
                )
                
                results.append(result)
                
                # Save individual result
                output_file = f"{output_dir}/{provider_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.evaluation_runner.save_results(result, output_file)
                
            except Exception as e:
                logger.error(f"Failed evaluation for {provider_name}/{model_name}: {e}")
                continue
        
        return results