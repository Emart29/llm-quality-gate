"""Dataset maintenance and growth automation tools."""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import shutil

from .dataset import EvaluationDataset, TestCase, DatasetLoader, TaskType, DifficultyLevel
from .validator import DatasetValidator, ValidationResult

logger = logging.getLogger(__name__)


class DatasetVersionManager:
    """Manages dataset versioning and change tracking."""
    
    def __init__(self, dataset_path: str, versions_dir: str = "dataset_versions"):
        self.dataset_path = Path(dataset_path)
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
    
    def create_version(self, dataset: EvaluationDataset, change_description: str) -> str:
        """Create a new version of the dataset."""
        # Increment version
        current_version = dataset.version
        version_parts = current_version.split('.')
        
        if len(version_parts) == 2:
            major, minor = int(version_parts[0]), int(version_parts[1])
            new_version = f"{major}.{minor + 1}"
        else:
            major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
            new_version = f"{major}.{minor}.{patch + 1}"
        
        # Update dataset version and timestamp
        dataset.version = new_version
        dataset.updated_at = datetime.utcnow()
        
        # Save versioned copy
        version_filename = f"dataset_v{new_version.replace('.', '_')}.json"
        version_path = self.versions_dir / version_filename
        
        DatasetLoader.save_to_file(dataset, str(version_path))
        
        # Create change log entry
        self._log_change(new_version, change_description, len(dataset.test_cases))
        
        logger.info(f"Created dataset version {new_version}: {change_description}")
        return new_version
    
    def _log_change(self, version: str, description: str, test_count: int) -> None:
        """Log a change to the dataset."""
        changelog_path = self.versions_dir / "CHANGELOG.md"
        
        entry = f"""
## Version {version} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

**Test Cases:** {test_count}
**Changes:** {description}

---
"""
        
        if changelog_path.exists():
            # Prepend to existing changelog
            existing_content = changelog_path.read_text()
            changelog_path.write_text(entry + existing_content)
        else:
            # Create new changelog
            header = "# Dataset Changelog\n\nThis file tracks all changes to the evaluation dataset.\n"
            changelog_path.write_text(header + entry)
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get the version history of the dataset."""
        versions = []
        for version_file in sorted(self.versions_dir.glob("dataset_v*.json")):
            try:
                dataset = DatasetLoader.load_from_file(str(version_file))
                versions.append({
                    "version": dataset.version,
                    "created_at": dataset.updated_at,
                    "test_count": len(dataset.test_cases),
                    "file_path": str(version_file)
                })
            except Exception as e:
                logger.warning(f"Failed to load version file {version_file}: {e}")
        
        return sorted(versions, key=lambda x: x["created_at"], reverse=True)
    
    def rollback_to_version(self, target_version: str) -> bool:
        """Rollback dataset to a specific version."""
        version_filename = f"dataset_v{target_version.replace('.', '_')}.json"
        version_path = self.versions_dir / version_filename
        
        if not version_path.exists():
            logger.error(f"Version {target_version} not found")
            return False
        
        try:
            # Create backup of current dataset
            backup_path = self.dataset_path.with_suffix(f".backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(self.dataset_path, backup_path)
            
            # Restore target version
            shutil.copy2(version_path, self.dataset_path)
            
            logger.info(f"Rolled back dataset to version {target_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback to version {target_version}: {e}")
            return False


class DatasetGrowthAutomator:
    """Automates dataset growth and maintenance tasks."""
    
    def __init__(self, dataset_path: str, validator: DatasetValidator):
        self.dataset_path = dataset_path
        self.validator = validator
        self.loader = DatasetLoader()
    
    def analyze_coverage_gaps(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Analyze gaps in dataset coverage."""
        gaps = {
            "task_types": self._analyze_task_type_coverage(dataset),
            "difficulty_levels": self._analyze_difficulty_coverage(dataset),
            "prompt_patterns": self._analyze_prompt_patterns(dataset),
            "output_lengths": self._analyze_output_lengths(dataset),
            "edge_cases": self._identify_missing_edge_cases(dataset)
        }
        
        return gaps
    
    def _analyze_task_type_coverage(self, dataset: EvaluationDataset) -> Dict[str, int]:
        """Analyze coverage across different task types."""
        task_counts = {}
        for task_type in TaskType:
            count = sum(1 for tc in dataset.test_cases if tc.task_type == task_type)
            task_counts[task_type.value] = count
        
        return task_counts
    
    def _analyze_difficulty_coverage(self, dataset: EvaluationDataset) -> Dict[str, int]:
        """Analyze coverage across difficulty levels."""
        difficulty_counts = {}
        for difficulty in DifficultyLevel:
            count = sum(1 for tc in dataset.test_cases if tc.difficulty == difficulty)
            difficulty_counts[difficulty.value] = count
        
        return difficulty_counts
    
    def _analyze_prompt_patterns(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Analyze patterns in prompts to identify gaps."""
        prompt_lengths = [len(tc.input_prompt.split()) for tc in dataset.test_cases]
        
        return {
            "avg_length": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
            "min_length": min(prompt_lengths) if prompt_lengths else 0,
            "max_length": max(prompt_lengths) if prompt_lengths else 0,
            "short_prompts": sum(1 for length in prompt_lengths if length < 10),
            "medium_prompts": sum(1 for length in prompt_lengths if 10 <= length <= 50),
            "long_prompts": sum(1 for length in prompt_lengths if length > 50)
        }
    
    def _analyze_output_lengths(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Analyze expected output length patterns."""
        output_lengths = []
        for tc in dataset.test_cases:
            if tc.expected_output:
                output_lengths.append(len(tc.expected_output.split()))
        
        if not output_lengths:
            return {"message": "No expected outputs to analyze"}
        
        return {
            "avg_length": sum(output_lengths) / len(output_lengths),
            "min_length": min(output_lengths),
            "max_length": max(output_lengths),
            "short_outputs": sum(1 for length in output_lengths if length < 20),
            "medium_outputs": sum(1 for length in output_lengths if 20 <= length <= 100),
            "long_outputs": sum(1 for length in output_lengths if length > 100)
        }
    
    def _identify_missing_edge_cases(self, dataset: EvaluationDataset) -> List[str]:
        """Identify missing edge cases that should be tested."""
        edge_cases = []
        
        # Check for common edge cases
        has_empty_input = any(not tc.input_prompt.strip() for tc in dataset.test_cases)
        has_very_long_input = any(len(tc.input_prompt.split()) > 500 for tc in dataset.test_cases)
        has_special_chars = any(any(char in tc.input_prompt for char in "!@#$%^&*()") for tc in dataset.test_cases)
        has_multilingual = any(any(ord(char) > 127 for char in tc.input_prompt) for tc in dataset.test_cases)
        
        if not has_empty_input:
            edge_cases.append("Empty or whitespace-only inputs")
        if not has_very_long_input:
            edge_cases.append("Very long inputs (>500 words)")
        if not has_special_chars:
            edge_cases.append("Inputs with special characters")
        if not has_multilingual:
            edge_cases.append("Non-English/multilingual inputs")
        
        return edge_cases
    
    def suggest_new_test_cases(self, dataset: EvaluationDataset, count: int = 10) -> List[Dict[str, Any]]:
        """Suggest new test cases based on coverage gaps."""
        gaps = self.analyze_coverage_gaps(dataset)
        suggestions = []
        
        # Suggest based on task type gaps
        task_counts = gaps["task_types"]
        min_task_count = min(task_counts.values()) if task_counts else 0
        
        for task_type, count in task_counts.items():
            if count <= min_task_count + 2:  # Underrepresented task types
                suggestions.append({
                    "type": "task_coverage",
                    "task_type": task_type,
                    "reason": f"Low coverage for {task_type} tasks ({count} cases)",
                    "priority": "high" if count == 0 else "medium"
                })
        
        # Suggest based on difficulty gaps
        difficulty_counts = gaps["difficulty_levels"]
        min_difficulty_count = min(difficulty_counts.values()) if difficulty_counts else 0
        
        for difficulty, count in difficulty_counts.items():
            if count <= min_difficulty_count + 2:
                suggestions.append({
                    "type": "difficulty_coverage",
                    "difficulty": difficulty,
                    "reason": f"Low coverage for {difficulty} difficulty ({count} cases)",
                    "priority": "medium"
                })
        
        # Suggest edge cases
        for edge_case in gaps["edge_cases"]:
            suggestions.append({
                "type": "edge_case",
                "description": edge_case,
                "reason": "Missing important edge case coverage",
                "priority": "high"
            })
        
        return suggestions[:count]
    
    def auto_clean_dataset(self, dataset: EvaluationDataset) -> Tuple[EvaluationDataset, List[str]]:
        """Automatically clean and optimize the dataset."""
        changes = []
        cleaned_cases = []
        
        for test_case in dataset.test_cases:
            # Clean whitespace
            original_prompt = test_case.input_prompt
            cleaned_prompt = test_case.input_prompt.strip()
            
            if original_prompt != cleaned_prompt:
                test_case.input_prompt = cleaned_prompt
                changes.append(f"Cleaned whitespace in test case {test_case.id}")
            
            # Clean expected output
            if test_case.expected_output:
                original_output = test_case.expected_output
                cleaned_output = test_case.expected_output.strip()
                
                if original_output != cleaned_output:
                    test_case.expected_output = cleaned_output
                    changes.append(f"Cleaned expected output in test case {test_case.id}")
            
            # Validate and keep valid cases
            validation_result = self.validator.validate_test_case(test_case)
            if validation_result.is_valid:
                cleaned_cases.append(test_case)
            else:
                changes.append(f"Removed invalid test case {test_case.id}: {validation_result.errors}")
        
        # Remove duplicates
        unique_cases = []
        seen_hashes = set()
        
        for test_case in cleaned_cases:
            case_hash = self._hash_test_case(test_case)
            if case_hash not in seen_hashes:
                unique_cases.append(test_case)
                seen_hashes.add(case_hash)
            else:
                changes.append(f"Removed duplicate test case {test_case.id}")
        
        # Create cleaned dataset
        cleaned_dataset = EvaluationDataset(
            version=dataset.version,
            created_at=dataset.created_at,
            updated_at=datetime.utcnow(),
            test_cases=unique_cases,
            metadata=dataset.metadata
        )
        
        return cleaned_dataset, changes
    
    def _hash_test_case(self, test_case: TestCase) -> str:
        """Create a hash for a test case to detect duplicates."""
        content = f"{test_case.input_prompt}|{test_case.expected_output}|{test_case.task_type.value}|{test_case.difficulty.value}"
        return hashlib.md5(content.encode()).hexdigest()


class DatasetMaintenanceScheduler:
    """Schedules and manages automated dataset maintenance tasks."""
    
    def __init__(self, dataset_path: str, maintenance_config: Dict[str, Any]):
        self.dataset_path = dataset_path
        self.config = maintenance_config
        self.loader = DatasetLoader()
        self.validator = DatasetValidator()
        self.version_manager = DatasetVersionManager(dataset_path)
        self.growth_automator = DatasetGrowthAutomator(dataset_path, self.validator)
    
    def run_maintenance_cycle(self) -> Dict[str, Any]:
        """Run a complete maintenance cycle."""
        logger.info("Starting dataset maintenance cycle")
        
        try:
            # Load current dataset
            dataset = self.loader.load_from_file(self.dataset_path)
            
            # Run maintenance tasks
            results = {
                "timestamp": datetime.utcnow().isoformat(),
                "original_test_count": len(dataset.test_cases),
                "tasks_completed": []
            }
            
            # 1. Validate dataset
            if self.config.get("validate", True):
                validation_result = self.validator.validate_dataset(dataset)
                results["validation"] = {
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
                results["tasks_completed"].append("validation")
            
            # 2. Clean dataset
            if self.config.get("auto_clean", True):
                cleaned_dataset, changes = self.growth_automator.auto_clean_dataset(dataset)
                results["cleaning"] = {
                    "changes_made": len(changes),
                    "changes": changes,
                    "final_test_count": len(cleaned_dataset.test_cases)
                }
                dataset = cleaned_dataset
                results["tasks_completed"].append("cleaning")
            
            # 3. Analyze coverage gaps
            if self.config.get("analyze_gaps", True):
                gaps = self.growth_automator.analyze_coverage_gaps(dataset)
                results["coverage_analysis"] = gaps
                results["tasks_completed"].append("coverage_analysis")
            
            # 4. Generate suggestions
            if self.config.get("generate_suggestions", True):
                suggestions = self.growth_automator.suggest_new_test_cases(dataset)
                results["suggestions"] = suggestions
                results["tasks_completed"].append("suggestions")
            
            # 5. Create version if changes were made
            if results.get("cleaning", {}).get("changes_made", 0) > 0:
                change_description = f"Automated maintenance: {len(results['cleaning']['changes'])} changes"
                new_version = self.version_manager.create_version(dataset, change_description)
                results["new_version"] = new_version
                
                # Save cleaned dataset
                self.loader.save_to_file(dataset, self.dataset_path)
                results["tasks_completed"].append("versioning")
            
            logger.info(f"Maintenance cycle completed. Tasks: {', '.join(results['tasks_completed'])}")
            return results
            
        except Exception as e:
            logger.error(f"Maintenance cycle failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "tasks_completed": []
            }
    
    def schedule_maintenance(self, interval_hours: int = 24) -> None:
        """Schedule regular maintenance (placeholder for actual scheduling)."""
        logger.info(f"Maintenance scheduled to run every {interval_hours} hours")
        # In a real implementation, this would integrate with a task scheduler
        # like Celery, APScheduler, or cron jobs