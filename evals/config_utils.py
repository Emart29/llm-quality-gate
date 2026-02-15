"""Configuration utilities for evaluation pipeline."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "llmq.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def get_evaluation_config(config_path: str = "llmq.yaml") -> Dict[str, Any]:
    """Get evaluation configuration with CI mode optimizations."""
    config = load_config(config_path)
    eval_config = config.get("evaluation", {})
    
    # Apply CI mode optimizations
    if os.environ.get('CI', '').lower() in ('true', '1', 'yes'):
        # Reduce consistency runs in CI
        eval_config["consistency_runs"] = min(eval_config.get("consistency_runs", 3), 2)
        # Enable lightweight mode
        eval_config["ci_lightweight_mode"] = True
        # Reduce parallel requests to avoid rate limits
        eval_config["parallel_requests"] = min(eval_config.get("parallel_requests", 5), 3)
        # Reduce retry attempts for faster failure
        eval_config["max_retries"] = min(eval_config.get("max_retries", 3), 2)
    
    return eval_config


def should_skip_heavy_metrics() -> bool:
    """Check if heavy metrics should be skipped (CI mode)."""
    return os.environ.get('CI', '').lower() in ('true', '1', 'yes')


def get_ci_optimized_thresholds() -> Optional[Dict[str, float]]:
    """Get CI-optimized thresholds that are more lenient for mock embeddings."""
    if not should_skip_heavy_metrics():
        return None
    
    return {
        "task_success": 0.6,  # Lower threshold for mock embeddings
        "relevance": 0.5,     # Lower threshold for mock embeddings
        "hallucination": 0.2, # Slightly higher tolerance
        "consistency": 0.6,   # Lower threshold for reduced runs
    }