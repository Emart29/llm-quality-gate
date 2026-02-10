"""CI runner script for automated LLM quality gate evaluation."""

import json
import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_git_info():
    """Get current git information."""
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    return {
        "commit_hash": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_message": _run(["git", "log", "-1", "--pretty=%s"]),
    }


def run_evaluation(provider: str = "groq", model: str = None):
    """Run the evaluation pipeline."""
    from llm.factory import LLMFactory, BaseLLM
    from evals.dataset import DatasetLoader
    from evals.runner import EvaluationRunner
    from evals.metrics import MetricsEngine
    from evals.comprehensive_runner import ComprehensiveEvaluationRunner
    import asyncio

    config = LLMFactory.load_config(str(PROJECT_ROOT / "config.yaml"))
    factory = LLMFactory(str(PROJECT_ROOT / "config.yaml"))

    if not model:
        model = config.get("providers", {}).get(provider, {}).get("model", "")

    dataset_path = str(PROJECT_ROOT / "evals" / "dataset.json")
    dataset = DatasetLoader.load_from_file(dataset_path)

    # Set up judge
    judge_llm = None
    try:
        judge_provider = factory.create_provider_for_role("judge", config=config)
        judge_llm = BaseLLM(judge_provider)
    except Exception:
        logger.warning("Could not initialize judge LLM, using heuristic metrics")

    thresholds = config.get("quality_gates", {})
    metrics_engine = MetricsEngine(judge_llm=judge_llm, thresholds=thresholds)
    runner = EvaluationRunner(llm_factory=factory, deterministic=True)
    comp_runner = ComprehensiveEvaluationRunner(runner, metrics_engine)

    result = asyncio.run(comp_runner.run(dataset, provider, model))
    return result


def send_notification(result_dict, git_info, webhook_url=None):
    """Send notification about evaluation results (Slack or console)."""
    gate = result_dict.get("quality_gate", {})
    status = "PASSED" if gate.get("passed") else "FAILED"
    provider = result_dict.get("provider_name", "?")
    model = result_dict.get("model_name", "?")
    score = gate.get("overall_score", 0)

    message = (
        f"LLM Quality Gate: {status}\n"
        f"Provider: {provider}/{model}\n"
        f"Score: {score:.2%}\n"
        f"Commit: {git_info.get('commit_hash', 'N/A')}\n"
        f"Branch: {git_info.get('branch', 'N/A')}"
    )

    if gate.get("failed_metrics"):
        message += f"\nFailed: {', '.join(gate['failed_metrics'])}"

    logger.info(message)

    if webhook_url:
        try:
            import httpx
            httpx.post(webhook_url, json={"text": message}, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    return message


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CI Runner for LLM Quality Gate")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--slack-webhook", default=os.getenv("SLACK_WEBHOOK_URL"))
    parser.add_argument("--fail-on-gate", action="store_true", default=True)
    args = parser.parse_args()

    git_info = get_git_info()
    logger.info(f"Git: {git_info['commit_hash']} on {git_info['branch']}")

    result = run_evaluation(args.provider, args.model)
    result_dict = result.to_dict()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ci_{args.provider}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Store in database
    try:
        from storage.database import Database
        from storage.repository import EvaluationRepository
        db = Database()
        repo = EvaluationRepository(db)
        repo.save_comprehensive_result(
            result_dict,
            commit_hash=git_info.get("commit_hash"),
            branch=git_info.get("branch"),
        )
        db.close()
    except Exception as e:
        logger.warning(f"Database storage failed: {e}")

    # Notify
    send_notification(result_dict, git_info, args.slack_webhook)

    # Exit code
    if args.fail_on_gate and not result.quality_gate.passed:
        logger.error("Quality gate FAILED")
        sys.exit(1)

    logger.info("Quality gate PASSED")


if __name__ == "__main__":
    main()
