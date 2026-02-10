"""LLMQ CLI - LLM Quality Gate command-line interface."""

import argparse
import asyncio
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Commands ──────────────────────────────────────────────────────────

def cmd_init(args):
    """Initialize project structure and configuration."""
    dirs = ["evals", "llm", "storage", "dashboard", "evaluation_results"]
    for d in dirs:
        p = PROJECT_ROOT / d
        p.mkdir(exist_ok=True)
        print(f"  [ok] {d}/")

    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        import shutil
        template = PROJECT_ROOT / "config.yaml.example"
        if template.exists():
            shutil.copy(template, config_path)
        print("  [ok] config.yaml created (edit API keys in .env)")
    else:
        print("  [ok] config.yaml already exists")

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, env_path)
        print("  [ok] .env created (add your API keys)")
    else:
        print("  [ok] .env already exists")

    print("\nProject initialized. Next steps:")
    print("  1. Add API keys to .env")
    print("  2. Run: llmq eval --provider groq")


def cmd_eval(args):
    """Run LLM evaluation."""
    from llm.factory import LLMFactory
    from evals.dataset import DatasetLoader
    from evals.runner import EvaluationRunner
    from evals.metrics import MetricsEngine
    from evals.comprehensive_runner import ComprehensiveEvaluationRunner

    config = LLMFactory.load_config(args.config)

    provider = args.provider or config.get("llm", {}).get("default_provider", "groq")
    model = args.model or config.get("providers", {}).get(provider, {}).get("model", "")

    dataset_path = args.dataset or str(PROJECT_ROOT / "evals" / "dataset.json")
    print(f"Loading dataset from {dataset_path}")
    dataset = DatasetLoader.load_from_file(dataset_path)
    print(f"Loaded {len(dataset.test_cases)} test cases")

    factory = LLMFactory(args.config)

    # Set up metrics engine with judge if configured
    judge_llm = None
    if not args.no_judge:
        try:
            from llm.factory import BaseLLM
            judge_provider = factory.create_provider_for_role("judge", config=config)
            judge_llm = BaseLLM(judge_provider)
        except Exception:
            pass

    thresholds = config.get("quality_gates", {})
    metrics_engine = MetricsEngine(judge_llm=judge_llm, thresholds=thresholds)

    runner = EvaluationRunner(
        llm_factory=factory,
        max_workers=args.workers,
        timeout_seconds=args.timeout,
        deterministic=not args.non_deterministic,
    )
    comp_runner = ComprehensiveEvaluationRunner(runner, metrics_engine)

    print(f"\nRunning evaluation: {provider}/{model}")
    print("-" * 50)

    result = asyncio.run(comp_runner.run(dataset, provider, model))
    result_dict = result.to_dict()

    # Print summary
    gate = result.quality_gate
    print(f"\n{'=' * 50}")
    print(f"Quality Gate: {'PASSED' if gate.passed else 'FAILED'}")
    print(f"Overall Score: {gate.overall_score:.2%}")
    print(f"Test Cases: {result.total_test_cases} total, {result.successful_executions} succeeded")
    print(f"Execution Time: {result.total_execution_time:.1f}s")

    for name, m in result.aggregated_metrics.items():
        mark = "PASS" if m.passed else "FAIL"
        print(f"  [{mark}] {name}: {m.score:.3f} (threshold {m.threshold})")

    if gate.failed_metrics:
        print(f"\nFailed metrics: {', '.join(gate.failed_metrics)}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_dir = PROJECT_ROOT / "evaluation_results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"{provider}_{model}_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Store in database
    if not args.no_db:
        try:
            from storage.database import Database
            from storage.repository import EvaluationRepository
            db = Database()
            repo = EvaluationRepository(db)
            commit_hash = _get_git_hash()
            branch = _get_git_branch()
            run_id = repo.save_comprehensive_result(result_dict, commit_hash=commit_hash, branch=branch)
            print(f"Stored in database (run_id: {run_id[:8]}...)")
            db.close()
        except Exception as e:
            print(f"Warning: Could not store in database: {e}")

    # Exit code based on quality gate
    if not gate.passed and args.fail_on_gate:
        sys.exit(1)


def cmd_dashboard(args):
    """Launch the web dashboard."""
    print(f"Starting dashboard on http://localhost:{args.port}")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "dashboard.app:app",
        "--host", args.host,
        "--port", str(args.port),
        "--reload" if args.reload else "--no-access-log",
    ], cwd=str(PROJECT_ROOT))


def cmd_compare(args):
    """Compare providers or models."""
    from storage.database import Database
    from storage.repository import EvaluationRepository

    db = Database()
    repo = EvaluationRepository(db)

    runs = repo.list_runs(limit=100)
    if not runs:
        print("No evaluation runs found. Run `llmq eval` first.")
        db.close()
        return

    # Group by provider/model
    groups: dict = {}
    for r in runs:
        key = f"{r['provider_name']}/{r['model_name']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    print(f"\n{'Provider/Model':<35} {'Runs':>5} {'Score':>8} {'Task':>8} {'Relev':>8} {'Halluc':>8} {'Gate':>6}")
    print("-" * 88)

    for name, group_runs in sorted(groups.items()):
        n = len(group_runs)
        avg = lambda field: sum(r.get(field, 0) or 0 for r in group_runs) / n
        gate_pct = sum(1 for r in group_runs if r.get("quality_gate_passed")) / n * 100

        print(f"{name:<35} {n:>5} {avg('overall_score'):>7.1%} {avg('task_success_score'):>7.1%} "
              f"{avg('relevance_score'):>7.1%} {avg('hallucination_score'):>7.1%} {gate_pct:>5.0f}%")

    db.close()


def cmd_validate(args):
    """Validate the evaluation dataset."""
    from evals.dataset import DatasetLoader
    from evals.validator import DatasetValidator

    dataset_path = args.dataset or str(PROJECT_ROOT / "evals" / "dataset.json")
    dataset = DatasetLoader.load_from_file(dataset_path)
    validator = DatasetValidator()
    result = validator.validate_dataset(dataset)

    print(result)

    if args.suggestions:
        suggestions = validator.suggest_improvements(dataset)
        if suggestions:
            print("Suggestions:")
            for s in suggestions:
                print(f"  - {s}")


# ── Helpers ───────────────────────────────────────────────────────────

def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


# ── Argument parser ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmq",
        description="LLMQ - LLM Quality Gate CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    sub.add_parser("init", help="Initialize project structure and configs")

    # eval
    p_eval = sub.add_parser("eval", help="Run LLM evaluation")
    p_eval.add_argument("--provider", "-p", help="LLM provider name")
    p_eval.add_argument("--model", "-m", help="Model name override")
    p_eval.add_argument("--dataset", "-d", help="Path to dataset JSON")
    p_eval.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    p_eval.add_argument("--output", "-o", help="Output file path")
    p_eval.add_argument("--workers", type=int, default=5, help="Parallel workers")
    p_eval.add_argument("--timeout", type=int, default=30, help="Timeout per test case")
    p_eval.add_argument("--no-judge", action="store_true", help="Disable LLM-as-Judge")
    p_eval.add_argument("--no-db", action="store_true", help="Skip database storage")
    p_eval.add_argument("--non-deterministic", action="store_true", help="Use temperature > 0")
    p_eval.add_argument("--fail-on-gate", action="store_true", help="Exit 1 if quality gate fails")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", type=int, default=8000, help="Port number")
    p_dash.add_argument("--host", default="127.0.0.1", help="Host address")
    p_dash.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # compare
    sub.add_parser("compare", help="Compare providers and models")

    # validate
    p_val = sub.add_parser("validate", help="Validate evaluation dataset")
    p_val.add_argument("--dataset", "-d", help="Path to dataset JSON")
    p_val.add_argument("--suggestions", "-s", action="store_true", help="Show improvement suggestions")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    commands = {
        "init": cmd_init,
        "eval": cmd_eval,
        "dashboard": cmd_dashboard,
        "compare": cmd_compare,
        "validate": cmd_validate,
    }

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
