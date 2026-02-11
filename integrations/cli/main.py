"""LLMQ CLI integration client (API-first)."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def configure_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/api/v1"


def _print_evaluation_summary(result: dict):
    payload = result.get("result", {})
    gate = payload.get("quality_gate", {})
    print(f"\nProvider/Model: {result.get('provider')}/{result.get('model')}")
    print(f"Quality Gate: {'PASSED' if gate.get('passed') else 'FAILED'}")
    print(f"Overall Score: {gate.get('overall_score', 0):.2%}")
    print(f"Run ID: {result.get('run_id')}")


def _handle_api_error(e: Exception, operation: str) -> None:
    """Handle API errors with developer-friendly messages."""
    if isinstance(e, httpx.ConnectError):
        print(f"❌ Cannot connect to LLMQ API server.")
        print(f"💡 Make sure the dashboard is running: llmq dashboard")
        print(f"   Or check if the server is running on a different port.")
    elif isinstance(e, httpx.TimeoutException):
        print(f"❌ Request timed out while {operation}.")
        print(f"💡 The evaluation might be taking longer than expected.")
        print(f"   Try increasing timeout with --timeout or check server logs.")
    elif isinstance(e, httpx.HTTPStatusError):
        print(f"❌ API error ({e.response.status_code}) while {operation}:")
        try:
            error_detail = e.response.json()
            print(f"   {error_detail.get('detail', 'Unknown error')}")
        except:
            print(f"   {e.response.text}")
        
        if e.response.status_code == 422:
            print(f"💡 Check your request parameters and try again.")
        elif e.response.status_code == 500:
            print(f"💡 Server error - check the dashboard logs for details.")
    else:
        print(f"❌ Unexpected error while {operation}: {e}")
        print(f"💡 Check your network connection and API server status.")
    sys.exit(1)


def cmd_eval(args):
    """Run LLM evaluation with quality gate enforcement."""
    body = {
        "provider": args.provider,
        "model": args.model,
        "dataset": args.dataset,
        "config": args.config,
        "workers": args.workers,
        "timeout": args.timeout,
        "no_judge": args.no_judge,
        "no_db": args.no_db,
        "non_deterministic": args.non_deterministic,
    }
    
    try:
        with httpx.Client(timeout=600) as client:
            resp = client.post(f"{_base_url(args.api_host, args.api_port)}/evaluate", json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "running evaluation")

    if "results" in data:
        for item in data["results"]:
            _print_evaluation_summary(item)
    else:
        _print_evaluation_summary(data)
        if args.fail_on_gate and not data.get("result", {}).get("quality_gate", {}).get("passed"):
            print("\n❌ Quality gate failed - exiting with error code 1")
            sys.exit(1)


def cmd_providers(args):
    """List available LLM providers and their status."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{_base_url(args.api_host, args.api_port)}/providers", params={"config": args.config})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching providers")
        
    print("\nAvailable LLM Providers:")
    print("-" * 40)
    for provider in data.get("providers", []):
        status = "✅ Available" if provider.get("available") else "❌ Not configured"
        print(f"{provider['name']:<15} {status}")
        print(f"{'':>15} Model: {provider.get('default_model', 'N/A')}")
        print()


def cmd_runs(args):
    """List historical evaluation runs."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{_base_url(args.api_host, args.api_port)}/runs",
                params={"provider": args.provider, "model": args.model, "limit": args.limit, "offset": args.offset},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching runs")
        
    print(json.dumps(data, indent=2, default=str))


def cmd_compare(args):
    """Compare performance across different LLM providers."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{_base_url(args.api_host, args.api_port)}/compare", params={"limit": args.limit})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching comparison data")

    print(f"\n{'Provider/Model':<35} {'Runs':>5} {'Score':>8} {'Task':>8} {'Relev':>8} {'Halluc':>8} {'Gate':>6}")
    print("-" * 88)
    for item in data.get("comparisons", []):
        print(
            f"{item['provider_model']:<35} {item['runs']:>5} {item['overall_score']:>7.1%} "
            f"{item['task_success_score']:>7.1%} {item['relevance_score']:>7.1%} "
            f"{item['hallucination_score']:>7.1%} {item['quality_gate_pass_rate']:>5.0%}"
        )


def cmd_settings(args):
    """Get or update quality gate settings."""
    try:
        with httpx.Client(timeout=30) as client:
            if args.set:
                payload = json.loads(args.set)
                resp = client.post(
                    f"{_base_url(args.api_host, args.api_port)}/settings",
                    params={"config": args.config},
                    json=payload,
                )
            else:
                resp = client.get(f"{_base_url(args.api_host, args.api_port)}/settings", params={"config": args.config})
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2))
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in --set parameter: {e}")
        example_json = '{"quality_gates": {"task_success_threshold": 0.9}}'
        print(f"💡 Example: --set '{example_json}'")
        sys.exit(1)
    except Exception as e:
        _handle_api_error(e, "updating settings")


def cmd_dashboard(args):
    """Launch the LLMQ web dashboard and API server."""
    print(f"🚀 Starting LLMQ dashboard on http://{args.host}:{args.port}")
    print(f"📊 Dashboard: http://{args.host}:{args.port}")
    print(f"🔌 API: http://{args.host}:{args.port}/api/v1")
    print(f"📖 API Docs: http://{args.host}:{args.port}/docs")
    print()
    
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "dashboard.app:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--reload" if args.reload else "--no-access-log",
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except FileNotFoundError:
        print("❌ uvicorn not found. Install with: pip install uvicorn")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmq", 
        description="LLMQ - LLM Quality Gate CLI\n\nA provider-agnostic evaluation framework for LLM applications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmq dashboard                          # Start web dashboard
  llmq providers                          # List available providers  
  llmq eval --provider groq               # Run evaluation with Groq
  llmq eval --provider openai --fail-on-gate  # Fail if quality gate fails
  llmq compare                            # Compare provider performance
  
For more help: https://github.com/Emart29/llm-quality-gate
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--api-host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port (default: 8000)")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluation command
    p_eval = sub.add_parser("eval", help="Run LLM evaluation with quality gates", 
                           description="Evaluate an LLM provider against your test dataset")
    p_eval.add_argument("--provider", "-p", required=True, 
                       help="LLM provider to evaluate (e.g., groq, openai, claude)")
    p_eval.add_argument("--model", "-m", 
                       help="Specific model to use (defaults to provider's default)")
    p_eval.add_argument("--dataset", "-d", 
                       help="Path to evaluation dataset (defaults to evals/dataset.json)")
    p_eval.add_argument("--config", "-c", default="config.yaml", 
                       help="Configuration file path (default: config.yaml)")
    p_eval.add_argument("--workers", type=int, default=5, 
                       help="Number of parallel workers (default: 5)")
    p_eval.add_argument("--timeout", type=int, default=30, 
                       help="Request timeout in seconds (default: 30)")
    p_eval.add_argument("--no-judge", action="store_true", 
                       help="Skip LLM-as-judge metrics (faster, less accurate)")
    p_eval.add_argument("--no-db", action="store_true", 
                       help="Skip saving results to database")
    p_eval.add_argument("--non-deterministic", action="store_true", 
                       help="Allow non-deterministic evaluation (temperature > 0)")
    p_eval.add_argument("--fail-on-gate", action="store_true", 
                       help="Exit with error code 1 if quality gate fails (useful for CI)")

    # Providers command
    p_providers = sub.add_parser("providers", help="List available LLM providers",
                                description="Show all configured providers and their availability status")
    p_providers.add_argument("--config", "-c", default="config.yaml", 
                            help="Configuration file path (default: config.yaml)")

    # Runs command  
    p_runs = sub.add_parser("runs", help="List historical evaluation runs",
                           description="View past evaluation results and performance history")
    p_runs.add_argument("--provider", help="Filter by provider name")
    p_runs.add_argument("--model", help="Filter by model name")
    p_runs.add_argument("--limit", type=int, default=50, help="Maximum number of runs to return (default: 50)")
    p_runs.add_argument("--offset", type=int, default=0, help="Number of runs to skip (default: 0)")

    # Compare command
    p_compare = sub.add_parser("compare", help="Compare provider performance",
                              description="Compare metrics and quality gate pass rates across providers")
    p_compare.add_argument("--limit", type=int, default=100, 
                          help="Maximum number of runs to include in comparison (default: 100)")

    # Settings command
    p_settings = sub.add_parser("settings", help="Manage quality gate settings",
                               description="View or update quality gate thresholds and configuration")
    p_settings.add_argument("--config", "-c", default="config.yaml", 
                           help="Configuration file path (default: config.yaml)")
    p_settings.add_argument("--set", help='Update settings with JSON (e.g., \'{"quality_gates": {"task_success_threshold": 0.9}}\')')

    # Dashboard command
    p_dash = sub.add_parser("dashboard", help="Launch web dashboard and API server",
                           description="Start the LLMQ web interface and REST API")
    p_dash.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    p_dash.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p_dash.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    commands = {
        "eval": cmd_eval,
        "providers": cmd_providers,
        "runs": cmd_runs,
        "compare": cmd_compare,
        "settings": cmd_settings,
        "dashboard": cmd_dashboard,
    }

    if not args.command:
        parser.print_help()
        print("\n💡 Start with: llmq dashboard")
        return
        
    commands[args.command](args)


if __name__ == "__main__":
    main()
