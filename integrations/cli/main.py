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


def cmd_eval(args):
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
    with httpx.Client(timeout=600) as client:
        resp = client.post(f"{_base_url(args.api_host, args.api_port)}/evaluate", json=body)
        resp.raise_for_status()
        data = resp.json()

    if "results" in data:
        for item in data["results"]:
            _print_evaluation_summary(item)
    else:
        _print_evaluation_summary(data)
        if args.fail_on_gate and not data.get("result", {}).get("quality_gate", {}).get("passed"):
            sys.exit(1)


def cmd_providers(args):
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{_base_url(args.api_host, args.api_port)}/providers", params={"config": args.config})
        resp.raise_for_status()
        data = resp.json()
    for provider in data.get("providers", []):
        status = "✓" if provider.get("available") else "✗"
        print(f"{status} {provider['name']} ({provider.get('default_model')})")


def cmd_runs(args):
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{_base_url(args.api_host, args.api_port)}/runs",
            params={"provider": args.provider, "model": args.model, "limit": args.limit, "offset": args.offset},
        )
        resp.raise_for_status()
        data = resp.json()
    print(json.dumps(data, indent=2, default=str))


def cmd_compare(args):
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{_base_url(args.api_host, args.api_port)}/compare", params={"limit": args.limit})
        resp.raise_for_status()
        data = resp.json()

    print(f"\n{'Provider/Model':<35} {'Runs':>5} {'Score':>8} {'Task':>8} {'Relev':>8} {'Halluc':>8} {'Gate':>6}")
    print("-" * 88)
    for item in data.get("comparisons", []):
        print(
            f"{item['provider_model']:<35} {item['runs']:>5} {item['overall_score']:>7.1%} "
            f"{item['task_success_score']:>7.1%} {item['relevance_score']:>7.1%} "
            f"{item['hallucination_score']:>7.1%} {item['quality_gate_pass_rate']:>5.0%}"
        )


def cmd_settings(args):
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


def cmd_dashboard(args):
    print(f"Starting dashboard on http://{args.host}:{args.port}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmq", description="LLMQ API-first CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser("eval", help="Trigger evaluation via API")
    p_eval.add_argument("--provider", "-p", required=True)
    p_eval.add_argument("--model", "-m")
    p_eval.add_argument("--dataset", "-d")
    p_eval.add_argument("--config", "-c", default="config.yaml")
    p_eval.add_argument("--workers", type=int, default=5)
    p_eval.add_argument("--timeout", type=int, default=30)
    p_eval.add_argument("--no-judge", action="store_true")
    p_eval.add_argument("--no-db", action="store_true")
    p_eval.add_argument("--non-deterministic", action="store_true")
    p_eval.add_argument("--fail-on-gate", action="store_true")

    p_providers = sub.add_parser("providers", help="List providers from API")
    p_providers.add_argument("--config", "-c", default="config.yaml")

    p_runs = sub.add_parser("runs", help="List historical runs")
    p_runs.add_argument("--provider")
    p_runs.add_argument("--model")
    p_runs.add_argument("--limit", type=int, default=50)
    p_runs.add_argument("--offset", type=int, default=0)

    p_compare = sub.add_parser("compare", help="Compare provider metrics")
    p_compare.add_argument("--limit", type=int, default=100)

    p_settings = sub.add_parser("settings", help="Get or update quality gate settings")
    p_settings.add_argument("--config", "-c", default="config.yaml")
    p_settings.add_argument("--set", help="JSON patch payload")

    p_dash = sub.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", type=int, default=8000)
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--reload", action="store_true")

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
        return
    commands[args.command](args)


if __name__ == "__main__":
    main()
