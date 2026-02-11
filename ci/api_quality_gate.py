"""CI quality gate runner that uses API-first evaluation endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git_short_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return ""


def wait_for_api(base_url: str, timeout_s: int = 45) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = httpx.get(f"{base_url}/providers", timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("API did not become ready in time")


def evaluate(base_url: str, provider: str, model: str | None, config: str, dataset: str | None) -> dict:
    body = {
        "provider": provider,
        "model": model,
        "config": config,
        "dataset": dataset,
        "workers": 5,
        "timeout": 30,
        "no_db": False,
    }
    with httpx.Client(timeout=1200) as client:
        resp = client.post(f"{base_url}/evaluate", json=body)
        resp.raise_for_status()
        return resp.json()


def enforce_thresholds(result: dict) -> tuple[bool, list[dict]]:
    metrics = (result.get("result") or {}).get("aggregated_metrics", {})
    failures = []
    for metric, values in metrics.items():
        score = float(values.get("score", 0))
        threshold = values.get("threshold")
        if threshold is None:
            continue
        if metric == "hallucination":
            ok = score <= float(threshold)
        else:
            ok = score >= float(threshold)
        if not ok:
            failures.append({"metric": metric, "score": score, "threshold": threshold})
    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--output", default="evaluation_results/ci_result.json")
    args = parser.parse_args()

    wait_for_api(args.api_base)
    response = evaluate(args.api_base, args.provider, args.model, args.config, args.dataset)
    passed, failures = enforce_thresholds(response)

    payload = {
        "commit_hash": git_short_hash(),
        "provider": response.get("provider"),
        "model": response.get("model"),
        "run_id": response.get("run_id"),
        "quality_gate": response.get("result", {}).get("quality_gate", {}),
        "aggregated_metrics": response.get("result", {}).get("aggregated_metrics", {}),
        "regression": response.get("result", {}).get("regression", {}),
        "threshold_failures": failures,
        "passed": passed and response.get("result", {}).get("quality_gate", {}).get("passed", False),
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not payload["passed"]:
        print("Quality gate failed.")
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
