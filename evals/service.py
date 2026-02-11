"""Evaluation engine exposed as an async service for API and CLI consumers."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import smtplib
import subprocess
import uuid
from pathlib import Path
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

import yaml

from evals.comprehensive_runner import ComprehensiveEvaluationRunner
from evals.dataset import DatasetLoader
from evals.metrics import MetricsEngine
from evals.runner import EvaluationRunner
from llm.factory import BaseLLM, LLMFactory
from storage.database import Database
from storage.repository import EvaluationRepository
from storage.models import BaselineMetric

logger = logging.getLogger(__name__)


class EvaluationService:
    """Orchestrates evaluations and storage with web/API-first semantics."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._alert_handlers = [self._send_slack_alert, self._send_webhook_alert, self._send_email_alert]

    def _get_git_hash(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.project_root),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return ""

    def _get_git_branch(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.project_root),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return ""

    def list_providers(self, config_path: str = "config.yaml") -> List[Dict[str, Any]]:
        config = LLMFactory.load_config(config_path)
        availability = LLMFactory.list_available_providers(config=config)
        providers = config.get("providers", {})

        return [
            {
                "name": name,
                "available": availability.get(name, False),
                "default_model": providers.get(name, {}).get("model"),
            }
            for name in sorted(providers.keys())
        ]

    def _regression_delta(self, metric: str, baseline: Optional[float], current: Optional[float]) -> float:
        if baseline is None or current is None:
            return 0.0
        if metric == "hallucination_score":
            return baseline - current
        return current - baseline

    def _compare_against_baseline(
        self,
        repo: EvaluationRepository,
        result_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        provider = result_dict.get("provider_name")
        model = result_dict.get("model_name")
        dataset_version = result_dict.get("dataset_version")
        baseline = repo.get_baseline(provider, model, dataset_version)
        if not baseline:
            return {"regression_detected": False, "reason": "no_baseline"}

        tolerance = float(os.getenv("LLMQ_REGRESSION_TOLERANCE", "0.03"))
        metric_keys = [
            "overall_score",
            "task_success_score",
            "relevance_score",
            "hallucination_score",
            "consistency_score",
        ]
        deltas = {
            key: self._regression_delta(key, baseline.get(key), result_dict.get(key))
            for key in metric_keys
        }
        regressions = [
            key for key, delta in deltas.items()
            if key != "hallucination_score" and delta < -tolerance
        ]
        regressions += [
            key for key, delta in deltas.items()
            if key == "hallucination_score" and delta < -tolerance
        ]

        return {
            "regression_detected": len(regressions) > 0,
            "baseline_id": baseline.get("id"),
            "baseline_commit_hash": baseline.get("commit_hash"),
            "tolerance": tolerance,
            "deltas": deltas,
            "regressions": regressions,
        }

    def _send_slack_alert(self, payload: Dict[str, Any]) -> None:
        webhook = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook:
            return
        import httpx
        httpx.post(webhook, json={"text": payload.get("message", "LLMQ alert")}, timeout=10)

    def _send_webhook_alert(self, payload: Dict[str, Any]) -> None:
        webhook = os.getenv("LLMQ_ALERT_WEBHOOK_URL")
        if not webhook:
            return
        import httpx
        httpx.post(webhook, json=payload, timeout=10)

    def _send_email_alert(self, payload: Dict[str, Any]) -> None:
        to_addr = os.getenv("ALERT_EMAIL_TO")
        smtp_host = os.getenv("ALERT_SMTP_HOST")
        smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        smtp_user = os.getenv("ALERT_SMTP_USER")
        smtp_password = os.getenv("ALERT_SMTP_PASSWORD")
        if not to_addr or not smtp_host:
            return

        msg = MIMEText(json.dumps(payload, indent=2), "plain")
        msg["Subject"] = f"LLMQ Alert: {payload.get('status', 'UNKNOWN')}"
        msg["From"] = smtp_user or "llmq@example.local"
        msg["To"] = to_addr

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)

    def _dispatch_alert(self, result: Dict[str, Any]) -> None:
        gate = result.get("quality_gate", {})
        regression = result.get("regression", {})
        should_alert = not gate.get("passed", True) or regression.get("regression_detected", False)
        if not should_alert:
            return
        payload = {
            "status": "FAILED" if not gate.get("passed", True) else "REGRESSION",
            "provider": result.get("provider_name"),
            "model": result.get("model_name"),
            "commit_hash": self._get_git_hash(),
            "failed_metrics": gate.get("failed_metrics", []),
            "regressions": regression.get("regressions", []),
            "message": (
                f"LLMQ alert for {result.get('provider_name')}/{result.get('model_name')} "
                f"on {self._get_git_hash()}: gate_passed={gate.get('passed')} "
                f"regression_detected={regression.get('regression_detected')}"
            ),
        }
        for handler in self._alert_handlers:
            try:
                handler(payload)
            except Exception as exc:
                logger.warning("Alert handler failed: %s", exc)

    async def evaluate_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        dataset_path: Optional[str] = None,
        config_path: str = "config.yaml",
        workers: int = 5,
        timeout: int = 30,
        no_judge: bool = False,
        non_deterministic: bool = False,
        no_db: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        config = LLMFactory.load_config(config_path)
        resolved_model = model or config.get("providers", {}).get(provider, {}).get("model", "")
        source_dataset = dataset_path or str(self.project_root / "evals" / "dataset.json")

        dataset = DatasetLoader.load_from_file(source_dataset)
        factory = LLMFactory(config_path)

        judge_llm = None
        if not no_judge:
            try:
                judge_provider = LLMFactory.create_provider_for_role("judge", config=config)
                judge_llm = BaseLLM(judge_provider)
            except Exception:
                judge_llm = None

        metrics_engine = MetricsEngine(judge_llm=judge_llm, thresholds=config.get("quality_gates", {}))
        runner = EvaluationRunner(
            llm_factory=factory,
            max_workers=workers,
            timeout_seconds=timeout,
            deterministic=not non_deterministic,
        )
        comprehensive = ComprehensiveEvaluationRunner(runner, metrics_engine)

        result = await comprehensive.run(
            dataset,
            provider,
            resolved_model,
            progress_callback=progress_callback,
        )
        result_dict = result.to_dict()
        gate = result_dict.get("quality_gate", {})
        agg = result_dict.get("aggregated_metrics", {})
        result_dict["overall_score"] = gate.get("overall_score")
        result_dict["task_success_score"] = agg.get("task_success", {}).get("score")
        result_dict["relevance_score"] = agg.get("relevance", {}).get("score")
        result_dict["hallucination_score"] = agg.get("hallucination", {}).get("score")
        result_dict["consistency_score"] = agg.get("consistency", {}).get("score")

        run_id = None
        regression = {"regression_detected": False, "reason": "db_disabled"}
        if not no_db:
            db = Database()
            repo = EvaluationRepository(db)
            regression = self._compare_against_baseline(repo, result_dict)
            result_dict["regression"] = regression
            run_id = repo.save_comprehensive_result(
                result_dict,
                commit_hash=self._get_git_hash(),
                branch=self._get_git_branch(),
                regression=regression,
            )
            db.close()
        else:
            result_dict["regression"] = regression

        self._dispatch_alert(result_dict)

        return {
            "run_id": run_id,
            "provider": provider,
            "model": resolved_model,
            "result": result_dict,
        }

    async def run_canary_evaluation(
        self,
        provider: str,
        model: Optional[str] = None,
        dataset_path: Optional[str] = None,
        config_path: str = "config.yaml",
        canary_ratio: float = 0.25,
        auto_promote: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        source_dataset = dataset_path or str(self.project_root / "evals" / "dataset.json")
        full_dataset = DatasetLoader.load_from_file(source_dataset)
        canary_size = max(1, int(len(full_dataset.test_cases) * canary_ratio))
        canary_dataset = copy.deepcopy(full_dataset)
        canary_dataset.test_cases = full_dataset.test_cases[:canary_size]

        config = LLMFactory.load_config(config_path)
        resolved_model = model or config.get("providers", {}).get(provider, {}).get("model", "")
        factory = LLMFactory(config_path)
        runner = EvaluationRunner(
            llm_factory=factory,
            max_workers=kwargs.get("workers", 5),
            timeout_seconds=kwargs.get("timeout", 30),
            deterministic=not kwargs.get("non_deterministic", False),
        )
        metrics_engine = MetricsEngine(thresholds=config.get("quality_gates", {}))
        comprehensive = ComprehensiveEvaluationRunner(runner, metrics_engine)
        canary_result = (await comprehensive.run(canary_dataset, provider, resolved_model)).to_dict()

        response = {
            "provider": provider,
            "model": resolved_model,
            "canary": {
                "size": canary_size,
                "total": len(full_dataset.test_cases),
                "ratio": canary_ratio,
                "result": canary_result,
            },
            "promoted": False,
            "full_result": None,
        }

        if auto_promote and canary_result.get("quality_gate", {}).get("passed"):
            response["promoted"] = True
            response["full_result"] = await self.evaluate_provider(
                provider=provider,
                model=model,
                dataset_path=dataset_path,
                config_path=config_path,
                **kwargs,
            )
        return response

    def mark_baseline(self, run_id: str) -> Dict[str, Any]:
        db = Database()
        repo = EvaluationRepository(db)
        run = repo.get_run(run_id)
        if not run:
            db.close()
            return {"error": "run_not_found", "run_id": run_id}

        baseline = BaselineMetric(
            id=str(uuid.uuid4()),
            provider_name=run["provider_name"],
            model_name=run["model_name"],
            dataset_version=run["dataset_version"],
            overall_score=run.get("overall_score"),
            task_success_score=run.get("task_success_score"),
            relevance_score=run.get("relevance_score"),
            hallucination_score=run.get("hallucination_score"),
            consistency_score=run.get("consistency_score"),
            source_run_id=run_id,
            commit_hash=run.get("commit_hash"),
        )
        baseline_id = repo.upsert_baseline(baseline)
        db.close()
        return {"baseline_id": baseline_id, "run_id": run_id}

    async def evaluate_many(self, providers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        tasks = [
            self.evaluate_provider(
                provider=entry["provider"],
                model=entry.get("model"),
                **kwargs,
            )
            for entry in providers
        ]
        return await asyncio.gather(*tasks)

    async def start_evaluation_job(
        self,
        provider: str,
        model: Optional[str] = None,
        dataset_path: Optional[str] = None,
        config_path: str = "config.yaml",
        workers: int = 5,
        timeout: int = 30,
        no_judge: bool = False,
        non_deterministic: bool = False,
        no_db: bool = False,
    ) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "provider": provider,
            "model": model,
            "progress": 0,
            "total": 0,
            "result": None,
            "error": None,
        }

        def on_progress(completed: int, total: int) -> None:
            self._jobs[job_id]["progress"] = completed
            self._jobs[job_id]["total"] = total

        async def _run() -> None:
            try:
                result = await self.evaluate_provider(
                    provider=provider,
                    model=model,
                    dataset_path=dataset_path,
                    config_path=config_path,
                    workers=workers,
                    timeout=timeout,
                    no_judge=no_judge,
                    non_deterministic=non_deterministic,
                    no_db=no_db,
                    progress_callback=on_progress,
                )
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = result
            except Exception as exc:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)

        asyncio.create_task(_run())
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self._jobs.get(job_id, {"job_id": job_id, "status": "not_found"})

    def get_active_jobs(self) -> List[Dict[str, Any]]:
        return [
            job for job in self._jobs.values() if job.get("status") == "running"
        ]

    def get_runs(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dataset_version: Optional[str] = None,
        commit_hash: Optional[str] = None,
        quality_gate_passed: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        db = Database()
        repo = EvaluationRepository(db)
        runs = repo.list_runs(
            provider=provider,
            model=model,
            dataset_version=dataset_version,
            commit_hash=commit_hash,
            quality_gate_passed=quality_gate_passed,
            limit=limit,
            offset=offset,
        )
        db.close()
        return {"runs": runs, "total": len(runs)}

    def get_run_filters(self) -> Dict[str, Any]:
        db = Database()
        repo = EvaluationRepository(db)
        filters = repo.list_run_filters()
        db.close()
        return filters

    def compare_metrics(self, limit: int = 100) -> Dict[str, Any]:
        db = Database()
        repo = EvaluationRepository(db)
        runs = repo.list_runs(limit=limit)
        db.close()

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for run in runs:
            key = f"{run['provider_name']}/{run['model_name']}"
            groups.setdefault(key, []).append(run)

        comparisons = []
        for key, group in groups.items():
            n = len(group)
            avg = lambda field: sum(r.get(field, 0) or 0 for r in group) / n
            comparisons.append(
                {
                    "provider_model": key,
                    "runs": n,
                    "overall_score": avg("overall_score"),
                    "task_success_score": avg("task_success_score"),
                    "relevance_score": avg("relevance_score"),
                    "hallucination_score": avg("hallucination_score"),
                    "quality_gate_pass_rate": sum(1 for r in group if r.get("quality_gate_passed")) / n,
                }
            )

        comparisons.sort(key=lambda item: item["overall_score"], reverse=True)
        return {"comparisons": comparisons}

    def get_settings(self, config_path: str = "config.yaml") -> Dict[str, Any]:
        return LLMFactory.load_config(config_path)

    def update_settings(self, payload: Dict[str, Any], config_path: str = "config.yaml") -> Dict[str, Any]:
        config = LLMFactory.load_config(config_path)
        config.update(payload)
        with open(config_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(config, fp, sort_keys=False)
        return config
