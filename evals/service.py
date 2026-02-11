"""Evaluation engine exposed as an async service for API and CLI consumers."""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from evals.comprehensive_runner import ComprehensiveEvaluationRunner
from evals.dataset import DatasetLoader
from evals.metrics import MetricsEngine
from evals.runner import EvaluationRunner
from llm.factory import BaseLLM, LLMFactory
from storage.database import Database
from storage.repository import EvaluationRepository


class EvaluationService:
    """Orchestrates evaluations and storage with web/API-first semantics."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._jobs: Dict[str, Dict[str, Any]] = {}

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

        run_id = None
        if not no_db:
            db = Database()
            repo = EvaluationRepository(db)
            run_id = repo.save_comprehensive_result(
                result_dict,
                commit_hash=self._get_git_hash(),
                branch=self._get_git_branch(),
            )
            db.close()

        return {
            "run_id": run_id,
            "provider": provider,
            "model": resolved_model,
            "result": result_dict,
        }

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
