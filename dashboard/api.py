"""Central API layer for web-first operation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import DEFAULT_CONFIG_NAME
from core.errors import ConfigError
from evals.service import EvaluationService

router = APIRouter(prefix="/api/v1", tags=["api-v1"])
service = EvaluationService()


class ProviderRequest(BaseModel):
    provider: str
    model: Optional[str] = None


class EvaluateRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    providers: List[ProviderRequest] = Field(default_factory=list)
    dataset: Optional[str] = None
    config: str = DEFAULT_CONFIG_NAME
    workers: int = 5
    timeout: int = 30
    no_judge: bool = False
    no_db: bool = False
    non_deterministic: bool = False


class CanaryRequest(EvaluateRequest):
    canary_ratio: float = Field(default=0.25, ge=0.05, le=1.0)
    auto_promote: bool = True


def _raise_config_error(exc: ConfigError) -> None:
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluate")
async def evaluate(payload: EvaluateRequest) -> Dict[str, Any]:
    try:
        if payload.providers:
            results = await service.evaluate_many(
                providers=[entry.model_dump() for entry in payload.providers],
                dataset_path=payload.dataset,
                config_path=payload.config,
                workers=payload.workers,
                timeout=payload.timeout,
                no_judge=payload.no_judge,
                no_db=payload.no_db,
                non_deterministic=payload.non_deterministic,
            )
            return {"results": results}

        if payload.provider:
            result = await service.evaluate_provider(
                provider=payload.provider,
                model=payload.model,
                dataset_path=payload.dataset,
                config_path=payload.config,
                workers=payload.workers,
                timeout=payload.timeout,
                no_judge=payload.no_judge,
                no_db=payload.no_db,
                non_deterministic=payload.non_deterministic,
            )
            return result

        return {"error": "Either provider or providers must be supplied."}
    except ConfigError as exc:
        _raise_config_error(exc)


@router.post("/evaluate/start")
async def start_evaluate(payload: EvaluateRequest) -> Dict[str, Any]:
    if not payload.provider:
        return {"error": "provider is required for async jobs"}
    try:
        job_id = await service.start_evaluation_job(
            provider=payload.provider,
            model=payload.model,
            dataset_path=payload.dataset,
            config_path=payload.config,
            workers=payload.workers,
            timeout=payload.timeout,
            no_judge=payload.no_judge,
            no_db=payload.no_db,
            non_deterministic=payload.non_deterministic,
        )
        return {"job_id": job_id, "status": "running"}
    except ConfigError as exc:
        _raise_config_error(exc)


@router.get("/evaluate/status/{job_id}")
async def evaluate_status(job_id: str) -> Dict[str, Any]:
    return service.get_job_status(job_id)


@router.get("/evaluate/active")
async def evaluate_active() -> Dict[str, Any]:
    return {"jobs": service.get_active_jobs()}


@router.get("/providers")
async def list_providers(config: str = DEFAULT_CONFIG_NAME) -> Dict[str, Any]:
    try:
        return {"providers": service.list_providers(config)}
    except ConfigError as exc:
        _raise_config_error(exc)


@router.get("/runs")
async def list_runs(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dataset_version: Optional[str] = None,
    commit_hash: Optional[str] = None,
    quality_gate_passed: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    return service.get_runs(
        provider=provider,
        model=model,
        dataset_version=dataset_version,
        commit_hash=commit_hash,
        quality_gate_passed=quality_gate_passed,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/filters")
async def list_run_filters() -> Dict[str, Any]:
    return service.get_run_filters()


@router.post("/runs/{run_id}/baseline")
async def mark_run_baseline(run_id: str) -> Dict[str, Any]:
    return service.mark_baseline(run_id)


@router.post("/evaluate/canary")
async def evaluate_canary(payload: CanaryRequest) -> Dict[str, Any]:
    if not payload.provider:
        return {"error": "provider is required"}
    try:
        return await service.run_canary_evaluation(
            provider=payload.provider,
            model=payload.model,
            dataset_path=payload.dataset,
            config_path=payload.config,
            canary_ratio=payload.canary_ratio,
            auto_promote=payload.auto_promote,
            workers=payload.workers,
            timeout=payload.timeout,
            no_judge=payload.no_judge,
            no_db=payload.no_db,
            non_deterministic=payload.non_deterministic,
        )
    except ConfigError as exc:
        _raise_config_error(exc)


@router.get("/compare")
async def compare(limit: int = Query(100, ge=1, le=500)) -> Dict[str, Any]:
    return service.compare_metrics(limit=limit)


@router.get("/settings")
async def get_settings(config: str = DEFAULT_CONFIG_NAME) -> Dict[str, Any]:
    try:
        return service.get_settings(config)
    except ConfigError as exc:
        _raise_config_error(exc)


@router.post("/settings")
async def update_settings(payload: Dict[str, Any], config: str = DEFAULT_CONFIG_NAME) -> Dict[str, Any]:
    try:
        return service.update_settings(payload, config)
    except ConfigError as exc:
        _raise_config_error(exc)
