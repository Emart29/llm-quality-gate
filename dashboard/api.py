"""Central API layer for web-first operation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

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
    config: str = "config.yaml"
    workers: int = 5
    timeout: int = 30
    no_judge: bool = False
    no_db: bool = False
    non_deterministic: bool = False


@router.post("/evaluate")
async def evaluate(payload: EvaluateRequest) -> Dict[str, Any]:
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


@router.get("/providers")
async def list_providers(config: str = "config.yaml") -> Dict[str, Any]:
    return {"providers": service.list_providers(config)}


@router.get("/runs")
async def list_runs(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    return service.get_runs(provider=provider, model=model, limit=limit, offset=offset)


@router.get("/compare")
async def compare(limit: int = Query(100, ge=1, le=500)) -> Dict[str, Any]:
    return service.compare_metrics(limit=limit)


@router.get("/settings")
async def get_settings(config: str = "config.yaml") -> Dict[str, Any]:
    return service.get_settings(config)


@router.post("/settings")
async def update_settings(payload: Dict[str, Any], config: str = "config.yaml") -> Dict[str, Any]:
    return service.update_settings(payload, config)
