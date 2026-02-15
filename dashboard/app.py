"""FastAPI backend for the LLM Quality Gate dashboard."""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from storage.database import Database
from storage.repository import EvaluationRepository
from dashboard.api import router as api_v1_router

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Quality Gate Dashboard", version="0.1.1")

app.include_router(api_v1_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Database singleton (lazy)
_db: Optional[Database] = None
_repo: Optional[EvaluationRepository] = None


def get_repo() -> EvaluationRepository:
    global _db, _repo
    if _repo is None:
        db_path = os.getenv("LLMQG_DB_PATH", "llmqg.duckdb")
        _db = Database(db_path)
        _repo = EvaluationRepository(_db)
    return _repo


# ── Pages ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── API endpoints ─────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dataset_version: Optional[str] = None,
    commit_hash: Optional[str] = None,
    quality_gate_passed: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    repo = get_repo()
    runs = repo.list_runs(
        provider=provider,
        model=model,
        dataset_version=dataset_version,
        commit_hash=commit_hash,
        quality_gate_passed=quality_gate_passed,
        limit=limit,
        offset=offset,
    )
    return {"runs": runs, "total": len(runs)}




@app.get("/api/run-filters")
async def run_filters():
    repo = get_repo()
    return repo.list_run_filters()

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    repo = get_repo()
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    test_cases = repo.get_test_case_results(run_id)
    return {"run": run, "test_cases": test_cases}


@app.get("/api/history")
async def get_history(
    provider: str = Query(...),
    model: str = Query(...),
    limit: int = Query(30, ge=1, le=100),
):
    repo = get_repo()
    history = repo.get_run_history(provider, model, limit=limit)
    return {"history": history}


@app.get("/api/quality-gates")
async def quality_gate_history(limit: int = Query(50, ge=1, le=200)):
    repo = get_repo()
    gates = repo.get_quality_gate_history(limit=limit)
    return {"gates": gates}


@app.get("/api/comparisons")
async def list_comparisons(limit: int = Query(20, ge=1, le=100)):
    repo = get_repo()
    comparisons = repo.get_comparisons(limit=limit)
    return {"comparisons": comparisons}


@app.get("/api/overview")
async def dashboard_overview():
    """Aggregate stats for the overview dashboard."""
    repo = get_repo()
    runs = repo.list_runs(limit=200)

    if not runs:
        return {
            "total_runs": 0,
            "avg_success_rate": 0,
            "avg_task_success": 0,
            "avg_relevance": 0,
            "avg_hallucination": 0,
            "quality_gate_pass_rate": 0,
            "providers": [],
            "recent_runs": [],
        }

    total = len(runs)
    avg_sr = sum(r.get("success_rate", 0) or 0 for r in runs) / total
    avg_ts = sum(r.get("task_success_score", 0) or 0 for r in runs) / total
    avg_rel = sum(r.get("relevance_score", 0) or 0 for r in runs) / total
    avg_hal = sum(r.get("hallucination_score", 0) or 0 for r in runs) / total
    avg_cons = sum(r.get("consistency_score", 0) or 0 for r in runs) / total
    gate_pass = sum(1 for r in runs if r.get("quality_gate_passed")) / total

    # Provider summary
    provider_data: dict = {}
    for r in runs:
        key = f"{r['provider_name']}/{r['model_name']}"
        if key not in provider_data:
            provider_data[key] = {"runs": 0, "total_score": 0}
        provider_data[key]["runs"] += 1
        provider_data[key]["total_score"] += r.get("overall_score", 0) or 0

    providers = [
        {"name": k, "runs": v["runs"], "avg_score": v["total_score"] / v["runs"]}
        for k, v in provider_data.items()
    ]

    return {
        "total_runs": total,
        "avg_success_rate": round(avg_sr, 3),
        "avg_task_success": round(avg_ts, 3),
        "avg_relevance": round(avg_rel, 3),
        "avg_hallucination": round(avg_hal, 3),
        "avg_consistency": round(avg_cons, 3),
        "quality_gate_pass_rate": round(gate_pass, 3),
        "providers": providers,
        "recent_runs": runs[:10],
    }
