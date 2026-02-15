"""Tests for API v1 surface."""

from fastapi.testclient import TestClient

from dashboard.app import app


client = TestClient(app)


def test_runs_endpoint_shape():
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert "total" in data


def test_compare_endpoint_shape():
    response = client.get("/api/v1/compare")
    assert response.status_code == 200
    data = response.json()
    assert "comparisons" in data


def test_evaluate_requires_provider_or_providers():
    response = client.post("/api/v1/evaluate", json={})
    assert response.status_code == 200
    assert "error" in response.json()


def test_run_filters_endpoint_shape():
    response = client.get("/api/v1/runs/filters")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "models" in data


def test_evaluate_active_endpoint_shape():
    response = client.get("/api/v1/evaluate/active")
    assert response.status_code == 200
    assert "jobs" in response.json()


def test_canary_requires_provider():
    response = client.post("/api/v1/evaluate/canary", json={})
    assert response.status_code == 200
    assert "error" in response.json()


def test_providers_missing_config_returns_400():
    response = client.get("/api/v1/providers", params={"config": "does-not-exist.yaml"})
    assert response.status_code == 400
    assert "Config file not found" in response.json()["detail"]
