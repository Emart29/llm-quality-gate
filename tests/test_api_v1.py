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
