"""Tests de integración para GET /health."""
from fastapi.testclient import TestClient
import os, sys
sys.path.insert(0, '.')


def test_health_returns_200():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from src.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_health_schema():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from src.api.main import app
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "classifier_available" in data
    assert "llm_active" in data


def test_health_llm_inactive_without_api_key():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from src.api.main import app
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["llm_active"] is False
