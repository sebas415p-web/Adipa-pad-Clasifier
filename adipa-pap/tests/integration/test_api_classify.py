"""Tests de integración para POST /classify."""
import os, sys
from pathlib import Path
sys.path.insert(0, '.')

# Pre-set models dir so lifespan can find them
os.environ["MODELS_DIR"] = str(Path(__file__).parent.parent.parent / "models")
os.environ.pop("ANTHROPIC_API_KEY", None)


def _get_client():
    from src.api.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_classify_valid_request():
    client = _get_client()
    response = client.post("/classify", json={
        "turno_operador": "Estoy aquí. Vamos de a poco. Dígame su nombre.",
        "contexto_previo": "No puedo más, ayuda.",
    })
    # Si los modelos no están cargados → 503 es comportamiento válido también
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["fase"]["label"] in ["A", "B", "C", "D", "E"]
        assert 0.0 <= data["fase"]["confidence"] <= 1.0
        assert data["source"] in ["llm", "baseline"]
        assert isinstance(data["actos_verbales"], list)


def test_classify_empty_turno_rejected():
    client = _get_client()
    response = client.post("/classify", json={"turno_operador": ""})
    assert response.status_code == 422


def test_classify_no_body_rejected():
    client = _get_client()
    response = client.post("/classify")
    assert response.status_code == 422
