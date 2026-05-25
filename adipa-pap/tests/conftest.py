"""
conftest.py — Fixtures compartidas para todos los tests.
"""
import pytest
from fastapi.testclient import TestClient


SAMPLE_TURNS = [
    {
        "turno_operador": "Estoy aquí. Vamos de a poco.",
        "contexto_previo": "No puedo más.",
        "expected_fase": "A",
    },
    {
        "turno_operador": "Inhale contando cuatro. Ahora suelte el aire.",
        "contexto_previo": "Me tiemblan las manos.",
        "expected_fase": "B",
    },
    {
        "turno_operador": "Tiene sentido que sienta eso. Su cuerpo todavía está en modo alarma.",
        "contexto_previo": "No entiendo por qué no me puedo calmar.",
        "expected_fase": "B",
    },
]


@pytest.fixture(scope="session")
def sample_turns():
    return SAMPLE_TURNS


@pytest.fixture(scope="session")
def test_client():
    """TestClient de FastAPI sin LLM (solo baseline)."""
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)  # forzar baseline
    from src.api.main import app
    return TestClient(app)
