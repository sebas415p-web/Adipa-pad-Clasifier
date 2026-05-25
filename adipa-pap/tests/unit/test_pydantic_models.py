"""Tests de los contratos Pydantic de la API."""
import pytest
from pydantic import ValidationError
from src.api.models import ClassifyRequest, ClassifyResponse, FaseResult, ActoVerbalResult


def test_classify_request_valid():
    req = ClassifyRequest(
        turno_operador="Estoy aquí con usted.",
        contexto_previo="No puedo más.",
    )
    assert req.turno_operador == "Estoy aquí con usted."
    assert req.contexto_previo == "No puedo más."


def test_classify_request_empty_turno_rejected():
    with pytest.raises(ValidationError):
        ClassifyRequest(turno_operador="")


def test_classify_request_whitespace_only_rejected():
    with pytest.raises(ValidationError):
        ClassifyRequest(turno_operador="   ")


def test_classify_request_contexto_optional():
    req = ClassifyRequest(turno_operador="Estoy aquí.")
    assert req.contexto_previo == ""


def test_classify_request_none_contexto_normalized():
    req = ClassifyRequest(turno_operador="Estoy aquí.", contexto_previo=None)
    assert req.contexto_previo == ""


def test_fase_result_valid_labels():
    for label in ["A", "B", "C", "D", "E"]:
        r = FaseResult(label=label, confidence=0.9)
        assert r.label == label


def test_acto_verbal_valid_labels():
    valid = [
        "validacion", "pregunta_abierta", "pregunta_cerrada",
        "reflejo", "interpretacion", "silencio_contencion",
        "confrontacion", "directivo",
    ]
    for label in valid:
        a = ActoVerbalResult(label=label, confidence=0.8)
        assert a.label == label
