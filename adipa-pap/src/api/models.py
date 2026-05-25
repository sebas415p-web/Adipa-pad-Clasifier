"""
models.py
---------
Contratos de entrada/salida de la API PAP con Pydantic v2.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Constantes de dominio
# ---------------------------------------------------------------------------

FaseLabel = Literal["A", "B", "C", "D", "E"]

ActoVerbalLabel = Literal[
    "validacion",
    "pregunta_abierta",
    "pregunta_cerrada",
    "reflejo",
    "interpretacion",
    "silencio_contencion",
    "confrontacion",
    "directivo",
]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    turno_operador: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Turno del operador (profesional PAP) a clasificar.",
        examples=["Entiendo. Puede quedarse en el suelo. ¿Está herida físicamente?"],
    )
    contexto_previo: Optional[str] = Field(
        default="",
        max_length=2000,
        description="Turno previo del paciente (contexto conversacional).",
        examples=["No puedo... no puedo sacármelo de la cabeza."],
    )

    @field_validator("turno_operador")
    @classmethod
    def no_empty_turno(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("turno_operador no puede ser vacío o solo espacios.")
        return v.strip()

    @field_validator("contexto_previo", mode="before")
    @classmethod
    def normalize_contexto(cls, v) -> str:
        if v is None:
            return ""
        return str(v).strip()


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class FaseResult(BaseModel):
    label: FaseLabel = Field(..., description="Fase ABCDE del modelo PAP.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza [0,1].")
    razon: Optional[str] = Field(
        default=None,
        description="Justificación clínica breve (disponible en modo LLM).",
    )


class ActoVerbalResult(BaseModel):
    label: ActoVerbalLabel = Field(..., description="Acto verbal detectado.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza [0,1].")


class ClassifyResponse(BaseModel):
    fase: FaseResult
    actos_verbales: list[ActoVerbalResult] = Field(
        default_factory=list,
        description="Lista de actos verbales detectados (multi-label).",
    )
    source: Literal["llm", "baseline"] = Field(
        ...,
        description="Clasificador que generó la respuesta ('llm' o 'baseline').",
    )
    latency_ms: int = Field(
        ...,
        ge=0,
        description="Latencia del clasificador en milisegundos.",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    classifier_available: bool = Field(
        ...,
        description="True si el clasificador (LLM o baseline) está disponible.",
    )
    llm_active: bool = Field(
        ...,
        description="True si el clasificador LLM está activo (vs. solo baseline).",
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    error: str
    detail: Optional[str] = None
