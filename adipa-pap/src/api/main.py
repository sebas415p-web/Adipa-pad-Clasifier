"""
main.py — Servicio FastAPI para clasificación clínica de turnos PAP.

Endpoints:
  GET  /health   → estado del servicio
  POST /classify → clasificar un turno del operador

Estrategia de clasificación:
  1. LLM (Claude Haiku) con few-shot clínico → respuesta estructurada JSON
  2. Fallback automático a TF-IDF + LogReg si el LLM falla o tarda > 5s

Seguridad clínica:
  - Input validation con Pydantic v2
  - Logging estructurado de cada clasificación
  - Errores HTTP claros (422 para input inválido, 503 para clasificador caído)
  - Nota: detección de riesgo NO implementada en este endpoint (ver writeup)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import (
    ClassifyRequest,
    ClassifyResponse,
    FaseResult,
    ActoVerbalResult,
    HealthResponse,
    ErrorDetail,
)
from src.inference.classifier import get_classifier, PAPClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("adipa.api")

# ---------------------------------------------------------------------------
# Lifespan: carga del clasificador al inicio
# ---------------------------------------------------------------------------

_classifier: PAPClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "models"
    logger.info(f"Cargando clasificador PAP (models_dir={models_dir})…")
    try:
        _classifier = get_classifier(models_dir=models_dir)
        logger.info("Clasificador PAP listo.")
    except Exception as e:
        logger.error(f"Error al cargar clasificador: {e}")
        _classifier = None
    yield
    logger.info("Servicio PAP detenido.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ADIPA PAP Classifier",
    description=(
        "Clasificador clínico de turnos PAP: fase ABCDE + actos verbales. "
        "Primer clasificador del simulador de Primeros Auxilios Psicológicos de ADIPA."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware de logging por request
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration = (time.time() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.0f}ms)")
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Sistema"],
)
async def health():
    """Retorna el estado del servicio y disponibilidad del clasificador."""
    available = _classifier is not None
    llm_active = available and _classifier.llm is not None
    return HealthResponse(
        status="ok",
        classifier_available=available,
        llm_active=llm_active,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Clasificar un turno del operador PAP",
    tags=["Clasificación"],
    responses={
        422: {"model": ErrorDetail, "description": "Input inválido"},
        503: {"model": ErrorDetail, "description": "Clasificador no disponible"},
    },
)
async def classify(request: ClassifyRequest):
    """
    Clasifica un turno del operador en:
    - **Fase ABCDE**: en qué fase del modelo PAP se ubica.
    - **Actos verbales**: qué técnica(s) clínica(s) usa (multi-label).

    El campo `source` indica si usó LLM ('llm') o el fallback baseline ('baseline').
    """
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Clasificador no disponible. Verifique que los modelos estén cargados.",
        )

    try:
        result = _classifier.classify(
            turno_operador=request.turno_operador,
            contexto_previo=request.contexto_previo or "",
        )
    except RuntimeError as e:
        logger.error(f"Clasificador falló completamente: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Error inesperado al clasificar: {e}")
        raise HTTPException(status_code=500, detail="Error interno del clasificador.")

    # Construir response tipado
    fase = FaseResult(
        label=result["fase"]["label"],
        confidence=result["fase"]["confidence"],
        razon=result["fase"].get("razon"),
    )

    actos = []
    for a in result.get("actos_verbales", []):
        try:
            actos.append(ActoVerbalResult(label=a["label"], confidence=a["confidence"]))
        except Exception:
            # Ignorar actos con etiqueta inválida (puede pasar con el LLM)
            logger.warning(f"Acto verbal con etiqueta inválida ignorado: {a}")

    return ClassifyResponse(
        fase=fase,
        actos_verbales=actos,
        source=result.get("source", "baseline"),
        latency_ms=result.get("latency_ms", 0),
    )


# ---------------------------------------------------------------------------
# Handler de errores de validación Pydantic
# ---------------------------------------------------------------------------

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Input inválido", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entrypoint local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )
