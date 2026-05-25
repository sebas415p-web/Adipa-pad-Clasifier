# ─────────────────────────────────────────────────────────────
# ADIPA PAP Classifier — Dockerfile
# Build:  docker build -t adipa-pap .
# Run:    docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... adipa-pap
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias Python primero (capa cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/

# Crear __init__.py vacíos para imports de paquete
RUN find src -type d -exec touch {}/__init__.py \;

# Usuario no-root por seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Exponer puerto
EXPOSE 8000

# Health check de Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Comando de arranque
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
