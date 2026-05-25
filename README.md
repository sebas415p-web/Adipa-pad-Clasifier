# ADIPA PAP Classifier

Clasificador clínico de turnos PAP (Primeros Auxilios Psicológicos) para el simulador de ADIPA Lab.

Clasifica cada turno del operador en:
- **Fase ABCDE**: en qué etapa del protocolo PAP se ubica
- **Actos verbales**: qué técnica(s) clínica(s) usa (multi-label)

---

## Arquitectura de clasificación

```
                      POST /classify
                           │
               ┌───────────▼────────────┐
               │      PAPClassifier     │
               └───────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   1. LLM few-shot        │  ← Claude Haiku + 5 ejemplos clínicos
              │      (primario)          │     ~800–1200 ms | ~$0.002/turno
              └────────────┬────────────┘
                           │ si falla / timeout
              ┌────────────▼────────────┐
              │   2. TF-IDF + LogReg     │  ← Baseline sklearn
              │      (fallback)          │     <5 ms | $0
              └─────────────────────────┘
```

**¿Por qué LLM-first?** El baseline entrenado muestra gap train/held-out de 37 puntos de accuracy (99% → 62%). Con solo 10 guiones escritos por el mismo equipo, el modelo memoriza léxico específico de cada caso. El LLM clasifica por comprensión semántica del español clínico, no por co-ocurrencia léxica. Ver `writeup.md` para justificación completa.

---

## Estructura del repositorio

```
adipa-pap/
├── data/
│   ├── raw/                    # guiones.txt extraído del PDF
│   ├── processed/              # dataset_turnos.csv / .json
│   └── evaluation/             # métricas, matrices de confusión
├── models/                     # phase_model.pkl, verbal_model.pkl, mlb.pkl
├── notebooks/                  # exploración (opcional)
├── src/
│   ├── preprocessing/
│   │   └── extract_dataset.py  # parser de guiones → dataset estructurado
│   ├── training/
│   │   └── train_baseline.py   # TF-IDF + LogReg
│   ├── inference/
│   │   └── classifier.py       # PAPClassifier (LLM + fallback)
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   └── models.py           # Pydantic v2 contracts
│   └── evaluation/
│       └── evaluate.py         # arnés de evaluación reproducible
├── Dockerfile
├── requirements.txt
├── writeup.md
└── README.md
```

---

## Setup rápido

### Opción 1: Docker (recomendado)

```bash
# Construir
docker build -t adipa-pap .

# Ejecutar (con LLM activo)
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  adipa-pap

# Ejecutar (solo baseline, sin LLM)
docker run -p 8000:8000 adipa-pap
```

### Opción 2: Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Exportar API key (opcional — activa LLM; sin ella usa solo baseline)
export ANTHROPIC_API_KEY=sk-ant-...

# Iniciar servicio
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Si es la primera vez, regenerar modelos baseline primero:
python src/preprocessing/extract_dataset.py
python src/training/train_baseline.py
```

---

## Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "classifier_available": true,
  "llm_active": true
}
```

### `POST /classify`

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "turno_operador": "Tiene sentido que sienta eso. Su cuerpo todavía está en modo alarma.",
    "contexto_previo": "No entiendo por qué no puedo calmarme si ya estoy en casa."
  }'
```

```json
{
  "fase": {
    "label": "B",
    "confidence": 0.94,
    "razon": "El operador valida la respuesta emocional y explica el mecanismo fisiológico (interpretación), ubicándose en la fase de regulación emocional."
  },
  "actos_verbales": [
    { "label": "validacion",     "confidence": 0.91 },
    { "label": "interpretacion", "confidence": 0.87 }
  ],
  "source": "llm",
  "latency_ms": 923
}
```

**Errores:**
- `422` → input inválido (turno vacío, texto > 2000 chars)
- `503` → clasificador no disponible (modelos no cargados)

---

## Evaluación reproducible

```bash
# Evaluación completa del baseline (train vs held-out)
python src/evaluation/evaluate.py --mode baseline

# Evaluación LLM sobre muestra held-out (requiere ANTHROPIC_API_KEY)
python src/evaluation/evaluate.py --mode llm --sample 20
```

Genera matrices de confusión en `data/evaluation/`.

### Métricas baseline (referencia)

| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Train (8 casos) | 0.990 | 0.992 |
| Held-out (Mercedes + Luis) | **0.616** | **0.535** |
| **Gap** | **0.374** | **0.457** |

El gap alto confirma que el baseline no generaliza a casos nuevos. El LLM few-shot es la arquitectura correcta para este régimen de datos.

---

## Dataset construido

| Atributo | Valor |
|----------|-------|
| Total turnos OPERADOR | 364 |
| Casos | 10 (Camila, Javiera, Patricio, Hernán, Rosa, Matías, Julia, Mercedes, Luis, Carolina) |
| Split train | 291 turnos (8 casos) |
| Split held-out | 73 turnos (Mercedes + Luis) |
| Etiquetado fase | Weak labels desde encabezados ABCDE del guion |
| Etiquetado actos | Reglas heurísticas (patrones regex clínicos) |
| Formato | CSV + JSON en `data/processed/` |

**Decisiones de etiquetado:**
- La fase ABCDE se infiere del encabezado de sección más cercano precedente (weak supervision). Los retrocesos marcados como "B — NUEVA REGULACIÓN" en sección 11 del guion Camila se etiquetan B, aunque ocurran después de la sección E.
- Los actos verbales se etiquetan con reglas heurísticas y se espera que el LLM mejore estas etiquetas en el pipeline de active learning.
- Las `RAMIFICACIONES` se marcan con `is_ramificacion=True` y se incluyen en el dataset con la fase del bloque donde aparecen.

---

## Producción real — qué falta

Ver `writeup.md` para discusión extendida. En resumen:
- Autenticación en `/classify`
- Rate limiting + observabilidad (Prometheus/Grafana)
- Persistencia de clasificaciones para auditoría clínica
- Drift detection sobre distribución de fases predichas
- Módulo de riesgo con escalamiento a revisor humano
- Tests de integración y carga

---

## Decisiones de diseño clave

1. **Split por caso, no por línea** — mezclar turnos del mismo guion entre train y test infla artificialmente las métricas porque el modelo memoriza el vocabulario específico del caso. El split por caso mide generalización real.

2. **Held-out = Mercedes y Luis** — son los casos sin nivel de riesgo predefinido, lo que los hace más desafiantes y más representativos de usuarios reales del sistema.

3. **Fallback automático** — si el LLM falla o supera 5 segundos, el baseline responde. El campo `source` en la respuesta permite monitorear la tasa de fallback en producción.

4. **Sin fine-tuning** — con 364 turnos y weak labels, fine-tuning de un transformer produciría overfitting severo. Se necesitan ≥1.000 turnos anotados por humanos antes de intentarlo.
