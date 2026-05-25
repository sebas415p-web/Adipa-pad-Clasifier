# API Documentation

## Base URL
`http://localhost:8000`

## Endpoints

### GET /health
Returns service status.

**Response:**
```json
{"status": "ok", "classifier_available": true, "llm_active": true}
```

### POST /classify
Classifies an operator turn into ABCDE phase and verbal acts.

**Request:**
```json
{
  "turno_operador": "Tiene sentido que sienta eso.",
  "contexto_previo": "No entiendo por qué no me puedo calmar."
}
```

**Response:**
```json
{
  "fase": {"label": "B", "confidence": 0.94, "razon": "..."},
  "actos_verbales": [{"label": "validacion", "confidence": 0.91}],
  "source": "llm",
  "latency_ms": 923
}
```

**Errors:**
- `422`: Invalid input (empty turno, too long)
- `503`: Classifier not available
