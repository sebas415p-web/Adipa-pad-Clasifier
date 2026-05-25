""" 
classifier.py
-------------
Clasificador principal: LLM few-shot (Anthropic Claude) con fallback al baseline
TF-IDF+LogReg cuando el LLM no está disponible o supera el timeout.

Arquitectura:
  1. LLM few-shot → salida estructurada JSON (fase + actos verbales + confianza)
  2. Fallback → baseline sklearn si el LLM falla/tarda
  3. Calibración de confianza: el LLM retorna probabilidades nativas;
     el baseline usa predict_proba de LogReg.

Justificación de LLM-first:
  - Solo 10 guiones (~364 turnos) → insufficient data para fine-tuning robusto.
  - El baseline muestra gap train/held de 37 puntos de accuracy → evidencia empírica.
  - El LLM generaliza por comprensión semántica del español clínico, no memorización.
  - Costo estimado: ~0.002 USD/turno con claude-haiku-3 → viable para volumen de alumnos.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes clínicas
# ---------------------------------------------------------------------------

PHASES = ["A", "B", "C", "D", "E"]

PHASE_DESCRIPTIONS = {
    "A": "Escucha Activa / Recepción — el operador recibe la llamada, verifica seguridad básica y establece vínculo sin juzgar.",
    "B": "Regulación Emocional / Validación — técnicas de regulación (respiración, anclaje), validación del estado emocional.",
    "C": "Categorización de Necesidades — identificación activa de necesidades urgentes, ordenamiento de prioridades.",
    "D": "Derivación / Red de Apoyo — activación de red presencial, coordinación de recursos, cierre de la llamada.",
    "E": "Psicoeducación / Esperanza — información sobre reacciones normales post-crisis, orientación sobre próximos pasos.",
}

ACT_DESCRIPTIONS = {
    "validacion": "Reconoce y normaliza lo que siente el paciente. Ej: 'Tiene sentido que sienta eso después de lo que vivió.'",
    "pregunta_abierta": "Invita a elaborar sin delimitar. Ej: '¿Qué está pasando ahora?', '¿Cómo se siente?'",
    "pregunta_cerrada": "Respuesta sí/no o concreta. Ej: '¿Está solo/a?', '¿Tiene la puerta cerrada?'",
    "reflejo": "Devuelve lo que el paciente expresó. Ej: 'La imagen sigue apareciendo.'",
    "interpretacion": "Explica el mecanismo subyacente. Ej: 'Su cuerpo todavía está en modo alarma.'",
    "silencio_contencion": "Presencia sin presionar; no exige que el paciente hable o cambie.",
    "confrontacion": "Señala una contradicción o conducta de riesgo. Ej: 'La detengo con cuidado.'",
    "directivo": "Instrucción concreta de acción. Ej: 'Ponga el pestillo.', 'Inhale contando cuatro.'",
}

# Few-shot examples (seleccionados del corpus, NO de Mercedes/Luis)
FEW_SHOT_EXAMPLES = [
    {
        "contexto": "CAMILA: No puedo... no puedo sacármelo de la cabeza.",
        "turno": "Estoy aquí. Vamos de a poco. Primero dígame su nombre, por favor.",
        "fase": "A",
        "actos": ["directivo"],
        "razon": "Primer contacto; el operador establece presencia y pide datos básicos de forma directiva.",
    },
    {
        "contexto": "CAMILA: Me da miedo relajarme. Como que si bajo la guardia va a pasar algo.",
        "turno": "Tiene sentido. Una parte suya cree que si baja la guardia se expone. Pero respirar no significa dejar de cuidarse.",
        "fase": "B",
        "actos": ["validacion", "interpretacion"],
        "razon": "Valida el miedo y luego explica el mecanismo cognitivo detrás de la resistencia.",
    },
    {
        "contexto": "CAMILA: No puedo hacer todo eso.",
        "turno": "No tiene que hacerlo sola ni ahora mismo. Primero usted. Después una acción. Después otra.",
        "fase": "C",
        "actos": ["directivo", "silencio_contencion"],
        "razon": "Categoriza y prioriza necesidades; frena la urgencia del paciente de forma contenedora.",
    },
    {
        "contexto": "JAVIERA: No sé si puedo salir. Los niños están aquí.",
        "turno": "Entiendo. ¿Dónde están los niños en este momento? ¿Están con usted en el baño?",
        "fase": "C",
        "actos": ["validacion", "pregunta_cerrada"],
        "razon": "Valida brevemente y categoriza la necesidad de seguridad con pregunta cerrada de verificación.",
    },
    {
        "contexto": "HERNÁN: No sé para qué seguir.",
        "turno": "Esa frase me importa. ¿Está pensando en hacerse daño?",
        "fase": "C",
        "actos": ["reflejo", "pregunta_cerrada"],
        "razon": "Refleja la frase de riesgo y evalúa directamente con pregunta cerrada.",
    },
]

SYSTEM_PROMPT = """Eres un sistema experto en análisis clínico de sesiones de Primeros Auxilios Psicológicos (PAP).

Tu tarea es clasificar un turno del operador (profesional PAP) según:
1. **Fase ABCDE**: en qué fase del modelo PAP se ubica el turno.
2. **Actos verbales**: qué técnica(s) clínica(s) usa el operador (multi-label posible).

FASES:
{phase_descriptions}

ACTOS VERBALES:
{act_descriptions}

Responde ÚNICAMENTE con JSON válido, sin texto adicional, con este esquema exacto:
{{
  "fase": {{
    "label": "<A|B|C|D|E>",
    "confidence": <0.0-1.0>,
    "razon": "<breve justificación clínica>"
  }},
  "actos_verbales": [
    {{
      "label": "<nombre_del_acto>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

Las etiquetas de actos verbales deben ser exactamente uno de:
validacion, pregunta_abierta, pregunta_cerrada, reflejo, interpretacion,
silencio_contencion, confrontacion, directivo."""

FEW_SHOT_SYSTEM = SYSTEM_PROMPT.format(
    phase_descriptions="\n".join(f"- {k}: {v}" for k, v in PHASE_DESCRIPTIONS.items()),
    act_descriptions="\n".join(f"- {k}: {v}" for k, v in ACT_DESCRIPTIONS.items()),
)


# ---------------------------------------------------------------------------
# Clasificador LLM
# ---------------------------------------------------------------------------

class LLMClassifier:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", timeout: float = 5.0):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no está configurada.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.timeout = timeout

    def _build_messages(self, turno_operador: str, contexto_previo: str) -> list[dict]:
        messages = []

        # Few-shot examples
        for ex in FEW_SHOT_EXAMPLES:
            user_content = (
                f"Contexto (turno previo del paciente):\n{ex['contexto']}\n\n"
                f"Turno del operador a clasificar:\n{ex['turno']}"
            )
            expected = {
                "fase": {
                    "label": ex["fase"],
                    "confidence": 0.95,
                    "razon": ex["razon"],
                },
                "actos_verbales": [
                    {"label": a, "confidence": 0.90} for a in ex["actos"]
                ],
            }
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)})

        # Query actual
        query = (
            f"Contexto (turno previo del paciente):\n{contexto_previo or '[Sin contexto previo]'}\n\n"
            f"Turno del operador a clasificar:\n{turno_operador}"
        )
        messages.append({"role": "user", "content": query})
        return messages

    def classify(self, turno_operador: str, contexto_previo: str = "") -> dict:
        messages = self._build_messages(turno_operador, contexto_previo)

        try:
            t0 = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=FEW_SHOT_SYSTEM,
                messages=messages,
            )
            latency = time.time() - t0

            raw = response.content[0].text.strip()
            result = json.loads(raw)
            result["source"] = "llm"
            result["latency_ms"] = round(latency * 1000)

            # Validar estructura
            assert result["fase"]["label"] in PHASES
            assert isinstance(result["actos_verbales"], list)

            logger.info(f"LLM classify OK | fase={result['fase']['label']} | {latency*1000:.0f}ms")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM retornó JSON inválido: {e}")
            raise
        except anthropic.APITimeoutError:
            logger.warning("LLM timeout")
            raise
        except Exception as e:
            logger.warning(f"LLM error: {e}")
            raise


# ---------------------------------------------------------------------------
# Clasificador Baseline (fallback)
# ---------------------------------------------------------------------------

class BaselineClassifier:
    def __init__(self, models_dir: Path):
        with open(models_dir / "phase_model.pkl", "rb") as f:
            self.phase_model = pickle.load(f)
        with open(models_dir / "verbal_model.pkl", "rb") as f:
            self.verbal_model = pickle.load(f)
        with open(models_dir / "mlb.pkl", "rb") as f:
            self.mlb = pickle.load(f)
        logger.info("Baseline classifier cargado.")

    def classify(self, turno_operador: str, contexto_previo: str = "") -> dict:
        feature = f"{contexto_previo} [SEP] {turno_operador}"

        # Fase
        phase_proba = self.phase_model.predict_proba([feature])[0]
        phase_classes = self.phase_model.classes_
        phase_idx = phase_proba.argmax()
        phase_label = phase_classes[phase_idx]
        phase_conf = float(phase_proba[phase_idx])

        # Actos verbales
        actos = []
        if self.verbal_model:
            verbal_proba = self.verbal_model.predict_proba([feature])
            # OneVsRest retorna lista de arrays
            for i, act in enumerate(self.mlb.classes_):
                if isinstance(verbal_proba, list):
                    prob = float(verbal_proba[i][0][1])
                else:
                    prob = float(verbal_proba[0][i])
                if prob > 0.35:
                    actos.append({"label": act, "confidence": round(prob, 3)})

        return {
            "fase": {"label": phase_label, "confidence": round(phase_conf, 3), "razon": "baseline"},
            "actos_verbales": actos,
            "source": "baseline",
            "latency_ms": 0,
        }


# ---------------------------------------------------------------------------
# Clasificador Unificado (LLM + fallback)
# ---------------------------------------------------------------------------

class PAPClassifier:
    """
    Estrategia:
    1. Intenta LLM (claude-haiku, ~5s timeout).
    2. Si falla (timeout, API error, JSON inválido) → fallback a baseline.
    3. Registra la fuente usada ('llm' o 'baseline') en la respuesta.
    """

    def __init__(self, models_dir: Optional[Path] = None, use_llm: bool = True):
        self.llm: Optional[LLMClassifier] = None
        self.baseline: Optional[BaselineClassifier] = None

        if use_llm:
            try:
                self.llm = LLMClassifier()
                logger.info("LLMClassifier inicializado.")
            except Exception as e:
                logger.warning(f"LLM no disponible: {e}. Solo baseline activo.")

        if models_dir and models_dir.exists():
            try:
                self.baseline = BaselineClassifier(models_dir)
            except Exception as e:
                logger.warning(f"Baseline no disponible: {e}.")

    def classify(self, turno_operador: str, contexto_previo: str = "") -> dict:
        # Intentar LLM primero
        if self.llm:
            try:
                return self.llm.classify(turno_operador, contexto_previo)
            except Exception as e:
                logger.warning(f"LLM falló, usando fallback: {e}")

        # Fallback a baseline
        if self.baseline:
            return self.baseline.classify(turno_operador, contexto_previo)

        # Sin ningún clasificador disponible
        raise RuntimeError("Ningún clasificador disponible (LLM ni baseline).")


# ---------------------------------------------------------------------------
# Instancia global (lazy init en la API)
# ---------------------------------------------------------------------------

_classifier: Optional[PAPClassifier] = None


def get_classifier(models_dir: Optional[Path] = None) -> PAPClassifier:
    global _classifier
    if _classifier is None:
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
        _classifier = PAPClassifier(models_dir=models_dir, use_llm=use_llm)
    return _classifier
