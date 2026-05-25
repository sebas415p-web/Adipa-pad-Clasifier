"""
extract_dataset.py
------------------
Transforma los guiones clínicos PAP en un dataset estructurado (CSV + JSON).

Lógica de etiquetado:
- fase_abcde: weak label inferida del encabezado de sección más cercano anterior al turno.
- acto_verbal: reglas heurísticas + lista de patrones clínicos.
  (Etiqueta final = multi-label, puede combinarse con LLM en pipeline posterior.)
- contexto_previo: turno inmediatamente anterior del paciente.
- is_ramificacion: flag para turnos dentro de bloques RAMIFICACIÓN.
"""

import re
import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CASE_NAMES = [
    "Camila", "Javiera", "Patricio", "Hernán", "Rosa",
    "Matías", "Julia", "Mercedes", "Luis", "Carolina",
]

HELD_OUT_CASES = {"Mercedes", "Luis"}

PHASE_PATTERN = re.compile(
    r"^([A-E])\s*[—\-–]\s*(ESCUCHA|REGULACI[OÓ]N|CATEGORI|DERIVACI[OÓ]N|PSICOEDUC)",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_HEADER_PATTERN = re.compile(
    r"^\d+\.\s+[A-ZÁÉÍÓÚÑ ]+$", re.MULTILINE
)

# Patrones heurísticos para actos verbales (multi-label posible)
VERBAL_ACT_RULES = {
    "validacion": [
        r"\bentiendo\b", r"\btiene sentido\b", r"\blamento\b", r"\bes comprensible\b",
        r"\bes normal\b", r"\bno es tu culpa\b", r"\bno es su culpa\b",
        r"\beso es muy difícil\b", r"\blo que siente es\b",
    ],
    "pregunta_abierta": [
        r"\b¿qué\b", r"\b¿cómo\b", r"\b¿de qué\b", r"\b¿cuénteme\b",
        r"\b¿puede contarme\b", r"\b¿qué pasó\b", r"\b¿cómo se siente\b",
        r"\b¿qué necesita\b",
    ],
    "pregunta_cerrada": [
        r"\b¿está\b", r"\b¿tiene\b", r"\b¿puede\b", r"\b¿hay\b",
        r"\b¿reconoce\b", r"\b¿quiere\b.*\?$", r"\b¿sigue\b",
    ],
    "reflejo": [
        r"^(OPERADOR\s+)?(la|lo|le) (imagen|pistola|miedo|soledad|culpa)",
        r"volvió fuerte", r"sigue apareciendo", r"está hablando muy fuerte",
        r"su cuerpo reaccionó", r"la parte suya",
    ],
    "interpretacion": [
        r"porque\b.{5,50}(cuerpo|mente|alarma)", r"lo que su cuerpo",
        r"no significa.*sino", r"buscando.*para sentir control",
        r"su sistema de alarma",
    ],
    "silencio_contencion": [
        r"^pausa", r"silencio", r"no tiene que\b", r"no voy a pedirle",
        r"no necesita describir", r"puede quedarse",
    ],
    "confrontacion": [
        r"la detengo", r"espere un momento", r"eso no es exactamente",
        r"le pido que no", r"necesito que pare",
    ],
    "directivo": [
        r"cierre la puerta", r"ponga el pestillo", r"no abra",
        r"llame a", r"salga", r"vaya al", r"presione", r"inhale",
        r"suelte el aire", r"mire la", r"busque un punto", r"vuelva a",
        r"levántese", r"siéntese", r"ponga una mano",
    ],
}


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def parse_guiones(raw_text: str) -> list[dict]:
    """Parsea el texto completo del PDF y retorna lista de turnos del OPERADOR."""

    # Dividir por caso usando el marcador de página + "Guion — Caso"
    case_splits = re.split(r"\fGuion — Caso (\w+)", raw_text)

    records = []
    # case_splits[0] es el preámbulo; luego alternamos nombre/contenido
    for i in range(1, len(case_splits), 2):
        case_name = case_splits[i].strip()
        case_text = case_splits[i + 1] if i + 1 < len(case_splits) else ""
        case_records = _parse_single_case(case_name, case_text)
        records.extend(case_records)

    return records


def _get_operator_name(text: str) -> str:
    """Extrae el nombre del operador desde la sección Personajes del guion."""
    # Buscar "OPERADOR", "OPERADORA", o un nombre propio designado como operador
    match = re.search(
        r"Personajes:.*?(OPERADOR[A]?|GABRIEL)\b", text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    return "OPERADOR"


def _parse_single_case(case_name: str, text: str) -> list[dict]:
    """Extrae todos los turnos del OPERADOR de un guion."""

    records = []
    lines = text.splitlines()

    # Determinar el nombre del operador en este guion
    operator_name = _get_operator_name(text)
    # Rosa usa GABRIEL en lugar de OPERADOR
    operator_names = {"OPERADOR", "OPERADORA", operator_name}

    current_phase = "A"           # fase ABCDE activa
    current_section = ""          # número de sección (1, 2, …)
    is_ramificacion = False
    prev_patient_text = ""        # turno previo del paciente
    turn_index = 0

    # Buffer para turnos multi-línea del OPERADOR
    operador_buffer = []
    patient_buffer = []
    current_speaker = None

    def flush_operador():
        nonlocal turn_index, prev_patient_text
        if not operador_buffer:
            return
        text_joined = " ".join(operador_buffer).strip()
        # Limpiar el prefijo "OPERADOR " si quedó
        text_clean = re.sub(r"^OPERADOR\s*", "", text_joined).strip()
        if len(text_clean) < 5:
            operador_buffer.clear()
            return

        actos = _detect_verbal_acts(text_clean)

        records.append({
            "caso": case_name,
            "split": "held_out" if case_name in HELD_OUT_CASES else "train",
            "turno_id": f"{case_name}_{turn_index:04d}",
            "fase_abcde": current_phase,
            "seccion": current_section,
            "is_ramificacion": is_ramificacion,
            "texto_operador": text_clean,
            "contexto_previo": prev_patient_text,
            "actos_verbales": actos,
            "actos_verbales_str": "|".join(actos) if actos else "sin_etiqueta",
        })
        turn_index += 1
        operador_buffer.clear()

    def flush_patient():
        nonlocal prev_patient_text
        if not patient_buffer:
            return
        prev_patient_text = " ".join(patient_buffer).strip()
        # Remover prefijo del nombre del paciente
        prev_patient_text = re.sub(r"^[A-ZÁÉÍÓÚ ]{2,20}\s+", "", prev_patient_text)
        patient_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detectar encabezado de FASE (ej: "A — ESCUCHA ACTIVA")
        phase_match = re.match(
            r"^([A-E])\s*[—\-–]\s*(ESCUCHA|REGULACI[OÓ]N|CATEGORI|DERIVACI[OÓ]N|PSICOEDUC)",
            stripped, re.IGNORECASE
        )
        if phase_match:
            flush_operador()
            flush_patient()
            current_phase = phase_match.group(1).upper()
            is_ramificacion = False
            current_speaker = None
            continue

        # Detectar sección numerada (ej: "3. SEGUNDO INTENTO DE RESPIRACIÓN")
        if re.match(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ]", stripped):
            flush_operador()
            flush_patient()
            section_match = re.match(r"^(\d+)\.", stripped)
            if section_match:
                current_section = section_match.group(1)
            current_speaker = None
            continue

        # Detectar RAMIFICACIÓN
        if re.match(r"^RAMIFICACI[OÓ]N\s+\d+", stripped, re.IGNORECASE):
            flush_operador()
            flush_patient()
            is_ramificacion = True
            current_speaker = None
            continue

        # Detectar turno OPERADOR (variantes: OPERADOR, OPERADORA, GABRIEL)
        op_match = re.match(r"^(OPERADOR[A]?|GABRIEL)(\s+|$)(.*)", stripped)
        if op_match and op_match.group(1) in operator_names:
            flush_patient()
            rest = op_match.group(3).strip()
            if current_speaker == "OPERADOR":
                if rest:
                    operador_buffer.append(rest)
            else:
                flush_operador()
                current_speaker = "OPERADOR"
                if rest:
                    operador_buffer.append(rest)
            continue

        # Detectar turno del PACIENTE u otro personaje (no OPERADOR)
        patient_match = re.match(
            r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]+?)\s{2,}(.+)$", stripped
        )
        if patient_match and patient_match.group(1) not in ("OPERADOR", "VOZ EXTERNA"):
            flush_operador()
            if current_speaker == "PATIENT":
                patient_buffer.append(patient_match.group(2))
            else:
                flush_patient()
                current_speaker = "PATIENT"
                patient_buffer.append(patient_match.group(2))
            continue

        # Líneas de acción/narración (descripción escénica) — ignorar para texto
        if stripped.startswith("Se ") or stripped.startswith("Pausa") or stripped.startswith("Silencio"):
            if current_speaker == "OPERADOR":
                flush_operador()
            elif current_speaker == "PATIENT":
                flush_patient()
            current_speaker = None
            continue

        # Continuación de turno anterior (línea wrapeada)
        if current_speaker == "OPERADOR":
            operador_buffer.append(stripped)
        elif current_speaker == "PATIENT":
            patient_buffer.append(stripped)

    # Flush final
    flush_operador()

    return records


def _detect_verbal_acts(text: str) -> list[str]:
    """Aplica reglas heurísticas para detectar actos verbales (multi-label)."""
    text_lower = text.lower()
    acts = []
    for act, patterns in VERBAL_ACT_RULES.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                acts.append(act)
                break
    # Si no se detecta nada, es "sin_etiqueta" (a revisar con LLM)
    return acts if acts else []


# ---------------------------------------------------------------------------
# Exportar dataset
# ---------------------------------------------------------------------------

def build_dataset(raw_path: Path, output_dir: Path) -> pd.DataFrame:
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    records = parse_guiones(raw_text)

    df = pd.DataFrame(records)

    # Estadísticas rápidas
    print(f"Total turnos OPERADOR extraídos: {len(df)}")
    print(f"Por caso:\n{df.groupby('caso')['turno_id'].count().to_string()}")
    print(f"\nDistribución de fases:\n{df['fase_abcde'].value_counts().to_string()}")
    print(f"\nSplit: {df['split'].value_counts().to_string()}")

    # CSV
    csv_path = output_dir / "dataset_turnos.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nCSV guardado: {csv_path}")

    # JSON (sin la columna list para ser JSON-serializable)
    json_records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        # actos_verbales es una list, dejarla como list en JSON
        json_records.append(rec)

    json_path = output_dir / "dataset_turnos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f"JSON guardado: {json_path}")

    return df


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    raw_path = base / "data" / "raw" / "guiones.txt"
    output_dir = base / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(raw_path, output_dir)
