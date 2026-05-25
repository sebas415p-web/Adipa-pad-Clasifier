"""
train_baseline.py
-----------------
Baseline clásico:
- Fase ABCDE: TF-IDF + LogisticRegression (multiclass)
- Actos verbales: TF-IDF + OneVsRestClassifier (multi-label)

Split honesto por caso: train en 8 casos, held-out Mercedes + Luis.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, hamming_loss
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PHASES = ["A", "B", "C", "D", "E"]
VERBAL_ACTS = [
    "validacion", "pregunta_abierta", "pregunta_cerrada",
    "reflejo", "interpretacion", "silencio_contencion",
    "confrontacion", "directivo",
]
HELD_OUT_CASES = {"Mercedes", "Luis"}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)

    # Reconstruir actos_verbales como lista desde string separado por |
    def parse_acts(s):
        if pd.isna(s) or s == "sin_etiqueta" or s == "":
            return []
        return [a for a in s.split("|") if a in VERBAL_ACTS]

    df["actos_list"] = df["actos_verbales_str"].apply(parse_acts)

    # Feature: concatenar turno + contexto
    df["feature_text"] = (
        df["contexto_previo"].fillna("") + " [SEP] " + df["texto_operador"].fillna("")
    )

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    held_df = df[df["split"] == "held_out"].copy().reset_index(drop=True)

    print(f"Train: {len(train_df)} turnos | Held-out: {len(held_df)} turnos")
    return train_df, held_df


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def build_phase_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def build_verbal_model(mlb: MultiLabelBinarizer) -> tuple[Pipeline, MultiLabelBinarizer]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
        )),
    ])
    return pipe


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate_phase(model, X, y_true, split_name: str) -> dict:
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=PHASES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=PHASES)

    print(f"\n{'='*60}")
    print(f"[FASE ABCDE] — Split: {split_name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, labels=PHASES, zero_division=0))
    print("  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=PHASES, columns=PHASES)
    print(cm_df.to_string())

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist()}


def evaluate_verbal(model, mlb, X, y_true_lists, split_name: str) -> dict:
    # Filtrar filas con al menos una etiqueta (las reglas dejan vacíos)
    mask = [len(lst) > 0 for lst in y_true_lists]
    if sum(mask) < 5:
        print(f"\n[ACTOS VERBALES] — {split_name}: insuficientes etiquetas ({sum(mask)})")
        return {}

    X_f = [x for x, m in zip(X, mask) if m]
    y_f = [y for y, m in zip(y_true_lists, mask) if m]

    y_true_bin = mlb.transform(y_f)
    y_pred_bin = model.predict([x for x in X_f])

    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    hl = hamming_loss(y_true_bin, y_pred_bin)

    print(f"\n{'='*60}")
    print(f"[ACTOS VERBALES] — Split: {split_name}")
    print(f"  Micro F1:     {micro_f1:.4f}")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  Hamming Loss: {hl:.4f}")
    print("\n  Per-label report:")
    print(classification_report(
        y_true_bin, y_pred_bin,
        target_names=mlb.classes_,
        zero_division=0
    ))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": hl,
    }


def show_errors(model, X, y_true, texts, split_name: str, n=5):
    y_pred = model.predict(X)
    errors = [
        (texts[i], y_true[i], y_pred[i])
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    print(f"\n[Errores fase — {split_name}] ({len(errors)} de {len(y_true)})")
    for text, yt, yp in errors[:n]:
        snippet = text[:80].replace("\n", " ")
        print(f"  Real={yt} Pred={yp} | '{snippet}…'")


# ---------------------------------------------------------------------------
# Entrenamiento y guardado
# ---------------------------------------------------------------------------

def train_and_evaluate(data_path: Path, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, held_df = load_data(data_path)

    X_train = train_df["feature_text"].tolist()
    y_phase_train = train_df["fase_abcde"].tolist()
    y_verbal_train = train_df["actos_list"].tolist()

    X_held = held_df["feature_text"].tolist()
    y_phase_held = held_df["fase_abcde"].tolist()
    y_verbal_held = held_df["actos_list"].tolist()

    # ---- Modelo de fase ----
    print("\nEntrenando modelo de FASE ABCDE…")
    phase_model = build_phase_model()
    phase_model.fit(X_train, y_phase_train)

    eval_train_phase = evaluate_phase(phase_model, X_train, y_phase_train, "TRAIN (8 casos)")
    show_errors(phase_model, X_train, y_phase_train, train_df["texto_operador"].tolist(), "TRAIN")

    eval_held_phase = evaluate_phase(phase_model, X_held, y_phase_held, "HELD-OUT (Mercedes + Luis)")
    show_errors(phase_model, X_held, y_phase_held, held_df["texto_operador"].tolist(), "HELD-OUT")

    # ---- Modelo de actos verbales ----
    print("\nEntrenando modelo de ACTOS VERBALES…")
    mlb = MultiLabelBinarizer(classes=VERBAL_ACTS)
    y_train_bin = mlb.fit_transform(y_verbal_train)

    verbal_model = build_verbal_model(mlb)
    # Solo entrenar con muestras que tienen al menos un acto
    has_label = y_train_bin.sum(axis=1) > 0
    if has_label.sum() > 10:
        verbal_model.fit(
            [x for x, m in zip(X_train, has_label) if m],
            y_train_bin[has_label]
        )
    else:
        print("  Advertencia: muy pocas etiquetas verbales — modelo verbal no entrenado.")
        verbal_model = None

    if verbal_model:
        eval_train_verbal = evaluate_verbal(verbal_model, mlb, X_train, y_verbal_train, "TRAIN")
        eval_held_verbal = evaluate_verbal(verbal_model, mlb, X_held, y_verbal_held, "HELD-OUT")

    # ---- Análisis de gap train vs held-out ----
    print(f"\n{'='*60}")
    print("ANÁLISIS DE GAP (train vs held-out)")
    print(f"  Fase — Accuracy: train={eval_train_phase['accuracy']:.3f} | "
          f"held={eval_held_phase['accuracy']:.3f} | "
          f"gap={eval_train_phase['accuracy'] - eval_held_phase['accuracy']:.3f}")
    print(f"  Fase — Macro F1: train={eval_train_phase['macro_f1']:.3f} | "
          f"held={eval_held_phase['macro_f1']:.3f} | "
          f"gap={eval_train_phase['macro_f1'] - eval_held_phase['macro_f1']:.3f}")
    print("""
  Interpretación:
  - Un gap alto (>0.15) indica overfitting al estilo de redacción de los casos de train.
  - Para un corpus de 10 guiones con fuerte coherencia de autor, cierto gap es esperable.
  - La métrica held-out es la que importa para generalización real.
""")

    # ---- Guardar modelos ----
    with open(models_dir / "phase_model.pkl", "wb") as f:
        pickle.dump(phase_model, f)
    with open(models_dir / "verbal_model.pkl", "wb") as f:
        pickle.dump(verbal_model, f)
    with open(models_dir / "mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)

    print(f"\nModelos guardados en {models_dir}")

    # Guardar métricas
    metrics = {
        "phase_train": eval_train_phase,
        "phase_held": eval_held_phase,
    }
    with open(models_dir / "metrics_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return phase_model, verbal_model, mlb


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    data_path = base / "data" / "processed" / "dataset_turnos.csv"
    models_dir = base / "models"
    train_and_evaluate(data_path, models_dir)
