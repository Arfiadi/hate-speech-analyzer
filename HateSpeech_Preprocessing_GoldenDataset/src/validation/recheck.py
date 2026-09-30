from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def validate_saved_baseline(run_dir: Path, threshold: float) -> dict[str, Any]:
    bundle = joblib.load(run_dir / "models" / "baseline_tfidf_sgd.joblib")
    test = pd.read_csv(run_dir / "splits" / "test.csv")
    if not test["toxicity_label"].isin([0, 1]).all():
        raise ValueError("Test set harus berisi hard label 0 atau 1 saja.")
    probabilities = bundle["classifier"].predict_proba(bundle["vectorizer"].transform(test["text_clean"]))[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    actual = test["toxicity_label"].astype(int).to_numpy()
    metrics = {
        "rows": int(len(test)),
        "f1_toxic": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
        "precision_toxic": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall_toxic": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    with (metrics_dir / "validation_recheck.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
