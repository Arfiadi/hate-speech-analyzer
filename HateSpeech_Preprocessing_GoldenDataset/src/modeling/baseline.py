from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing.evaluation import fit_final_baseline


def train_baseline(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    processed = run_dir / "processed" / "golden_dataset_model_ready.csv"
    settings_path = run_dir / "reports" / "selected_strategy.json"
    if not processed.exists() or not settings_path.exists():
        raise FileNotFoundError("Jalankan preprocessing terlebih dahulu agar Golden Dataset tersedia.")

    data = pd.read_csv(processed)
    with settings_path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)["selected_settings"]
    metrics, vectorizer, classifier, splits = fit_final_baseline(data, settings, config)
    models_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    splits_dir = run_dir / "splits"
    for directory in (models_dir, metrics_dir, splits_dir):
        directory.mkdir(exist_ok=True)

    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "settings": settings},
        models_dir / "baseline_tfidf_sgd.joblib",
    )
    for name, frame in splits.items():
        frame.to_csv(splits_dir / f"{name}.csv", index=False, encoding="utf-8")
    with (metrics_dir / "baseline_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
