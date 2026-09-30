from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


def _safe_n_splits(frame: pd.DataFrame, requested: int) -> int:
    group_counts = frame.groupby("toxicity_label")["dedup_key"].nunique()
    if len(group_counts) < 2:
        return 0
    return min(requested, int(group_counts.min()))


def _expand_soft_labels(frame: pd.DataFrame, tie_handling: str, tie_downweight: float) -> pd.DataFrame:
    expanded: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        base = {
            "text_clean": row.text_clean,
            "sample_weight": float(row.sample_weight),
            "training_phase": row.training_phase,
            "dedup_key": row.dedup_key,
        }
        if row.toxicity_label in (0, 1):
            expanded.append({**base, "target": int(row.toxicity_label)})
        elif row.toxicity_label == 0.5 and tie_handling in {"soft", "downweight"}:
            multiplier = 0.5 if tie_handling == "soft" else 0.5 * float(tie_downweight)
            expanded.append({**base, "target": 0, "sample_weight": base["sample_weight"] * multiplier})
            expanded.append({**base, "target": 1, "sample_weight": base["sample_weight"] * multiplier})
    return pd.DataFrame(expanded)


def _balanced_weights(target: np.ndarray) -> np.ndarray:
    counts = pd.Series(target).value_counts()
    total = len(target)
    return np.asarray([total / (2 * counts[int(label)]) for label in target], dtype=float)


def _fit_classifier(
    train_frame: pd.DataFrame,
    settings: dict[str, str],
    config: dict[str, Any],
) -> tuple[TfidfVectorizer, SGDClassifier]:
    candidate_cfg = config["candidates"]
    evaluator_cfg = config["evaluator"]
    expanded = _expand_soft_labels(
        train_frame,
        settings["tie_handling"],
        float(candidate_cfg["tie_downweight"]),
    )
    if expanded.empty or expanded["target"].nunique() < 2:
        raise ValueError("Data latih tidak memiliki dua kelas setelah penanganan label.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=int(evaluator_cfg["max_features"]),
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(expanded["text_clean"])
    y_train = expanded["target"].to_numpy(dtype=int)
    sample_weight = expanded["sample_weight"].to_numpy(dtype=float) * _balanced_weights(y_train)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=float(evaluator_cfg["alpha"]),
        max_iter=int(evaluator_cfg["max_iter"]),
        random_state=int(config["project"]["random_seed"]),
        tol=1e-3,
    )

    if settings["annotator_strategy"] == "curriculum":
        phase_one = expanded.loc[expanded["training_phase"] == "phase_1_consensus"]
        if phase_one["target"].nunique() == 2:
            phase_one_idx = phase_one.index.to_numpy()
            classifier.partial_fit(
                X_train[phase_one_idx],
                y_train[phase_one_idx],
                classes=np.array([0, 1]),
                sample_weight=sample_weight[phase_one_idx],
            )
            for _ in range(4):
                classifier.partial_fit(X_train, y_train, sample_weight=sample_weight)
        else:
            classifier.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        classifier.fit(X_train, y_train, sample_weight=sample_weight)
    return vectorizer, classifier


def score_candidate(candidate: pd.DataFrame, settings: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    """Score every candidate with the same leak-safe TF-IDF baseline and Toxic F1."""
    hard = candidate.loc[candidate["use_for_validation"]].copy()
    n_splits = _safe_n_splits(hard, int(config["evaluator"]["validation_folds"]))
    if n_splits < 2:
        return {"status": "SKIPPED", "reason": "insufficient_grouped_examples_per_class"}

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(config["project"]["random_seed"]),
    )
    try:
        train_idx, validation_idx = next(
            splitter.split(hard["text_clean"], hard["toxicity_label"], groups=hard["dedup_key"])
        )
        hard_train = hard.iloc[train_idx].copy()
        validation = hard.iloc[validation_idx].copy()
        validation_groups = set(validation["dedup_key"])
        soft_rows = candidate.loc[
            candidate["use_for_training"]
            & (candidate["toxicity_label"] == 0.5)
            & ~candidate["dedup_key"].isin(validation_groups)
        ].copy()
        train = pd.concat([hard_train, soft_rows], ignore_index=True)
        vectorizer, classifier = _fit_classifier(train, settings, config)
        probabilities = classifier.predict_proba(vectorizer.transform(validation["text_clean"]))[:, 1]
        threshold = float(config["evaluator"]["threshold"])
        predicted = (probabilities >= threshold).astype(int)
        actual = validation["toxicity_label"].astype(int).to_numpy()
        cm = confusion_matrix(actual, predicted, labels=[0, 1]).tolist()
        return {
            "status": "OK",
            "validation_rows": int(len(validation)),
            "train_rows": int(len(train)),
            "f1_toxic": round(float(f1_score(actual, predicted, pos_label=1, zero_division=0)), 4),
            "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
            "precision_toxic": round(float(precision_score(actual, predicted, pos_label=1, zero_division=0)), 4),
            "recall_toxic": round(float(recall_score(actual, predicted, pos_label=1, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(actual, predicted)), 4),
            "confusion_matrix": cm,
        }
    except ValueError as error:
        return {"status": "SKIPPED", "reason": str(error)}


def fit_final_baseline(
    model_ready: pd.DataFrame, settings: dict[str, str], config: dict[str, Any]
) -> tuple[dict[str, Any], TfidfVectorizer, SGDClassifier, dict[str, pd.DataFrame]]:
    """Create a small baseline only to prove that the Golden Dataset can reach model validation."""
    hard = model_ready.loc[model_ready["use_for_validation"]].copy()
    n_splits = _safe_n_splits(hard, 3)
    if n_splits < 3:
        raise ValueError("Minimal tiga grup per kelas diperlukan untuk train validation test baseline.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(config["project"]["random_seed"]),
    )
    folds = list(splitter.split(hard["text_clean"], hard["toxicity_label"], groups=hard["dedup_key"]))
    validation_idx = folds[0][1]
    test_idx = folds[1][1]
    holdout_idx = set(validation_idx).union(set(test_idx))
    train_idx = [idx for idx in range(len(hard)) if idx not in holdout_idx]
    train_hard = hard.iloc[train_idx].copy()
    validation = hard.iloc[validation_idx].copy()
    test = hard.iloc[test_idx].copy()
    heldout_groups = set(validation["dedup_key"]).union(set(test["dedup_key"]))
    soft_train = model_ready.loc[
        (model_ready["toxicity_label"] == 0.5) & ~model_ready["dedup_key"].isin(heldout_groups)
    ].copy()
    train = pd.concat([train_hard, soft_train], ignore_index=True)
    vectorizer, classifier = _fit_classifier(train, settings, config)

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        probabilities = classifier.predict_proba(vectorizer.transform(frame["text_clean"]))[:, 1]
        predicted = (probabilities >= float(config["evaluator"]["threshold"])).astype(int)
        actual = frame["toxicity_label"].astype(int).to_numpy()
        return {
            "rows": int(len(frame)),
            "f1_toxic": round(float(f1_score(actual, predicted, zero_division=0)), 4),
            "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
            "precision_toxic": round(float(precision_score(actual, predicted, zero_division=0)), 4),
            "recall_toxic": round(float(recall_score(actual, predicted, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(actual, predicted)), 4),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        }

    return {"validation": metrics(validation), "test": metrics(test)}, vectorizer, classifier, {
        "train": train,
        "validation": validation,
        "test": test,
    }
