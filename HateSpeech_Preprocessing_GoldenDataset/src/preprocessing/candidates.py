from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .labels import combine_vote_lists, parse_votes, vote_summary
from .text_cleaning import clean_text, normalise_for_dedup, text_is_valid


@dataclass(frozen=True)
class CandidateSettings:
    text_cleaning: str
    deduplication: str
    tie_handling: str
    annotator_strategy: str

    def as_dict(self) -> dict[str, str]:
        return {
            "text_cleaning": self.text_cleaning,
            "deduplication": self.deduplication,
            "tie_handling": self.tie_handling,
            "annotator_strategy": self.annotator_strategy,
        }


def prepare_source_frame(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_cfg = config["input"]
    filters = config["filters"]
    required = [input_cfg["text_column"], input_cfg["toxicity_column"], input_cfg["spam_column"]]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))

    id_column = input_cfg["id_column"]
    source = pd.DataFrame()
    source["text_id"] = raw[id_column].astype(str) if id_column in raw.columns else [f"row-{i:06d}" for i in raw.index]
    source["raw_text"] = raw[input_cfg["text_column"]].fillna("").astype(str)
    source["toxicity_votes"] = raw[input_cfg["toxicity_column"]].map(parse_votes)
    source["spam_votes"] = raw[input_cfg["spam_column"]].map(parse_votes)
    source["input_row"] = range(len(source))
    source["dedup_key"] = source["raw_text"].map(normalise_for_dedup)
    source["source_row_count"] = 1
    source["source_text_ids"] = source["text_id"]

    valid = source["raw_text"].map(
        lambda value: text_is_valid(value, filters["min_characters"], filters["min_tokens"])
    )
    invalid = source.loc[~valid].copy()
    invalid["review_reason"] = "invalid_or_too_short_text"
    return source.loc[valid].reset_index(drop=True), invalid.reset_index(drop=True)


def _pool_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    pooled_rows: list[dict[str, Any]] = []
    for _, group in frame.groupby("dedup_key", sort=False, dropna=False):
        first = group.iloc[0]
        pooled_rows.append(
            {
                "text_id": str(first["text_id"]),
                "raw_text": first["raw_text"],
                "toxicity_votes": combine_vote_lists(group["toxicity_votes"].tolist()),
                "spam_votes": combine_vote_lists(group["spam_votes"].tolist()),
                "input_row": int(group["input_row"].min()),
                "dedup_key": first["dedup_key"],
                "source_row_count": int(group["source_row_count"].sum()),
                "source_text_ids": "|".join(group["source_text_ids"].astype(str)),
            }
        )
    return pd.DataFrame(pooled_rows)


def _apply_deduplication(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "keep":
        return frame.copy()
    if strategy == "drop":
        return frame.drop_duplicates(subset=["dedup_key"], keep="first").copy()
    if strategy == "pool":
        return _pool_duplicates(frame)
    raise ValueError(f"Unknown deduplication strategy: {strategy}")


def _add_label_and_training_fields(
    frame: pd.DataFrame, settings: CandidateSettings, config: dict[str, Any]
) -> pd.DataFrame:
    filters = config["filters"]
    candidate_cfg = config["candidates"]
    data = frame.copy()

    toxicity = data["toxicity_votes"].map(vote_summary)
    spam = data["spam_votes"].map(vote_summary)
    data["toxicity_label"] = toxicity.map(lambda item: item["label"])
    data["toxicity_label_status"] = toxicity.map(lambda item: item["status"])
    data["annotator_count"] = toxicity.map(lambda item: item["annotator_count"])
    data["spam_label"] = spam.map(lambda item: item["label"])
    data["spam_label_status"] = spam.map(lambda item: item["status"])
    data["spam_annotator_count"] = spam.map(lambda item: item["annotator_count"])
    data["text_clean"] = data["raw_text"].map(lambda text: clean_text(text, settings.text_cleaning))

    data["sample_weight"] = 1.0
    data["training_phase"] = "all_data"
    if settings.annotator_strategy == "downweight_single":
        data.loc[data["annotator_count"] == 1, "sample_weight"] = float(candidate_cfg["single_annotator_weight"])
    elif settings.annotator_strategy == "multi_only":
        data["training_phase"] = "multi_annotator_only"
    elif settings.annotator_strategy == "curriculum":
        data["training_phase"] = data["annotator_count"].map(
            lambda count: "phase_1_consensus" if count >= 2 else "phase_2_full_data"
        )
    elif settings.annotator_strategy != "equal":
        raise ValueError(f"Unknown annotator strategy: {settings.annotator_strategy}")

    reasons: list[str] = []
    use_training: list[bool] = []
    use_validation: list[bool] = []
    for row in data.itertuples(index=False):
        reason = "ready_for_training"
        train_ready = True
        validation_ready = True
        if row.toxicity_label_status == "missing":
            reason, train_ready, validation_ready = "missing_toxicity_label", False, False
        elif row.spam_label == 1 and filters["exclude_spam"]:
            reason, train_ready, validation_ready = "spam_or_noise", False, False
        elif row.spam_label_status in {"tie", "missing"} and filters["exclude_spam_tie"]:
            reason, train_ready, validation_ready = "spam_label_uncertain", False, False
        elif row.toxicity_label == 0.5:
            validation_ready = False
            if settings.tie_handling == "drop":
                reason, train_ready = "toxicity_tie_excluded", False
            elif settings.tie_handling == "soft":
                reason = "soft_toxicity_label_for_training_only"
            elif settings.tie_handling == "downweight":
                reason = "downweighted_soft_toxicity_label_for_training_only"
            else:
                raise ValueError(f"Unknown tie strategy: {settings.tie_handling}")
        elif settings.annotator_strategy == "multi_only" and row.annotator_count < 2:
            reason, train_ready, validation_ready = "single_annotator_excluded", False, False
        reasons.append(reason)
        use_training.append(train_ready)
        use_validation.append(validation_ready and train_ready and row.toxicity_label in (0, 1))

    data["review_reason"] = reasons
    data["use_for_training"] = use_training
    data["use_for_validation"] = use_validation
    data["candidate_text_cleaning"] = settings.text_cleaning
    data["candidate_deduplication"] = settings.deduplication
    data["candidate_tie_handling"] = settings.tie_handling
    data["candidate_annotator_strategy"] = settings.annotator_strategy
    return data.sort_values("input_row").reset_index(drop=True)


def build_candidate(source: pd.DataFrame, settings: CandidateSettings, config: dict[str, Any]) -> pd.DataFrame:
    """Build one reproducible candidate dataset without silently discarding spam labels."""
    candidate = _apply_deduplication(source, settings.deduplication)
    return _add_label_and_training_fields(candidate, settings, config)


def candidate_summary(candidate: pd.DataFrame) -> dict[str, int | float]:
    return {
        "rows_after_deduplication": int(len(candidate)),
        "duplicate_rows_collapsed": int(candidate["source_row_count"].sum() - len(candidate)),
        "model_ready_rows": int(candidate["use_for_training"].sum()),
        "validation_ready_rows": int(candidate["use_for_validation"].sum()),
        "spam_or_noise_rows": int((candidate["spam_label"] == 1).sum()),
        "spam_uncertain_rows": int(candidate["spam_label_status"].isin(["tie", "missing"]).sum()),
        "toxicity_tie_rows": int((candidate["toxicity_label"] == 0.5).sum()),
        "single_annotator_rows": int((candidate["annotator_count"] == 1).sum()),
    }
