from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .candidates import CandidateSettings, build_candidate, candidate_summary, prepare_source_frame
from .evaluation import score_candidate


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported value for JSON export: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _export_csv(frame: pd.DataFrame, path: Path) -> None:
    export = frame.copy()
    for column in ("toxicity_votes", "spam_votes"):
        if column in export.columns:
            export[column] = export[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    export.to_csv(path, index=False, encoding="utf-8")


def _candidate_folder_name(option: str) -> str:
    return option.replace("_", "-")


def _run_stage(
    stage_number: int,
    stage_name: str,
    setting_key: str,
    options: list[str],
    selected: dict[str, str],
    source: pd.DataFrame,
    config: dict[str, Any],
    experiments_dir: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    stage_dir = experiments_dir / f"{stage_number:02d}_{stage_name}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for option in options:
        settings_dict = {**selected, setting_key: option}
        settings = CandidateSettings(**settings_dict)
        candidate = build_candidate(source, settings, config)
        candidate_dir = stage_dir / _candidate_folder_name(option)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model_ready = candidate.loc[candidate["use_for_training"]].copy()
        review_queue = candidate.loc[candidate["review_reason"] != "ready_for_training"].copy()
        _export_csv(candidate, candidate_dir / "candidate_all.csv")
        _export_csv(model_ready, candidate_dir / "candidate_model_ready.csv")
        _export_csv(review_queue, candidate_dir / "candidate_review_queue.csv")

        evaluation = score_candidate(candidate, settings.as_dict(), config)
        summary = candidate_summary(candidate)
        manifest = {"settings": settings.as_dict(), "summary": summary, "evaluation": evaluation}
        _write_json(candidate_dir / "candidate_manifest.json", manifest)
        rows.append(
            {
                "stage": stage_name,
                "tested_option": option,
                **settings.as_dict(),
                **summary,
                **evaluation,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(stage_dir / "stage_results.csv", index=False, encoding="utf-8")
    eligible = results.loc[results["status"] == "OK"].copy()
    if not eligible.empty:
        # The first priority is Toxic F1. Macro F1 becomes a deterministic tie breaker.
        best = eligible.sort_values(["f1_toxic", "macro_f1"], ascending=[False, False], kind="mergesort").iloc[0]
        selected[setting_key] = str(best[setting_key])
    else:
        selected[setting_key] = options[0]
    return selected, results


def _quality_report(
    source_rows: int,
    invalid: pd.DataFrame,
    golden: pd.DataFrame,
    settings: dict[str, str],
) -> dict[str, Any]:
    model_ready = golden.loc[golden["use_for_training"]].copy()
    report = {
        "input_rows": int(source_rows),
        "invalid_or_short_rows": int(len(invalid)),
        "golden_rows": int(len(golden)),
        "model_ready_rows": int(len(model_ready)),
        "review_queue_rows": int((golden["review_reason"] != "ready_for_training").sum()),
        "spam_or_noise_excluded": int((golden["review_reason"] == "spam_or_noise").sum()),
        "spam_label_uncertain_excluded": int((golden["review_reason"] == "spam_label_uncertain").sum()),
        "toxicity_tie_excluded": int((golden["review_reason"] == "toxicity_tie_excluded").sum()),
        "soft_ties_for_training": int(golden["review_reason"].isin([
            "soft_toxicity_label_for_training_only",
            "downweighted_soft_toxicity_label_for_training_only",
        ]).sum()),
        "duplicate_source_rows_collapsed": int(golden["source_row_count"].sum() - len(golden)),
        "duplicate_keys_remaining": int(golden["dedup_key"].duplicated().sum()),
        "model_ready_spam_rows": int((model_ready["spam_label"] == 1).sum()),
        "model_ready_hard_label_rows": int(model_ready["toxicity_label"].isin([0, 1]).sum()),
        "selected_settings": settings,
    }
    report["quality_gate_passed"] = bool(
        report["model_ready_spam_rows"] == 0
        and report["model_ready_hard_label_rows"] > 0
        and report["invalid_or_short_rows"] >= 0
    )
    return report


def run_preprocessing_experiments(
    input_path: Path,
    config_path: Path,
    run_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the four-stage preprocessing experiment and persist every candidate."""
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Folder hasil sudah ada: {run_dir}. Gunakan nama run baru atau tambahkan --overwrite."
            )
        shutil.rmtree(run_dir)

    config = load_config(config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments").mkdir()
    (run_dir / "processed").mkdir()
    (run_dir / "reports").mkdir()
    raw = pd.read_csv(input_path)
    source, invalid = prepare_source_frame(raw, config)
    _export_csv(invalid, run_dir / "reports" / "invalid_rows.csv")

    candidate_cfg = config["candidates"]
    selected = {
        "text_cleaning": candidate_cfg["text_cleaning"][0],
        "deduplication": "pool" if "pool" in candidate_cfg["deduplication"] else candidate_cfg["deduplication"][0],
        "tie_handling": "drop" if "drop" in candidate_cfg["tie_handling"] else candidate_cfg["tie_handling"][0],
        "annotator_strategy": "equal" if "equal" in candidate_cfg["annotator_strategy"] else candidate_cfg["annotator_strategy"][0],
    }
    stages = [
        (1, "text_cleaning", "text_cleaning", candidate_cfg["text_cleaning"]),
        (2, "deduplication", "deduplication", candidate_cfg["deduplication"]),
        (3, "tie_handling", "tie_handling", candidate_cfg["tie_handling"]),
        (4, "annotator_strategy", "annotator_strategy", candidate_cfg["annotator_strategy"]),
    ]
    history: list[pd.DataFrame] = []
    for number, stage_name, setting_key, options in stages:
        selected, result = _run_stage(
            number, stage_name, setting_key, options, selected, source, config, run_dir / "experiments"
        )
        history.append(result)

    golden_settings = CandidateSettings(**selected)
    golden = build_candidate(source, golden_settings, config)
    model_ready = golden.loc[golden["use_for_training"]].copy()
    review_queue = golden.loc[golden["review_reason"] != "ready_for_training"].copy()
    high_confidence = model_ready.loc[model_ready["annotator_count"] >= 2].copy()
    _export_csv(golden, run_dir / "processed" / "golden_dataset_all.csv")
    _export_csv(model_ready, run_dir / "processed" / "golden_dataset_model_ready.csv")
    _export_csv(review_queue, run_dir / "processed" / "golden_dataset_review_queue.csv")
    _export_csv(high_confidence, run_dir / "processed" / "golden_dataset_high_confidence.csv")

    history_frame = pd.concat(history, ignore_index=True)
    history_frame.to_csv(run_dir / "reports" / "experiment_results.csv", index=False, encoding="utf-8")
    label_distribution = (
        model_ready.assign(label=model_ready["toxicity_label"].map({0: "non_toxic", 0.5: "soft_tie", 1: "toxic"}))
        .groupby("label", dropna=False)
        .size()
        .reset_index(name="rows")
    )
    label_distribution.to_csv(run_dir / "reports" / "model_ready_label_distribution.csv", index=False, encoding="utf-8")
    report = _quality_report(len(raw), invalid, golden, selected)
    _write_json(run_dir / "reports" / "quality_report.json", report)
    _write_json(
        run_dir / "reports" / "selected_strategy.json",
        {
            "selected_settings": selected,
            "selection_rule": "Highest Toxic F1, then highest Macro F1, using the same grouped baseline.",
            "input_file": str(input_path),
            "config_file": str(config_path),
        },
    )
    _write_json(
        run_dir / "reports" / "model_handoff.json",
        {
            "next_input": "processed/golden_dataset_model_ready.csv",
            "high_confidence_phase": "processed/golden_dataset_high_confidence.csv",
            "do_not_train_with": "processed/golden_dataset_review_queue.csv",
            "non_negotiable_checks": [
                "spam_or_noise must stay excluded from training and validation",
                "validation and test must contain hard labels only",
                "duplicate groups must never cross train validation test boundaries",
            ],
        },
    )
    return {"run_dir": str(run_dir), "report": report, "selected": selected}
