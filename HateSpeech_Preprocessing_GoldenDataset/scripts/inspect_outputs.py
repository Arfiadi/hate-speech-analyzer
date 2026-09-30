from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a short, human-readable analysis of preprocessing outputs.")
    parser.add_argument("--run-name", default="preprocessing_v1")
    args = parser.parse_args()
    run_dir = ROOT / "artifacts" / args.run_name
    with (run_dir / "reports" / "quality_report.json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    with (run_dir / "reports" / "selected_strategy.json").open("r", encoding="utf-8") as handle:
        selected = json.load(handle)["selected_settings"]
    results = pd.read_csv(run_dir / "reports" / "experiment_results.csv")

    print("=== ANALISIS SINGKAT PREPROCESSING ===")
    print("Strategi Golden Dataset:", selected)
    print(f"Input {report['input_rows']} baris -> Golden Dataset {report['golden_rows']} baris.")
    print(
        f"Siap model: {report['model_ready_rows']} | spam/noise dikeluarkan: {report['spam_or_noise_excluded']} "
        f"| spam tidak pasti: {report['spam_label_uncertain_excluded']} "
        f"| tie yang direview/diproses: {report['toxicity_tie_excluded'] + report['soft_ties_for_training']}."
    )
    print("Quality gate:", "PASS" if report["quality_gate_passed"] else "CHECK")
    print("\nPemenang tiap tahap (Toxic F1 tertinggi):")
    for stage, group in results.groupby("stage", sort=False):
        valid = group.loc[group["status"] == "OK"].sort_values(["f1_toxic", "macro_f1"], ascending=False)
        if valid.empty:
            print(f"- {stage}: belum dapat dinilai; cek jumlah data per kelas.")
        else:
            winner = valid.iloc[0]
            print(f"- {stage}: {winner['tested_option']} (Toxic F1={winner['f1_toxic']:.4f}, Macro F1={winner['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
