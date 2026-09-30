from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import run_preprocessing_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged preprocessing experiments and create a Golden Dataset.")
    parser.add_argument("--input", required=True, type=Path, help="CSV source with text, toxicity, and spam labels.")
    parser.add_argument("--config", default=ROOT / "config" / "experiment.yaml", type=Path)
    parser.add_argument("--run-name", default="preprocessing_v1", help="Name of the saved run under artifacts/.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run with the same name.")
    args = parser.parse_args()
    result = run_preprocessing_experiments(
        input_path=args.input,
        config_path=args.config,
        run_dir=ROOT / "artifacts" / args.run_name,
        overwrite=args.overwrite,
    )
    report = result["report"]
    print("\n=== PREPROCESSING SELESAI ===")
    print("Run folder          :", result["run_dir"])
    print("Strategi terpilih   :", result["selected"])
    print("Golden rows         :", report["golden_rows"])
    print("Model-ready rows    :", report["model_ready_rows"])
    print("Spam/noise dikeluarkan:", report["spam_or_noise_excluded"])
    print("Spam label tidak pasti:", report["spam_label_uncertain_excluded"])
    print("Quality gate        :", "PASS" if report["quality_gate_passed"] else "CHECK")
    print("Lanjutkan dengan    : python scripts/train_baseline.py --run-name", args.run_name)


if __name__ == "__main__":
    main()
