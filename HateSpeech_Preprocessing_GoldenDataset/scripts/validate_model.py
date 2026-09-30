from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import load_config
from src.validation.recheck import validate_saved_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Recheck a saved baseline against the held-out test split.")
    parser.add_argument("--run-name", default="preprocessing_v1")
    parser.add_argument("--config", default=ROOT / "config" / "experiment.yaml", type=Path)
    args = parser.parse_args()
    threshold = float(load_config(args.config)["evaluator"]["threshold"])
    metrics = validate_saved_baseline(ROOT / "artifacts" / args.run_name, threshold)
    print("=== VALIDATION RECHECK ===")
    print(metrics)


if __name__ == "__main__":
    main()
