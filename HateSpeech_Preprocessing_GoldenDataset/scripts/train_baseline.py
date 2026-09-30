from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.baseline import train_baseline
from src.preprocessing.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small baseline to verify the Golden Dataset handoff.")
    parser.add_argument("--run-name", default="preprocessing_v1")
    parser.add_argument("--config", default=ROOT / "config" / "experiment.yaml", type=Path)
    args = parser.parse_args()
    metrics = train_baseline(ROOT / "artifacts" / args.run_name, load_config(args.config))
    print("=== BASELINE SELESAI ===")
    print("Validation:", metrics["validation"])
    print("Test      :", metrics["test"])
    print("Lanjutkan : python scripts/validate_model.py --run-name", args.run_name)


if __name__ == "__main__":
    main()
