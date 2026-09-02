#!/usr/bin/env python3
"""Create patient-wise train, validation, and test folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mammography.data import (  # noqa: E402
    discover_images,
    materialize_splits,
    patient_wise_split,
    summarize_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_images(args.raw_dir)
    splits = patient_wise_split(
        records,
        n_splits=args.n_splits,
        test_fold=args.test_fold,
        validation_fold=args.validation_fold,
        seed=args.seed,
    )
    manifest = materialize_splits(splits, args.output_dir, args.raw_dir)
    for split_name, counts in summarize_splits(splits).items():
        summary = ", ".join(f"{name}={counts.get(name, 0)}" for name in sorted(counts))
        print(f"{split_name}: {summary}")
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()

