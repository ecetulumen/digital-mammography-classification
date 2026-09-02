#!/usr/bin/env python3
"""Apply a configurable preprocessing pipeline to a directory of images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mammography.data import IMAGE_EXTENSIONS  # noqa: E402
from mammography.preprocessing import preprocess_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--denoise",
        choices=("none", "median", "gaussian", "bilateral", "nlm", "wiener"),
        required=True,
    )
    parser.add_argument(
        "--enhance", choices=("none", "clahe", "histogram"), default="clahe"
    )
    parser.add_argument(
        "--noise-variance",
        type=float,
        default=None,
        help="Optional synthetic Gaussian-noise variance; disabled by default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [
        path
        for path in sorted(args.input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not sources:
        raise SystemExit(f"No supported images found under {args.input_dir}")

    for index, source in enumerate(sources):
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read {source}")
        processed = preprocess_image(
            image,
            denoise_method=args.denoise,
            enhancement_method=args.enhance,
            noise_variance=args.noise_variance,
            seed=args.seed + index,
        )
        destination = args.output_dir / source.relative_to(args.input_dir)
        destination = destination.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), processed):
            raise RuntimeError(f"Could not write {destination}")

    print(f"Processed {len(sources)} images into {args.output_dir}")


if __name__ == "__main__":
    main()
