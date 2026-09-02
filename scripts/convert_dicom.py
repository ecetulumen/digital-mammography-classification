#!/usr/bin/env python3
"""Convert DICOM images to lossless 16-bit PNG while preserving folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


def convert_dicom(source: Path, destination: Path) -> None:
    dataset = pydicom.dcmread(source)
    pixels = apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    low, high = np.percentile(pixels, (0.5, 99.5))
    if high <= low:
        normalized = np.zeros_like(pixels, dtype=np.uint16)
    else:
        normalized = np.clip((pixels - low) / (high - low), 0, 1)
        normalized = np.round(normalized * 65535).astype(np.uint16)

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized, mode="I;16").save(destination, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.rglob("*.dcm"))
    if not sources:
        raise SystemExit(f"No DICOM files found under {args.input_dir}")
    for source in sources:
        relative = source.relative_to(args.input_dir).with_suffix(".png")
        convert_dicom(source, args.output_dir / relative)
    print(f"Converted {len(sources)} DICOM images to {args.output_dir}")


if __name__ == "__main__":
    main()

