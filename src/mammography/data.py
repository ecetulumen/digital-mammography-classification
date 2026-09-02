"""Dataset discovery, patient-wise splitting, and TensorFlow input helpers."""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .labels import CLASS_NAMES, birads_to_class, infer_birads, infer_patient_id

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pgm"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    patient_id: str
    birads: str
    class_name: str


def discover_images(raw_dir: str | Path) -> list[ImageRecord]:
    """Discover labeled images below ``raw_dir``."""

    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw image directory not found: {root}")

    records: list[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            birads = infer_birads(path.relative_to(root))
            records.append(
                ImageRecord(
                    path=path,
                    patient_id=infer_patient_id(path),
                    birads=birads,
                    class_name=birads_to_class(birads),
                )
            )
    if not records:
        raise ValueError(f"No supported images were found under {root}")
    return records


def patient_wise_split(
    records: Sequence[ImageRecord],
    *,
    n_splits: int = 5,
    test_fold: int = 0,
    validation_fold: int = 1,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Create stratified train/validation/test splits without patient overlap."""

    if n_splits < 3:
        raise ValueError("n_splits must be at least 3")
    if test_fold == validation_fold:
        raise ValueError("test_fold and validation_fold must be different")

    y = np.asarray([record.class_name for record in records])
    groups = np.asarray([record.patient_id for record in records])
    paths = np.asarray([str(record.path) for record in records])

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(paths, y, groups))
    test_indices = set(folds[test_fold % n_splits][1].tolist())
    validation_indices = set(folds[validation_fold % n_splits][1].tolist())

    splits = {"train": [], "val": [], "test": []}
    for index, record in enumerate(records):
        if index in test_indices:
            splits["test"].append(record)
        elif index in validation_indices:
            splits["val"].append(record)
        else:
            splits["train"].append(record)

    patient_sets = {
        name: {record.patient_id for record in subset} for name, subset in splits.items()
    }
    if patient_sets["train"] & patient_sets["val"]:
        raise RuntimeError("Patient overlap detected between train and validation splits")
    if patient_sets["train"] & patient_sets["test"]:
        raise RuntimeError("Patient overlap detected between train and test splits")
    if patient_sets["val"] & patient_sets["test"]:
        raise RuntimeError("Patient overlap detected between validation and test splits")
    return splits


def materialize_splits(
    splits: dict[str, Sequence[ImageRecord]], output_dir: str | Path, raw_dir: str | Path
) -> Path:
    """Copy split images and write a manifest. The destination must be empty."""

    output = Path(output_dir).expanduser().resolve()
    source_root = Path(raw_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new directory to avoid stale files."
        )
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for split_name, records in splits.items():
        for record in records:
            relative = record.path.relative_to(source_root)
            suffix = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:8]
            destination_name = f"{record.patient_id}_{suffix}{record.path.suffix.lower()}"
            destination = output / split_name / record.class_name / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.path, destination)
            rows.append(
                {
                    "source": relative.as_posix(),
                    "patient_id": record.patient_id,
                    "birads": record.birads,
                    "class_name": record.class_name,
                    "split": split_name,
                    "destination": destination.relative_to(output).as_posix(),
                }
            )

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def summarize_splits(splits: dict[str, Iterable[ImageRecord]]) -> dict[str, Counter[str]]:
    return {
        split_name: Counter(record.class_name for record in records)
        for split_name, records in splits.items()
    }


def load_image_dataset(
    directory: str | Path,
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
):
    """Load a directory dataset with a stable class order."""

    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def adapt_dataset(dataset, model_name: str, *, training: bool):
    """Apply train-only augmentation and backbone-specific preprocessing."""

    import tensorflow as tf

    name = model_name.lower()
    preprocessors = {
        "vgg16": tf.keras.applications.vgg16.preprocess_input,
        "vgg19": tf.keras.applications.vgg19.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
        "resnet101": tf.keras.applications.resnet.preprocess_input,
        "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
        "densenet121": tf.keras.applications.densenet.preprocess_input,
    }
    try:
        preprocess = preprocessors[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="train_augmentation",
    )

    def transform(images, labels):
        images = tf.cast(images, tf.float32)
        if training:
            images = augmentation(images, training=True)
        return preprocess(images), labels

    return dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )


def class_weights_from_directory(train_dir: str | Path) -> dict[int, float]:
    """Calculate balanced weights from the split directory."""

    root = Path(train_dir)
    counts = np.asarray(
        [
            sum(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in (root / class_name).iterdir()
            )
            for class_name in CLASS_NAMES
        ],
        dtype=float,
    )
    if np.any(counts == 0):
        raise ValueError(f"Every class must contain images. Counts: {counts.tolist()}")
    total = counts.sum()
    return {index: float(total / (len(counts) * count)) for index, count in enumerate(counts)}

