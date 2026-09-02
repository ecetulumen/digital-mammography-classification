#!/usr/bin/env python3
"""Train and fine-tune one supported transfer-learning model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mammography.data import (  # noqa: E402
    adapt_dataset,
    class_weights_from_directory,
    load_image_dataset,
)
from mammography.models import (  # noqa: E402
    BACKBONES,
    build_classifier,
    enable_fine_tuning,
    image_size_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=40)
    parser.add_argument("--trainable-layers", type=int, default=40)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def callbacks(checkpoint: Path):
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]


def serializable_history(history) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_size = image_size_for(args.model)
    train_raw = load_image_dataset(
        args.data_dir / "train",
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_raw = load_image_dataset(
        args.data_dir / "val",
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    train_ds = adapt_dataset(train_raw, args.model, training=True)
    val_ds = adapt_dataset(val_raw, args.model, training=False)
    class_weights = class_weights_from_directory(args.data_dir / "train")

    model = build_classifier(args.model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.head_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    head_checkpoint = args.output_dir / "best_head.keras"
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks(head_checkpoint),
    )

    model = tf.keras.models.load_model(head_checkpoint)
    enable_fine_tuning(model, args.model, args.trainable_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    fine_tune_checkpoint = args.output_dir / "best_finetuned.keras"
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs + args.fine_tune_epochs,
        initial_epoch=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks(fine_tune_checkpoint),
    )

    history = {
        "head": serializable_history(head_history),
        "fine_tuning": serializable_history(fine_tune_history),
        "configuration": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Best fine-tuned model: {fine_tune_checkpoint}")


if __name__ == "__main__":
    main()

