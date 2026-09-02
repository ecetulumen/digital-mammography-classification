"""Transfer-learning model factory for the six evaluated CNN backbones."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    constructor_name: str
    image_size: int


BACKBONES = {
    "vgg16": BackboneSpec("VGG16", 224),
    "vgg19": BackboneSpec("VGG19", 224),
    "resnet50": BackboneSpec("ResNet50", 224),
    "resnet101": BackboneSpec("ResNet101", 224),
    "inceptionv3": BackboneSpec("InceptionV3", 299),
    "densenet121": BackboneSpec("DenseNet121", 224),
}


def image_size_for(model_name: str) -> int:
    try:
        return BACKBONES[model_name.lower()].image_size
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def build_classifier(model_name: str, num_classes: int = 3, weights: str = "imagenet"):
    """Build a frozen ImageNet backbone with a regularized classification head."""

    import tensorflow as tf

    name = model_name.lower()
    try:
        spec = BACKBONES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc

    constructor = getattr(tf.keras.applications, spec.constructor_name)
    backbone = constructor(
        weights=weights,
        include_top=False,
        input_shape=(spec.image_size, spec.image_size, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input((spec.image_size, spec.image_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="head_dense",
    )(x)
    x = tf.keras.layers.Dropout(0.5, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)
    return tf.keras.Model(inputs, outputs, name=f"{name}_classifier")


def enable_fine_tuning(model, model_name: str, trainable_layers: int = 40) -> None:
    """Unfreeze the top convolutional layers while keeping BatchNorm stable."""

    import tensorflow as tf

    nested_models = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(nested_models) != 1:
        raise ValueError(
            f"Expected one CNN backbone inside {model_name}, found {len(nested_models)}"
        )
    backbone = nested_models[0]
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - trainable_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
