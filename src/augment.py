from dataclasses import dataclass

import tensorflow as tf


@dataclass
class AugmentConfig:
    mode: str = "light"
    allow_horizontal_flip: bool = False


def _apply_keras_spatial_ops(image: tf.Tensor, mode: str) -> tf.Tensor:
    if mode == "light":
        rotation = 0.04
        translate = 0.04
        zoom = 0.05
    else:
        rotation = 0.06
        translate = 0.06
        zoom = 0.08

    image = tf.keras.layers.RandomRotation(factor=rotation, fill_mode="reflect")(image)
    image = tf.keras.layers.RandomTranslation(height_factor=translate, width_factor=translate, fill_mode="reflect")(image)
    image = tf.keras.layers.RandomZoom(height_factor=(-zoom, zoom), width_factor=(-zoom, zoom), fill_mode="reflect")(image)
    return image


def augment_image(image: tf.Tensor, aug_cfg: dict | None = None) -> tf.Tensor:
    aug_cfg = aug_cfg or {}
    mode = aug_cfg.get("aug_mode", aug_cfg.get("mode", "light"))
    allow_horizontal_flip = bool(aug_cfg.get("allow_horizontal_flip", False))

    image = tf.cast(image, tf.float32)
    image = _apply_keras_spatial_ops(image, mode)

    brightness_delta = 0.04 if mode == "light" else 0.06
    contrast_lower, contrast_upper = (0.96, 1.04) if mode == "light" else (0.94, 1.06)
    noise_std = 2.0 if mode == "light" else 3.5

    image = tf.image.random_brightness(image, max_delta=brightness_delta * 255.0)
    image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)

    if allow_horizontal_flip:
        image = tf.image.random_flip_left_right(image)

    noise = tf.random.normal(shape=tf.shape(image), stddev=noise_std)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image
