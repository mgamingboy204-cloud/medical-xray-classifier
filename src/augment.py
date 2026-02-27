from dataclasses import dataclass

import tensorflow as tf


@dataclass
class AugmentConfig:
    mode: str = "light"
    allow_horizontal_flip: bool = False


_AUGMENTER_CACHE: dict[tuple[str, bool], tf.keras.Sequential] = {}


def _resolve_aug_params(mode: str) -> dict[str, float]:
    if mode == "light":
        return {
            "rotation": 0.04,
            "translate": 0.04,
            "zoom": 0.05,
            "contrast": 0.04,
            "noise_std": 2.0,
        }
    return {
        "rotation": 0.06,
        "translate": 0.06,
        "zoom": 0.08,
        "contrast": 0.06,
        "noise_std": 3.5,
    }


def get_augmenter(aug_cfg: dict | None = None) -> tf.keras.Sequential:
    aug_cfg = aug_cfg or {}
    mode = aug_cfg.get("aug_mode", aug_cfg.get("mode", "light"))
    allow_horizontal_flip = bool(aug_cfg.get("allow_horizontal_flip", False))

    cache_key = (mode, allow_horizontal_flip)
    if cache_key in _AUGMENTER_CACHE:
        return _AUGMENTER_CACHE[cache_key]

    params = _resolve_aug_params(mode)
    layers = [
        tf.keras.layers.RandomRotation(factor=params["rotation"], fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(
            height_factor=params["translate"], width_factor=params["translate"], fill_mode="reflect"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-params["zoom"], params["zoom"]),
            width_factor=(-params["zoom"], params["zoom"]),
            fill_mode="reflect",
        ),
        tf.keras.layers.RandomContrast(factor=params["contrast"]),
    ]
    if allow_horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip(mode="horizontal"))

    augmenter = tf.keras.Sequential(layers, name=f"xray_augmenter_{mode}_{int(allow_horizontal_flip)}")
    augmenter.build((None, None, None, 3))
    _AUGMENTER_CACHE[cache_key] = augmenter
    return augmenter


def augment_image(image: tf.Tensor, augmenter: tf.keras.Sequential, aug_cfg: dict | None = None) -> tf.Tensor:
    aug_cfg = aug_cfg or {}
    mode = aug_cfg.get("aug_mode", aug_cfg.get("mode", "light"))
    params = _resolve_aug_params(mode)
    use_gaussian_noise = bool(aug_cfg.get("gaussian_noise", True))

    image = tf.cast(image, tf.float32)

    if image.shape.rank == 3:
        image = tf.expand_dims(image, axis=0)
        image = augmenter(image, training=True)
        image = tf.squeeze(image, axis=0)
    elif image.shape.rank == 4:
        image = augmenter(image, training=True)
    else:
        rank = tf.rank(image)
        image = tf.cond(
            tf.equal(rank, 3),
            lambda: tf.squeeze(augmenter(tf.expand_dims(image, axis=0), training=True), axis=0),
            lambda: augmenter(image, training=True),
        )

    if use_gaussian_noise:
        noise_std = float(aug_cfg.get("noise_std", params["noise_std"]))
        image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_std, dtype=image.dtype)

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image
