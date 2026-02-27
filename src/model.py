import tensorflow as tf


def _get_backbone(model_name: str, input_shape):
    model_name = model_name.lower()
    if model_name == "mobilenetv3":
        backbone = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    elif model_name == "efficientnetv2b0":
        backbone = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return backbone, preprocess


def build_model(config: dict, num_classes: int):
    img_size = int(config["img_size"])
    input_shape = (img_size, img_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="image")

    backbone, preprocess = _get_backbone(config["model_name"], input_shape)
    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(int(config.get("dense_units", 512)), activation="relu")(x)
    x = tf.keras.layers.Dropout(float(config.get("dropout", 0.5)))(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{config['model_name']}_classifier")
    return model, backbone


def set_trainable_for_phase(backbone: tf.keras.Model, freeze: bool, unfreeze_layers: int = 0):
    if freeze:
        backbone.trainable = False
        return

    backbone.trainable = True
    if unfreeze_layers <= 0:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        return

    cutoff = max(len(backbone.layers) - unfreeze_layers, 0)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cutoff
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
