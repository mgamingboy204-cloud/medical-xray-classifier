import tensorflow as tf


def sparse_categorical_crossentropy():
    return tf.keras.losses.SparseCategoricalCrossentropy()


def sparse_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_true_cast = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        y_true_oh = tf.one_hot(y_true_cast, depth=tf.shape(y_pred)[-1], dtype=tf.float32)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)

        focal_weight = alpha * tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(focal_weight * ce)

    return loss
