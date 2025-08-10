import inspect
import tensorflow as tf
from tensorflow.keras import backend as K

# ─────────────────────────────────────────
# Dice – as before (kept unchanged here) …
# ─────────────────────────────────────────
def dice_coef(y_true, y_pred, smooth: float = 1.0):
    y_pred = tf.nn.sigmoid(y_pred)
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, smooth=1e-5, name="dice_coef", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.inter = self.add_weight(name="inter",  initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        self.inter.assign_add(tf.reduce_sum(y_true_f * y_pred_f))
        self.union.assign_add(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

    def result(self):
        return (2.0 * self.inter + self.smooth) / (self.union + self.smooth)

    def reset_states(self):
        self.inter.assign(0.0)
        self.union.assign(0.0)
