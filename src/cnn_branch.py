"""
cnn_branch.py
-------------
Lightweight CNN branch for the hybrid model.

v2 used MobileNetV2 pretrained on ImageNet, almost entirely frozen; on real
data it trained to barely above the random-guess baseline (~33% on 5
classes) -- ImageNet's filters have little relevant structure for noise
residuals. v2.2 replaced it with a small CNN trained from scratch, which
fixed that (test accuracy jumped from ~50% to ~77% hybrid), but on a real
run its OWN training log showed a different problem: train accuracy rising
smoothly (47% -> 63%) while val_loss became wildly unstable (spiking as
high as 28+), and its solo accuracy was actually below chance (19% on 5
classes). That pattern -- confident, unstable, wrong -- is classic
overfitting-to-spurious-detail: with no regularization at all, a network
this size can and will memorize incidental leftover scene texture in a
handful of training patches that happens to correlate with camera label by
chance in that split, rather than the actual sensor fingerprint, and that
"knowledge" actively hurts it on validation patches where the coincidence
doesn't hold.

This version adds three standard, well-understood countermeasures instead
of just crossing fingers on more data:
    1. L2 weight regularization on every conv/dense layer, so large,
       confident weights (which is what "memorize this one patch" looks
       like numerically) are penalized rather than free.
    2. Label smoothing (0.1) on the loss, which caps how confident the
       model is allowed to be even when it's right -- this directly
       prevents the "wildly overconfident and wrong" failure mode that
       produced those val_loss spikes, since the loss can no longer reach
       enormous values for a single misclassified example.
    3. A lower learning rate (3e-4 instead of 1e-3), since the instability
       symptom (loss exploding between epochs, not just failing to
       improve) usually means the optimizer step size is too large for
       the loss landscape here, not too small.
"""

from tensorflow.keras import layers, models, optimizers, regularizers


def build_cnn_branch(num_classes: int = 5, input_shape=(96, 96, 3),
                      dropout_rate: float = 0.5, learning_rate: float = 3e-4,
                      l2: float = 1e-4, label_smoothing: float = 0.1) -> models.Model:
    """Build and compile the lightweight, from-scratch patch classifier.

    Args:
        num_classes: number of camera classes.
        input_shape: patch size, must match dataset.py's patch_size.
        dropout_rate: dropout before the final classification layer.
        learning_rate: Adam learning rate.
        l2: L2 weight regularization strength applied to every conv/dense
            kernel.
        label_smoothing: softens one-hot targets (e.g. [0,1,0,0,0] becomes
            roughly [0.02,0.92,0.02,0.02,0.02]) so the loss can't blow up
            to huge values on a single confidently-wrong prediction.
    """
    reg = regularizers.l2(l2)

    inputs = layers.Input(shape=input_shape)
    x = inputs
    # 4 conv blocks: 96 -> 48 -> 24 -> 12 -> 6 spatial size.
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                           kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=_sparse_ce_with_label_smoothing(label_smoothing),
        metrics=["accuracy"],
    )
    return model


def _sparse_ce_with_label_smoothing(label_smoothing: float):
    """Keras' built-in label_smoothing only exists for the one-hot
    (CategoricalCrossentropy) loss, but dataset.py produces plain integer
    labels (sparse). This wraps sparse integer labels so label smoothing
    can still be applied, without requiring a one-hot rewrite of the whole
    data pipeline."""
    import tensorflow as tf

    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    def loss_fn(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32),
                                    depth=tf.shape(y_pred)[-1])
        return cce(y_true_onehot, y_pred)

    return loss_fn


if __name__ == "__main__":
    model = build_cnn_branch()
    model.summary()
