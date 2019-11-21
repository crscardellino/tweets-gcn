import tensorflow as tf


@tf.function
def masked_accuracy(true, pred, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(true, 1), tf.argmax(pred, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


@tf.function
def masked_f1_score(true, pred, mask, macro_average=True):
    """F1-Score with masking"""
    pred = tf.cast(tf.equal(tf.reduce_max(pred, axis=1, keepdims=True), pred), tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    pred *= tf.reshape(mask, shape=(-1, 1))

    TP = tf.reduce_sum(true * pred, axis=0)
    FP = tf.reduce_sum((1-true) * pred, axis=0)
    FN = tf.reduce_sum(true * (1-pred), axis=0)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = 2 * precision * recall / (precision + recall)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)  # Cast to 0 the NaN values

    if macro_average:
        f1 = tf.reduce_mean(f1)

    return f1


@tf.function
def masked_softmax_cross_entropy(true, pred, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=true)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)
