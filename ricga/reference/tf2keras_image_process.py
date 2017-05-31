import tensorflow as tf


def tf2keras_image(tf_image):
    tf_image = tf_image * 255
    r, g, b = tf.split(tf_image, num_or_size_splits=3, axis=2)
    r -= 103.939
    g -= 116.779
    b -= 123.68
    keras_image = tf.concat([r, g, b], axis=2)
    return keras_image
