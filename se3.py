import tensorflow as tf
import so3


def log(g):
    """
    'Logarithm' map of special euclidean group.
    :param g: 3x4 rigid body transformation.
    :return: [w, t]
    """
    return tf.concat([so3.log(g[0:3, 0:3]), g[0:3, 3]], axis=0)


def exp(v):
    """
    'Exponential' map of twist [w, t]
    :param v:
    :return:
    """
    t = tf.expand_dims(v[3:6], 1)
    return tf.concat([so3.exp(v[0:3]), t], axis=1)

