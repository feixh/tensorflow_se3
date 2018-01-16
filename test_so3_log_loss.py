import so3
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    rot = tf.placeholder(dtype=tf.float32, shape=(None, 3, 3))
    ref_rot = tf.placeholder(dtype=tf.float32, shape=(None, 3, 3))
    dw = so3.batch_log(tf.matmul(tf.matrix_transpose(ref_rot), rot))
    norm_dw = tf.norm(dw, axis=1)

    with tf.Session() as sess:
        result = sess.run(norm_dw,
                          feed_dict={ref_rot: np.random.random([10, 3, 3]).astype(np.float32),
                                     rot: np.random.random([10, 3, 3]).astype(np.float32)})
        assert result.shape[0] == 10
        print result
