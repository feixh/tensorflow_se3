from se3 import log, exp

import tensorflow as tf
import cv2
import numpy as np

EPS = 1e-5
NUM_TESTS = 100


class SE3Test(tf.test.TestCase):
    def test_log_exp(self):
        v_tf = tf.placeholder(dtype=tf.float32, shape=(6,))
        g_tf = exp(v_tf)
        vv_tf = log(g_tf)

        for i in range(NUM_TESTS):
            v = np.random.random(6)
            R, _ = cv2.Rodrigues(v[0:3])
            g = np.hstack([R, v[3:, np.newaxis]])

            with self.test_session() as sess:
                # print sess.run(vv_tf, feed_dict={v_tf: v})
                # print v
                self.assertNDArrayNear(g,
                                       sess.run(g_tf, feed_dict={v_tf: v}),
                                       EPS)
                self.assertNDArrayNear(v,
                                       sess.run(vv_tf, feed_dict={v_tf: v}),
                                       EPS)


if __name__ == '__main__':
    tf.test.main()
