from so3 import tilde, tilde_inv, log, exp
import tensorflow as tf

import numpy as np
import cv2

EPS = 1e-3
NUM_TESTS = 100


class SO3Test(tf.test.TestCase):
    def test_tilde(self):
        v = tf.placeholder(dtype=tf.float32, shape=(3,))
        tv = tilde(v)
        with self.test_session() as sess:
            # d1, d2 = tf.test.compute_gradient(x=v, x_shape=(3,),
            #                                   y=tv, y_shape=(3, 3))
            # print d1
            # print d2
            for i in range(NUM_TESTS):
                init_val = np.random.random(3)
                self.assertLess(
                    tf.test.compute_gradient_error(x=v, x_shape=(3,),
                                                   y=tv, y_shape=(3, 3),
                                                   x_init_value=init_val,
                                                   delta=EPS*0.1),
                    EPS)
                self.assertNDArrayNear(
                    sess.run(tv, feed_dict={v: init_val}),
                    np.array([[0, -init_val[2], init_val[1]],
                              [init_val[2], 0, -init_val[0]],
                              [-init_val[1], init_val[0], 0]]),
                    EPS)

    def test_tilde_inv(self):
        v = tf.placeholder(dtype=tf.float32, shape=(3,))
        tv = tilde(v)
        vv = tilde_inv(tv)

        with self.test_session() as sess:
            init_val = np.random.random(3)
            result = sess.run({'tv': tv, 'vv': vv}, feed_dict={v: init_val})
            self.assertNDArrayNear(result['vv'], init_val, EPS)
            self.assertLess(
                tf.test.compute_gradient_error(x=tv, x_shape=(3, 3),
                                               y=vv, y_shape=(3,),
                                               x_init_value=result['tv'],
                                               delta=EPS*0.1),
                EPS)

    def test_exp0_log0(self):
        R0 = exp(tf.zeros((3,), dtype=tf.float32))
        v0 = log(R0)

        with self.test_session():
            print 'R0=', R0.eval()
            self.assertNDArrayNear(R0.eval(),
                                   np.identity(3, dtype=np.float32),
                                   EPS)
            print 'v0=', v0.eval()
            self.assertNDArrayNear(v0.eval(), np.zeros(3), EPS)

    def test_exp_log(self):
        v0 = tf.placeholder(dtype=tf.float32, shape=(3,))
        R = exp(v0)
        v = log(R)
        with self.test_session() as sess:
            for i in range(100):
                init_val = np.random.random(3)
                result = sess.run({'v': v}, feed_dict={v0: init_val})
                self.assertNDArrayNear(result['v'],
                                       init_val,
                                       EPS)

    def test_exp(self):
        v = tf.placeholder(dtype=tf.float32, shape=(3,))
        R = exp(v)
        with self.test_session() as sess:
            for i in range(NUM_TESTS):
                init_val = np.random.random(3)
                self.assertLess(
                    tf.test.compute_gradient_error(x=v, x_shape=(3,),
                                                   y=R, y_shape=(3, 3),
                                                   x_init_value=init_val,
                                                   delta=EPS*0.1),
                    EPS)
                Rtf = sess.run(R, feed_dict={v: init_val})
                Rcv, _ = cv2.Rodrigues(init_val)
                print 'Rtf=', Rtf
                print 'Rcv=', Rcv
                self.assertNDArrayNear(Rtf, Rcv, EPS)

    def test_log(self):
        R = tf.placeholder(dtype=tf.float32, shape=(3, 3))
        v = log(R)
        with self.test_session() as sess:
            for i in range(NUM_TESTS):
                init_v = np.random.random(3)
                Rcv, _ = cv2.Rodrigues(init_v)
                vtf = sess.run(v, feed_dict={R: Rcv})
                print 'v_true=', init_v
                print 'vtf=', vtf
                self.assertNDArrayNear(vtf, init_v, EPS)


if __name__ == '__main__':
    tf.test.main()
