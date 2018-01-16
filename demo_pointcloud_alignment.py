import so3
import cv2
import numpy as np
import tensorflow as tf

NUM_POINTS = 100

if __name__ == '__main__':
    w_gt = np.random.random(3)-0.5
    Rgt, _ = cv2.Rodrigues(w_gt)
    Rgt = Rgt.astype(np.float32)
    src_pc = np.random.random([3, NUM_POINTS])
    tgt_pc = Rgt.dot(src_pc)

    # problem: given two sets of corresponding points
    X_src = tf.placeholder(dtype=tf.float32, shape=(3, NUM_POINTS))
    X_tgt = tf.placeholder(dtype=tf.float32, shape=(3, NUM_POINTS))
    w = tf.Variable(initial_value=w_gt + 0.5 * np.random.randn(3),
                    dtype=tf.float32,
                    trainable=True)

    R = so3.exp(w)
    loss = tf.reduce_sum(tf.squared_difference(X_tgt, tf.matmul(R, X_src)))
    err = tf.norm(so3.log(tf.matmul(R, Rgt.T)))

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'wgt=', w_gt
        print 'w0=', w.eval()
        for i in range(100):
            print sess.run([w, loss, err], feed_dict={X_src: src_pc, X_tgt: tgt_pc})
            sess.run(optimizer, feed_dict={X_src: src_pc, X_tgt: tgt_pc})
