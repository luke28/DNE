import numpy as np
import tensorflow as tf

class CalculateOptima(object):
    def __init__(self, x_, c_, w_, delta_c_, lbd_):
        n, d = c_.shape
        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            x = tf.constant(x_, shape = (n, 1), dtype = tf.float64)
            c = tf.constant(c_, dtype = tf.float64)
            delta_c = tf.constant(delta_c_, dtype = tf.float64)
            lbd = tf.constant(lbd_, dtype = tf.float64)
            w = tf.constant(w_, shape = (d, 1), dtype = tf.float64)

            self.delta_w = tf.matmul(tf.matmul(tf.matrix_inverse(lbd * tf.eye(d, dtype = tf.float64) - tf.matmul(c, c, transpose_a = True)), c, transpose_b = True), x - tf.matmul(delta_c, w))

    def train(self, epoch_num = None):
        with tf.Session(graph = self.tensor_graph) as sess:
            return sess.run(tf.reshape(self.delta_w, [-1]))
