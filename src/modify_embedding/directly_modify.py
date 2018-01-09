import numpy as np
import tensorflow as tf
import math
import sys


class ModifyEmbedding(object):
    def __init__(self, params, w, c, G):
        self.num_nodes, self.embedding_size = w.shape
        self.batch_size = params["batch_size"]
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.tol = params["tol"] if "tol" in params else 0.001
        self.lbd = params["lambda"]
        self.alpha = params["alpha"]
        self.epoch_num = params["epoch_num"]

        self.bs = __import__(
                "batch_strategy." + params["batch_strategy"]["func"],
                fromlist = ["batch_strategy"]
                ).BatchStrategy(G, params["batch_strategy"])


        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            tf.set_random_seed(157)
            self.init_w = tf.constant(w)
            self.init_c = tf.constant(c)

            self.idx_w = tf.placeholder(tf.int64, shape = [None])
            self.idx_c = tf.placeholder(tf.int64, shape = [None])
            self.delta = tf.placeholder(tf.float32, shape = [None])

            self.delta_w = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0))
            self.delta_c = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0))

            self.gathered_delta_w = tf.gather(self.delta_w, self.idx_w)
            self.gathered_delta_c = tf.gather(self.delta_c, self.idx_c)
            self.gathered_init_w = tf.gather(self.init_w, self.idx_w)
            self.gathered_init_c = tf.gather(self.init_c, self.idx_c)

            self.dot1 = tf.reduce_sum(tf.multiply(self.gathered_delta_w, self.gathered_init_c), 1)
            self.dot2 = tf.reduce_sum(tf.multiply(self.gathered_delta_c, self.gathered_init_w), 1)
            self.dot3 = tf.reduce_sum(tf.multiply(self.gathered_delta_w, self.gathered_delta_c), 1)

            self.loss = tf.reduce_mean(tf.square(self.dot1 + self.dot2 + self.dot3 - self.delta)) + self.lbd * tf.norm(self.gathered_delta_w) + self.alpha * tf.norm(self.gathered_delta_c)

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)

            #self.train_step = tf.train.AdamOptimizer(self.learnRate).minimize(self.cross_entropy)

    def train(self, save_path = None):
        print("modify embedding: ")
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            pre = float('inf')
            for i in xrange(self.epoch_num):
                batch_idx_w, batch_idx_c, batch_delta = self.bs.get_batch(self.batch_size)
                self.train_step.run({self.idx_w : batch_idx_w, self.idx_c : batch_idx_c, self.delta : batch_delta})
                if (i % 100 == 0):
                    loss = self.loss.eval({self.idx_w : batch_idx_w, self.idx_c : batch_idx_c, self.delta : batch_delta})
                    if (i % 1000 == 0):
                        print(loss)
                    if abs(loss - pre) < self.tol:
                        break
                    else:
                        pre = loss
            if save_path is not None:
                saver = tf.train.Saver()
                saver.save(sess, save_path)
            return sess.run(self.init_w + self.delta_w), sess.run(self.init_c + self.delta_c)

    def load_model(self, save_path):
        with tf.Session(graph = self.tensor_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            return sess.run(self.embeddings)

