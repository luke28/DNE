import numpy as np
import tensorflow as tf
import math

from utils.env import *
from utils.data_handler import DataHandler as dh
class NodeEmbedding(object):
    def __init__(self, params, init_embeddings, init_weights, G, new_id):
        self.num_nodes, self.embedding_size = init_embeddings.shape

        self.batch_size = params["batch_size"]
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.tol = params["tol"] if "tol" in params else 0.0001
        self.neighbor_size = params["neighbor_size"]
        self.negative_distortion = params["negative_distortion"]
        self.num_sampled = params["num_sampled"]
        self.epoch_num = params["epoch_num"]

        self.bs = __import__("batch_strategy." + params["batch_strategy"]["func"], fromlist = ["batch_strategy"]).BatchStrategy(    G, new_id, params["batch_strategy"])
        unigrams = None
        if "negative_sampling_distribution" in params:
            unigrams = getattr(dh, params["negative_sampling_distribution"]["func"])(G, self.num_nodes, params["negative_sampling_distribution"])

        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            tf.set_random_seed(157)
            self.init_embeddings = tf.constant(init_embeddings)
            self.init_weights = tf.constant(init_weights)

            self.labels = tf.placeholder(tf.int64, shape = [None, self.neighbor_size])

            self.embedding = tf.Variable(tf.random_uniform([1, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.nce_biases = tf.zeros([self.num_nodes], tf.float32)

            #self.embed = tf.concat([self.init_embeddings, self.embedding], 0)
            #self.w = tf.concat([self.init_weights, self.weight], 0)

            self.embedding_tile = tf.tile(self.embedding, [self.batch_size, 1])

            if unigrams is None:
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.init_weights,
                        biases = self.nce_biases,
                        labels = self.labels,
                        inputs = self.embedding_tile,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes,
                        num_true = self.neighbor_size))
            else:
                self.sampled_values = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes = self.labels,
                    num_true = self.neighbor_size,
                    num_sampled = self.num_sampled,
                    unique = False,
                    range_max = self.num_nodes,
                    distortion = self.negative_distortion,
                    unigrams = unigrams)
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.init_weights,
                        biases = self.nce_biases,
                        labels = self.labels,
                        inputs = self.embedding_tile,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes,
                        num_true = self.neighbor_size,
                        sampled_values = self.sampled_values))

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)

    def train(self):
        print("new embedding: ")
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(self.epoch_num):
                batch_labels = self.bs.get_batch(self.batch_size)
                if i % 500 == 0:
                    print (self.loss.eval({self.labels : batch_labels}))
                self.train_step.run({self.labels : batch_labels})
            return sess.run(self.embedding)
