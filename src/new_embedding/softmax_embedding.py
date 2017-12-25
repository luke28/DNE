import numpy as np
import tensorflow as tf

class NodeEmbedding(object):
    def __init__(self, params, init_embeddings, init_weights):
        self.num_node, self.embedding_size = init_embeddings.shape
        self.num_node += 1
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.tol = params["tol"] if "tol" in params else 0.001

        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            self.init_embeddings = tf.constant(init_embeddings)
            self.init_weights = tf.constant(init_weights)

            self.y_in_ = tf.placeholder(tf.float32, shape = [self.num_nodes])
            self.y_out_ = tf.placeholder(tf.float32, shape = [self.num_nodes])

            self.embedding = tf.Variable(tf.random_uniform( 1, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.weight = tf.Variable(tf.random_uniform(self.embedding_size, 1], -1.0, 1.0), dtype = tf.float32)

            self.embed = tf.concat(
                    self.init_embeddings, self.embedding)
            self.w = tf.concat(
                    self.init_weights, self.weight, 1)

            self.y_in = tf.nn.softmax(tf.matmul(self.embedding, self.w))
            self.y_out = tf.nn.softmax(tf.matmul(self.embed, self.weight))

            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.y_sof, 1e-10, 1.0)), reduction_indices=[1]))

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(getattr(self, self.loss_func))

            #self.train_step = tf.train.AdamOptimizer(self.learnRate).minimize(self.cross_entropy)

    def train(self, get_batch, epoch_num = 10001, save_path = None):
        print("neural embedding: ")
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            pre = float('inf')
            for i in xrange(epoch_num):
                batch_nodes, batch_y = get_batch(self.batch_size)
                self.train_step.run({self.x_nodes : batch_nodes, self.y_ : batch_y})
                if (i % 100 == 0):
                    loss = getattr(self, self.loss_func).eval({self.x_nodes : batch_nodes, self.y_ : batch_y})
                    if (i % 1000 == 0):
                        print(loss)
                    if abs(loss - pre) < self.tol:
                        break
                    else:
                        pre = loss
            if save_path is not None:
                saver = tf.train.Saver()
                saver.save(sess, save_path)
            return sess.run(self.embeddings)

    def load_model(self, save_path):
        with tf.Session(graph = self.tensor_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            return sess.run(self.embeddings)

