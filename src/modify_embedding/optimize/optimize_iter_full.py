import numpy as np
import tensorflow as tf

class CalculateOptima(object):
    def __init__(self, params, x_list, c_list, w_list, delta_c_list, lbd_):
        self.num_nodes = len(x_list)
        self.embedding_size = params["embedding_size"]
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "AdamOptimizer"
        self.tol = params["tol"] if "tol" in params else 0.001
        self.lbd = lbd_
        self.epoch_num = params["epoch_num"]

        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            self.x_list = []
            for x in x_list:
                self.x_list.append(tf.constant(x, shape = (len(x), 1), dtype = tf.float32))

            self.delta_w_list = []
            for _ in xrange(self.num_nodes):
                self.delta_w_list.append(tf.Variable(tf.random_uniform([self.embedding_size, 1], -1.0, 1.0)))

            self.w_list = []
            for w in w_list:
                self.w_list.append(tf.constant(w, shape = (self.embedding_size, 1), dtype = tf.float32))

            self.delta_c_ph_list = []
            for delta_c in delta_c_list:
                self.delta_c_ph_list.append(tf.placeholder(tf.float32, shape = delta_c.shape))

            self.c_ph_list = []
            self.c_list = []
            cnt = 0
            for c in c_list:
                self.c_ph_list.append(tf.placeholder(tf.float32, shape = c.shape))
                self.c_list.append(tf.Variable(self.c_ph_list[-1], trainable = False, name = 'c' + str(cnt)))
                cnt+= 1

            cnt = 0
            self.y_list = []
            for i in xrange(self.num_nodes):
                self.y_list.append(tf.Variable(self.x_list[i] - tf.matmul(self.delta_c_ph_list[i], self.w_list[i]), trainable = False, name = 'y' + str(cnt)))
                cnt += 1

            self.loss_list = []
            for i in xrange(self.num_nodes):
                self.loss_list.append(tf.expand_dims(tf.reduce_mean(tf.square(tf.matmul(self.c_list[i], self.delta_w_list[i]) - self.y_list[i])) + self.lbd * tf.norm(self.delta_w_list[i]), axis = 0))

            self.loss = tf.reduce_mean(tf.concat(self.loss_list, 0))
            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)

        self.sess = tf.Session(graph = self.tensor_graph)


    def train(self, c_list, delta_c_list):
        print("modify embedding: ")
        feed_dict = {}
        for i in xrange(self.num_nodes):
            feed_dict[self.c_ph_list[i]] = c_list[i]
            feed_dict[self.delta_c_ph_list[i]] = delta_c_list[i]
        with self.tensor_graph.as_default():
            self.sess.run(tf.global_variables_initializer(), feed_dict = feed_dict)

        pre = float('inf')
        for i in xrange(self.epoch_num):
            self.sess.run(self.train_step)
            if (i % 100 == 0):
                loss = self.sess.run(self.loss)
                print loss
                if (abs(loss - pre) < self.tol):
                    break
                pre = loss
        return self.sess.run(self.delta_w_list)

    def __del__(self):
        self.sess.close()
