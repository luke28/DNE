import numpy as np
import tensorflow as tf
import math
import sys
import datetime

class ModifyEmbedding(object):
    def __init__(self, params, w, c, G):
        self.num_nodes, self.embedding_size = w.shape
        params["optimizer"]["embedding_size"] = self.embedding_size
        self.epoch_num = params["epoch_num"]
        self.lbd = params["lambda"]
        self.alpha = params["alpha"]

        self.G = G
        self.w = w
        self.c = c
        self.w_id = [{} for _ in xrange(self.num_nodes)]
        self.c_id = [{} for _ in xrange(self.num_nodes)]
        self.w_x = [[] for _ in xrange(self.num_nodes)]
        self.c_x = [[] for _ in xrange(self.num_nodes)]
        # for cal w_u, store c
        self.w_init = [[] for _ in xrange(self.num_nodes)]
        self.c_init = [[] for _ in xrange(self.num_nodes)]
        self.w_delta = [None] * self.num_nodes
        self.c_delta = [None] * self.num_nodes

        for u, v in G.edges():
            self.w_id[u][v] = len(self.c_init[v])
            self.c_id[v][u] = len(self.w_init[u])
            self.w_x[u].append(G[u][v]['delta'])
            self.c_x[v].append(G[u][v]['delta'])
            self.w_init[u].append(c[v])
            self.c_init[v].append(w[u])

        for u in G:
            self.w_x[u] = np.array(self.w_x[u])
            self.c_x[u] = np.array(self.c_x[u])
            self.w_init[u] = np.array(self.w_init[u])
            self.c_init[u] = np.array(self.c_init[u])
            self.w_delta[u] = np.zeros(self.w_init[u].shape, dtype = np.float32)
            self.c_delta[u] = np.zeros(self.c_init[u].shape, dtype = np.float32)

        op = __import__(
                "modify_embedding.optimize." + params["optimizer"]["func"],
                fromlist = ["modify_embedding", "optimize"]).CalculateOptima
        self.w_optimizer = op(
                params["optimizer"],
                self.w_x,
                self.w_init,
                self.w,
                self.w_delta,
                self.lbd)
        self.c_optimizer = op(
                params["optimizer"],
                self.c_x,
                self.c_init,
                self.c,
                self.c_delta,
                self.alpha)

    def train(self):
        print("modify embedding: ")
        delta_w = [None] * self.num_nodes
        delta_c = [None] * self.num_nodes
        for _ in xrange(self.epoch_num):
            st = datetime.datetime.now()
            delta_w = self.w_optimizer.train(self.w_init, self.w_delta)
            ed = datetime.datetime.now()
            print "time:"
            print ed - st

            for u in self.G:
                np.add(self.w[u], delta_w[u], out = self.w[u])
            for u in self.G:
                for c, idx in self.w_id[u].items():
                    np.add(self.c_init[c][idx], delta_w[u], out = self.c_init[c][idx])
                    np.add(self.c_delta[c][idx], delta_w[u], out = self.c_delta[c][idx])


            st = datetime.datetime.now()
            delta_c = self.c_optimizer.train(self.c_init, self.c_delta)
            ed = datetime.datetime.now()
            print "time:"
            print ed - st
            for u in self.G:
                np.add(self.c[u], delta_c[u], out = self.c[u])
            for u in self.G:
                for w, idx in self.c_id[u].items():
                    np.add(self.w_init[w][idx], delta_c[u], out = self.w_init[w][idx])
                    np.add(self.w_delta[w][idx], delta_c[u], out = self.w_delta[w][idx])
        return self.w, self.c
