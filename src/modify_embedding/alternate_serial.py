import numpy as np
import tensorflow as tf
import math
import sys
from modify_embedding.calculate_optima import CalculateOptima as co


class ModifyEmbedding(object):
    def __init__(self, params, w, c, G):
        self.num_nodes, self.embedding_size = w.shape
        self.tol = params["tol"] if "tol" in params else 0.01
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
            self.w_delta[u] = np.zeros(self.w_init[u].shape, dtype = np.float64)
            self.c_delta[u] = np.zeros(self.c_init[u].shape, dtype = np.float64)

    def optimize(self, x_, c_, w_, delta_c_, lbd_):
        h = co(x_, c_, w_, delta_c_, lbd_)
        return h.train()

    def train(self):
        print("modify embedding: ")
        delta_w = [None] * self.num_nodes
        delta_c = [None] * self.num_nodes
        s = np.zeros([self.embedding_size])
        out_w = np.copy(self.w)
        out_c = np.copy(self.c)
        for _ in xrange(self.epoch_num):
            s.fill(0)
            for u in self.G:
                delta_w[u] = self.optimize(self.w_x[u], self.w_init[u], self.w[u], self.w_delta[u], self.lbd)
                np.add(s, delta_w[u], out = s)
                np.add(out_w[u], delta_w[u], out = out_w[u])
            for u in self.G:
                for c, idx in self.w_id[u].items():
                    np.add(self.c_init[c][idx], delta_w[u], out = self.c_init[c][idx])
                    np.add(self.c_delta[c][idx], delta_w[u], out = self.c_delta[c][idx])
            delta_mean = np.mean(s) / self.num_nodes
            #print abs(delta_mean)
            if abs(delta_mean) < self.tol:
                break
            s.fill(0)
            for u in self.G:
                delta_c[u] = self.optimize(self.c_x[u], self.c_init[u], self.c[u], self.c_delta[u], self.alpha)
                np.add(s, delta_c[u], out = s)
                np.add(out_c[u], delta_c[u], out = out_c[u])
            for u in self.G:
                for w, idx in self.c_id[u].items():
                    np.add(self.w_init[w][idx], delta_c[u], out = self.w_init[w][idx])
                    np.add(self.w_delta[w][idx], delta_c[u], out = self.w_delta[w][idx])
            delta_mean = np.mean(s) / self.num_nodes
            #print abs(delta_mean)
            if abs(delta_mean) < self.tol:
                break
        return out_w, out_c
