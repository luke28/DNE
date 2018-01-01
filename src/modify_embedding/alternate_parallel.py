import numpy as np
import tensorflow as tf
import math
import sys
from modify_embedding.calculate_optima import CalculateOptima as co
from multiprocessing import Pool
from threading import Thread
import time
import datetime

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


    def train(self):
        delta_w = [None] * self.num_nodes
        delta_c = [None] * self.num_nodes
        def optimize_w(u):
            h = co(self.w_x[u], self.w_init[u], self.w[u], self.w_delta[u], self.lbd)
            delta_w[u] = h.train()
        def optimize_c(u):
            h = co(self.c_x[u], self.c_init[u], self.c[u], self.c_delta[u], self.alpha)
            delta_c[u] = h.train()
        print("modify embedding: ")
        s = np.zeros([self.embedding_size])
        out_w = np.copy(self.w)
        out_c = np.copy(self.c)
        processes_w = [None] * self.num_nodes
        processes_c = [None] * self.num_nodes

        for _ in xrange(self.epoch_num):
            for i in xrange(self.num_nodes):
                processes_w[i] = Thread(target = optimize_w, args = (i, ))
                processes_c[i] = Thread(target = optimize_c, args = (i, ))
            s.fill(0)
            st = datetime.datetime.now()
            for p in processes_w:
                p.start()
            for p in processes_w:
                p.join()
            ed = datetime.datetime.now()
            print ed - st

            for u in self.G:
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
            for p in processes_c:
                p.start()
            for p in processes_c:
                p.join()
            for u in self.G:
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
