import sys
import os
import re
import networkx as nx
import random
import numpy as np

class BatchStrategy(object):
    # G is a DiGraph with edge weights
    def __init__(self, G, params = None):
        self.edges = G.edges()
        self.delta = [G[i[0]][i[1]]['delta'] for i in self.edges]
        self.n = len(self.edges)

    def get_batch(self, batch_size):
        batch_idx_w = []
        batch_idx_c = []
        batch_delta = []
        for _ in xrange(batch_size):
            idx = random.randrange(self.n)
            batch_idx_w.append(self.edges[idx][0])
            batch_idx_c.append(self.edges[idx][1])
            batch_delta.append(self.delta[idx])
        return np.array(batch_idx_w, dtype = np.int32), np.array(batch_idx_c, dtype = np.int32), np.array(batch_delta)

