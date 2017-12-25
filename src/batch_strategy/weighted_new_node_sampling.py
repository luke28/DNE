import sys
import os
import re
import networkx as nx
import random
import numpy as np

from alias_table_sampling import AliasTable as at

class BatchStrategy(object):
    # G is a DiGraph with edge weights
    def __init__(self, G, u, params = None):
        self.id_lst = []
        probs = []
        for v in G[u]:
            probs.append(G[u][v]['weight'])
            self.id_lst.append(v)
        self.sampling_handler = at(probs)

    def get_batch(self, batch_size):
        batch_labels = []
        for _ in xrange(batch_size):
            idx = self.sampling_handler.sample()
            batch_labels.append([self.id_lst[idx]])
        batch_labels = np.array(batch_labels, dtype = np.int32)
        return batch_labels, batch_labels

if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edge(1, 2, weight = 1)
    G.add_edge(3, 4, weight = 2)
    bs = BatchStrategy(G)
    batch_x, batch_y = bs.get_batch(1000)
    m = {}
    for it in batch_x:
        if it in m:
            m[it] += 1
        else:
            m[it] = 1
    print m
    #print batch_x
    #print batch_y
