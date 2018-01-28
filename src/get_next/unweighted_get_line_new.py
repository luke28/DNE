import networkx as nx
import os
import sys

class GetNext(object):
    def __init__(self, params):
        self.f = open(params["input_file"], "r")
        self.is_directed = params["is_directed"]
        self.n = params["num_new_nodes"]

    @staticmethod
    def dict_add(d, key, add):
        if key in d:
            d[key] += add
        else:
            d[key] = add

    def get_next(self, G, init_n):
        for num in xrange(self.n):
            while True:
                line = self.f.readline()
                if not line:
                    return num
                line = line.strip()
                if len(line) == 0:
                    continue
                u_now, m = [int(i) for i in line.split()]
                G.add_node(u_now)
                G.node[u_now]['out_degree'] = 0
                G.node[u_now]['in_degree'] = 0
                break
            for i in xrange(m):
                line = self.f.readline()
                line = line.strip()
                u, v = [int(i) for i in line.split()]
                if u != u_now:
                    if self.is_directed:
                        continue
                    else:
                        u, v = v, u
                if v >= init_n:
                    continue
                G.add_edge(u, v, weight = 1)
                GetNext.dict_add(G.node[u], 'out_degree', 1)
                GetNext.dict_add(G.node[v], 'in_degree', 1)
                GetNext.dict_add(G.graph, 'degree', 1)

        return self.n

    def __del__(self):
        self.f.close()

