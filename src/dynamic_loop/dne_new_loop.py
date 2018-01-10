import sys
import os
import json
import numpy as np
import time
import datetime
from Queue import PriorityQueue as pq

from utils.env import *
from utils.data_handler import DataHandler as dh

def loop(params, G, embeddings, weights, metric, output_path, draw):
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])
    module_next = __import__(
            "get_next." + params["get_next"]["func"], fromlist = ["get_next"]).GetNext
    gn = module_next(params["get_next"])

    params_dynamic = params["dynamic_embedding"]
    K = params_dynamic["num_sampled"]
    def cal_delta(G, embeddings, weights, num_new):
        num_pre = G.number_of_nodes() - num_new
        delta_real = np.matmul(embeddings, np.transpose(weights))
        for u, v in G.edges():
            if u >= num_pre or v >= num_pre:
                continue
            G[u][v]['delta'] = delta_real[u, v] - np.log(
                    float(G[u][v]['weight'] * G.graph['degree']) / float(G.node[u]['in_degree'] * G.node[v]['out_degree'])) + np.log(K)


    def get_modify_list(G, num_new):
        num_pre = G.number_of_nodes() - num_new
        delta_list = [0.0] * num_pre
        for u, v in G.edges():
            if u >= num_pre or v >= num_pre:
                continue
            delta_list[u] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])
            delta_list[v] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])
        for u in G:
            if u >= num_pre:
                continue
            delta_list[u] /= (G.node[u]['in_degree'] + G.node[u]['out_degree'])

        q = pq()
        for u in G:
            if u >= num_pre:
                continue
            if q.qsize() < params_dynamic['num_modify']:
                q.put_nowait((delta_list[u], u))
                continue
            items = q.get_nowait()
            if items[0] < delta_list[u]:
                q.put_nowait((delta_list[u], u))
            else:
                q.put_nowait(items)
        modify_list = []
        while not q.empty():
            modify_list.append(q.get_nowait()[1])
        return modify_list

    module_dynamic_embedding = __import__(
            "dynamic_embedding." + params_dynamic["func"],
            fromlist = ["dynamic_embedding"]).NodeEmbedding

    time_path = output_path + "_time"
    dynamic_embeddings = []
    while True:
        num_new = gn.get_next(G)
        if num_new == 0:
            break
        cal_delta(G, embeddings, weights, num_new)
        modify_list = get_modify_list(G, num_new)
        ne = module_dynamic_embedding(params_dynamic, embeddings, weights, G, modify_list, num_new)
        
        st = datetime.datetime.now()
        embeddings, weights = ne.train()
        ed = datetime.datetime.now()
        dh.append_to_file(time_path, str(ed - st) + "\n")

        res = metric(embeddings)
        draw(embeddings)
        dynamic_embeddings.append({"embeddings": embeddings.tolist(), "weights": weights.tolist()})

    with open(output_path + "_dynamic", "w") as f:
        f.write(json.dumps(dynamic_embeddings))
