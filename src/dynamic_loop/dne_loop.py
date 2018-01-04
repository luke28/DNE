import sys
import os
import json
import numpy as np
import time

from utils.env import *
from utils.data_handler import DataHandler as dh

def loop(params, G, embeddings, weights, metric, output_path):
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])
    module_next = __import__(
            "get_next." + params["get_next"]["func"], fromlist = ["get_next"]).GetNext
    gn = module_next(params["get_next"])

    params_new = params["new_embedding"]
    module_new_batch = __import__(
            "batch_strategy." + params_new["batch_strategy"]["func"], fromlist = ["batch_strategy"]).BatchStrategy
    module_new_embedding = __import__(
            "new_embedding." + params_new["embedding_model"]["func"], fromlist = ["new_embedding"]).NodeEmbedding

    def new_embedding(G, init_embeddings, init_weights, n):
        unigrams_in = dh.in_degree_distribution(G)
        unigrams_out = dh.out_degree_distribution(G)
        bs = module_new_batch(G, n, params_new["batch_strategy"])
        ne = module_new_embedding(params_new["embedding_model"], init_embeddings, init_weights, unigrams_in, unigrams_out, n)
        embeddings, weights = ne.train(bs.get_batch)
        return embeddings, weights

    K = params_new["embedding_model"]["num_sampled"]
    def cal_delta(G, embeddings, weights):
        delta_real = np.matmul(embeddings, np.transpose(weights))
        for u, v in G.edges():
            G[u][v]['delta'] = delta_real[u, v] - np.log(
                    float(G[u][v]['weight'] * G.graph['degree']) / float(G.node[u]['in_degree'] * G.node[v]['out_degree'])) + np.log(K)

    params_modify = params["modify_embedding"]
    module_modify_embedding = __import__(
            "modify_embedding." + params_modify["func"],
            fromlist = ["modify_embedding"]).ModifyEmbedding
    def modify_embedding(G, w, c):
        ne = module_modify_embedding(params_modify, w, c, G)
        w, c = ne.train()

    while True:
        num_new = gn.get_next(G)
        if num_new == 0:
            break
        embeddings, weights = new_embedding(G, embeddings, weights, num_new)
        res = metric(embeddings)
        cal_delta(G, embeddings, weights)
        modify_embedding(G, embeddings, weights)
        res = metric(embeddings)
