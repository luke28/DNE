import sys
import os
import json
import numpy as np
import time
import datetime
import networkx as nx

from utils.env import *
from utils.data_handler import DataHandler as dh

def loop(params, G, embeddings, weights, metric, output_path, draw):
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])
    module_next = __import__(
            "get_next." + params["get_next"]["func"], fromlist = ["get_next"]).GetNext
    gn = module_next(params["get_next"])

    params_dynamic = params["dynamic_embedding"]
    module_dynamic_embedding = __import__(
            "dynamic_embedding." + params_dynamic["func"],
            fromlist = ["dynamic_embedding"]).NodeEmbedding

    time_path = output_path + "_time"
    init_n = G.number_of_nodes()
    cnt = init_n
    init_embeddings = np.array(embeddings)
    embedding_size = init_embeddings.shape[1]
    while True:
        num_new = gn.get_next(G, init_n)
        if num_new == 0:
            break
        st = datetime.datetime.now()
        for _ in xrange(num_new):
            if G.node[cnt]['out_degree'] == 0:
                embed = np.random.rand(1, embedding_size) * 2.0 - 1.0
            else:
                ne = module_dynamic_embedding(params_dynamic, init_embeddings, weights, G, cnt)
                embed = ne.train()
            np.append(embeddings, embed, axis = 0)
            cnt += 1

        ed = datetime.datetime.now()
        dh.append_to_file(time_path, str(ed - st) + "\n")

        res = metric(embeddings)
        draw(embeddings)
