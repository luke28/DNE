import sys
import os
import json
import numpy as np
import time
import datetime

from utils.env import *
from utils.data_handler import DataHandler as dh

def loop(params, G, embeddings, weights, metric, output_path, draw):
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])
    module_next = __import__(
            "get_next." + params["get_next"]["func"], fromlist = ["get_next"]).GetNext
    gn = module_next(params["get_next"])

    params_new = params["new_embedding"]
    module_new_embedding = __import__(
            "init_embedding." + params_new["func"], fromlist = ["new_embedding"]).NodeEmbedding
    def new_embedding(G, init_embeddings, init_weights, n):
        ne = module_new_embedding(params_new, G)
        embeddings, weights = ne.train()
        return embeddings, weights
    time_path = output_path + "_time"
    dynamic_embeddings = []
    while True:
        num_new = gn.get_next(G)
        if num_new == 0:
            break
        st = datetime.datetime.now()
        embeddings, weights = new_embedding(G, embeddings, weights, num_new)
        ed = datetime.datetime.now()
        dh.append_to_file(time_path, str(ed - st) + "\n")
        res = metric(embeddings)
        draw(embeddings)
        dynamic_embeddings.append({"embeddings": embeddings.tolist(), "weights": weights.tolist()})

    with open(output_path + "_dynamic", "w") as f:
        f.write(json.dumps(dynamic_embeddings))

