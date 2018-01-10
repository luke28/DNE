import sys
import os
import json
import numpy as np
import time

from utils.env import *
from utils.data_handler import DataHandler as dh

def init(params, metric, output_path, draw):
    # load graph structure
    def load_data(params):
        params["network_file"] = os.path.join(DATA_PATH, params["network_file"])
        G = getattr(dh, params["func"])(params)
        return G

    # get initial embedding results
    def init_train(G, params):
        module_embedding = __import__(
                "init_embedding." + params["func"], fromlist = ["init_embedding"]).NodeEmbedding

        ne = module_embedding(params, G)
        embeddings, weights = ne.train()
        return embeddings, weights

    G = load_data(params["load_data"])
    embeddings, weights = init_train(G, params["init_train"])
    metric(embeddings)
    draw(embeddings)
    return G, embeddings, weights

