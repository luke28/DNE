import sys
import os
import json
import numpy as np
import time

from utils.env import *
from utils.data_handler import DataHandler as dh

def init(params, metric, output_path):
    # load graph structure
    def load_data(params):
        params["network_file"] = os.path.join(DATA_PATH, params["network_file"])
        G = getattr(dh, params["func"])(params)
        return G

    # get initial embedding results
    def init_train(G, params):
        module_batch = __import__(
                "batch_strategy." + params["batch_strategy"]["func"], fromlist = ["batch_strategy"]).BatchStrategy

        module_embedding = __import__(
                "init_embedding." + params["embedding_model"]["func"], fromlist = ["init_embedding"]).NodeEmbedding

        unigrams = None
        if "negative_sampling_distribution" in params:
            unigrams = getattr(
                    dh, params["negative_sampling_distribution"]["func"])(G, params["negative_sampling_distribution"])

        bs = module_batch(G, params["batch_strategy"])
        ne = module_embedding(params["embedding_model"], unigrams)
        embeddings, weights = ne.train(bs.get_batch)
        return embeddings, weights

    G = load_data(params["load_data"])
    embeddings, weights = init_train(G, params["init_train"])
    metric(embeddings)
    return G, embeddings, weights

