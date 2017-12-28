import os
import sys
import re
import json
import math
import argparse
import time
import subprocess
import numpy as np
import networkx as nx
import tensorflow as tf
import datetime
from operator import itemgetter

from utils.env import *
from utils.metric import Metric
from utils.data_handler import DataHandler as dh

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def init(params):
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
        embeddings, weights = ne.train(bs.get_batch, 10001)
        return embeddings, weights

    G = load_data(params["load_data"])
    embeddings, weights = init_train(G, params["init_train"])
    return G, embeddings, weights

def metric(embeddings, params):
    for metric in params:
        res = getattr(Metric, metric["func"])(embeddings, metric)
        print res
    return res

def main_loop(params, G, embeddings, weights):
    params["main_loop"]["get_next"]["input_file"] = os.path.join(
            DATA_PATH, params["main_loop"]["get_next"]["input_file"])
    module_next = __import__(
            "get_next." + params["main_loop"]["get_next"]["func"], fromlist = ["get_next"]).GetNext
    gn = module_next(params["main_loop"]["get_next"])

    params_new = params["main_loop"]["new_embedding"]
    module_new_batch = __import__(
            "batch_strategy." + params_new["batch_strategy"]["func"], fromlist = ["batch_strategy"]).BatchStrategy
    module_new_embedding = __import__(
            "new_embedding." + params_new["embedding_model"]["func"], fromlist = ["new_embedding"]).NodeEmbedding

    def new_embedding(G, init_embeddings, init_weights, u):
        unigrams_in = dh.in_degree_distribution(G)
        unigrams_out = dh.out_degree_distribution(G)
        bs = module_new_batch(G, u, params_new["batch_strategy"])
        ne = module_new_embedding(params_new["embedding_model"], init_embeddings, init_weights, unigrams_in, unigrams_out)
        embeddings, weights = ne.train(bs.get_batch, 1001)
        return embeddings, weights

    K = float(params["init_train"]["embedding_model"]["num_sampled"])
    def cal_delta(G, embeddings, weights):
        delta_real = np.matmul(embeddings, np.transpose(weights))
        for u, v in G.edges():
            G[u][v]['delta'] = delta_real[u, v] - np.log(
                    float(G[u][v]['weight'] * G.graph['degree']) / float(G.node[u]['in_degree'] * G.node[v]['out_degree'])) + np.log(K)

    params_modify = params["main_loop"]["modify_embedding"]
    module_modify_embedding = __import__(
            "modify_embedding." + params_modify["func"],
            fromlist = ["modify_embedding"]).ModifyEmbedding
    def modify_embedding(G, w, c):
        ne = module_modify_embedding(params_modify, w, c, G)
        w, c = ne.train()

    #out_f = open("out", "w")
    while True:
        ret = gn.get_next(G)
        if ret == 0:
            break
        n = G.number_of_nodes()
        embeddings, weights = new_embedding(G, embeddings, weights, n - 1)
        res = metric(embeddings, params["metrics"])
        #out_f.write(str(res) + "\n")
        cal_delta(G, embeddings, weights)
        modify_embedding(G, embeddings, weights)
        res = metric(embeddings, params["metrics"])


def main():

    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--operation', type = str, default = "all", help = "[all | init | train | metric | draw]")
    parser.add_argument('--conf', type = str, default = "default")
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))

    if args.operation == "all":
        G, embeddings, weights = init(params)
        metric(embeddings, params["metrics"])
        main_loop(params, G, embeddings, weights)
    elif args.operation == "init":
        init(G, params)
    elif args.operation == "draw":
        pass
    else:
        print "Not Support!"

if __name__ == "__main__":
    main()
