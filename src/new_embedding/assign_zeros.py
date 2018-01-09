import numpy as np
import tensorflow as tf
import math

from utils.data_handler import DataHandler as dh

class NodeEmbedding(object):
    def __init__(self, params, init_embeddings, init_weights, G, num_new = 1):
        self.num_nodes, self.embedding_size = init_embeddings.shape
        self.num_new = num_new
        self.init_embeddings, self.init_weights = init_embeddings, init_weights

    def train(self):
        tmp = np.zeros([self.num_new, self.embedding_size], dtype = np.float32)
        return np.concatenate([self.init_embeddings, tmp], axis = 0), np.concatenate([self.init_weights, tmp], axis = 0)
