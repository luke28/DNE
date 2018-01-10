import sys
import os
import json
import numpy as np
import time
from Queue import PriorityQueue as pq

from utils.env import *
from utils.data_handler import DataHandler as dh

def loop(params, G, embeddings, weights, metric, output_path, draw):
    embeddings_path = os.path.join(RES_PATH, params["embeddings_path"])
    dynamic_embeddings = dh.load_json_file(embeddings_path)
    for items in dynamic_embeddings:
        embeddings = np.array(items["embeddings"])
        metric(embeddings)
        draw(embeddings)
