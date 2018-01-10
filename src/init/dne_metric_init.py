import sys
import os
import json
import numpy as np
import time

from utils.env import *
from utils.data_handler import DataHandler as dh

def init(params, metric, output_path, draw):
    embeddings_path = os.path.join(RES_PATH, params["embeddings_path"])
    dic = dh.load_json_file(embeddings_path)
    embeddings = np.array(dic["embeddings"])
    metric(embeddings)
    draw(embeddings)
    return None, None, None

