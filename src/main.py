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

def metric(embeddings, params):
    for metric in params:
        res = getattr(Metric, metric["func"])(embeddings, metric)
        print res
    return res


def main():

    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--operation', type = str, default = "all", help = "[all | init | train | metric | draw]")
    parser.add_argument('--conf', type = str, default = "default")
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))

    if args.operation == "all":
        G, embeddings, weights = __import__("init." + params["init"]["func"], fromlist = ["init"]).init(params["init"], metric, params["metrics"])
        __import__("dynamic_loop." + params["main_loop"]["func"], fromlist = ["dynamic_loop"]).loop(params["main_loop"], G, embeddings, weights, metric, params["metrics"])
    elif args.operation == "init":
        G, embeddings, weights = __import__("init." + params["init"]["func"], fromlist = ["init"]).init(params["init"], metric, params["metrics"])
    elif args.operation == "draw":
        pass
    else:
        print "Not Support!"

if __name__ == "__main__":
    main()
