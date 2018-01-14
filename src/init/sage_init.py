import sys
import os
import json
import numpy as np
import time
import datetime

from contrast_experiment.graphsage import unsupervised_train as ut
from utils.data_handler import DataHandler as dh

def init(params, metric, output_path, draw):
    params['output_path'] = output_path

    time_path = output_path + "_time"
    
    start_time = datetime.datetime.now()
    ut.sage_main(params)
    train_time = datetime.datetime.now()-start_time
    print("the train_time is" + str(train_time))
    dh.append_to_file(time_path, str(train_time)+"\n")
    G = None
    embedding = None
    weight = None
    return G, embedding, weight 
