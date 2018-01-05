import sys
import os
import json
import numpy as np
import time

from contrast_experiment.graphsage import unsupervised_train as ut

def init(params, metric, output_path):
    params['output_path'] = output_path
    start_time = time.time()
    ut.sage_main(params)
    train_time = time.time()-start_time
    print("the train_time is" + str(train_time))
    G = None
    embedding = None
    weight = None
    return G, embedding, weight
     
