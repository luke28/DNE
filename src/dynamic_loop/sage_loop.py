import sys
import os
import json
import numpy as np
import time

from utils.env import *

from contrast_experiment.graphsage import unsupervised_train as ut 
def loop(params, G, embeddings, weights, metric, output_path):
    params['output_path'] = output_path
    start_time = time.time()
    ut.sage_main(params, metric)
    test_time = time.time()-start_time
    print("the test time is:" + str(test_time))

