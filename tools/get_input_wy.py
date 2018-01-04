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
import collections


def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dir', type = str, required = True, help="the inputfile dir")
    parser.add_argument('--output_dir', type = str, required = True, help="the outputfile dir")
    parser.add_argument('--data_name', type=str, required=True, help='the name of dataset')
    parser.add_argument('--ratio', type = float, required = True, help="the ratio of train nodes")
    parser.add_argument('--self_loop', type = str, default = "yes")
    args = parser.parse_args()
    nw_file = os.path.join(args.input_dir, args.data_name + "_nw.dat")
    ratio = args.ratio
    m = 0
    n = 0

    G_init = []
    G_dynamic = {}
    with open(nw_file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if (len(items) == 1):
                n = int(int(items[0])*ratio)
                continue
            if (len(items) != 2):
                continue
            m = max(int(items[1]), int(items[0]), m)
            if int(items[1]) < n and int(items[0]) < n:
                G_init.append(items)
            else:
                it = max(int(items[0]), int(items[1]))
                if it not in G_dynamic:
                    G_dynamic[it] = [items]
                else:
                    G_dynamic[it].append(items)

    if args.self_loop == "yes":
        for i in xrange(n):
            G_init.append((str(i), str(i)))
        for i in xrange(n, m + 1):
            if i not in G_dynamic:
                G_dynamic[i] = [(str(i), str(i))]
            else:
                G_dynamic[i].append((str(i), str(i)))
    
    init_nw_file = os.path.join(args.output_dir, args.data_name + "_" + str(n) + "_nw_init")
    dynamic_nw_file = os.path.join(args.output_dir, args.data_name + "_" + str(n) + "_nw_dynamic")

    with open(init_nw_file, "w") as f:
        f.write(str(n) + "\n")
        for u, v in G_init:
            f.write(str(u) + "\t" + str(v) + "\n")

    tmp = [(k, G_dynamic[k]) for k in sorted(G_dynamic.keys())]
    with open(dynamic_nw_file, "w") as f:
        for u, s in tmp:
            f.write(str(u) + "\t" + str(len(s)) + "\n")
            for v, w in s:
                f.write(str(v) + "\t" + str(w) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
