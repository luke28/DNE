from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import collections

import networkx as nx
from networkx.readwrite import json_graph
from utils.env import *
version_info = map(int, nx.__version__.split('.'))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

#Set random seed
seed=123
np.random.seed(seed)

WALK_LEN=5
N_WALKS=50
#N_WALKS=5
#WALK_LEN=2

def load_data(prefix, normalize=True, load_walks=False):
    print("in load data")
    print(prefix+"-G.json")
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    print("in load data func")
    print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
    print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    print("in load data func")
    print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
    print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

def generate_traindata_for_SAGE(nwFile, flagFile, ratio_train_val, ratio_train, feature_size, dataname, self_loop='yes'):
    id_max = 0
    id_train_val = 0
    id_train = 0

    # read flag
    node_flag = {}
    flag_set = []
    flag_no = 0
    with open(flagFile, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            line = line.strip()
            items = line.split()
            if len(items) < 2:
                continue
            items = [int(it) for it in items]
            node_flag[items[0]] = items[1:]
            for it in node_flag[items[0]]:
                if it not in flag_set:
                    flag_set.append(it)
                    flag_no += 1

    G = nx.Graph(name='disjoint_union(, )')
    class_map = {}
    id_map_unorder = {}
    feats=[]
    with open(nwFile, 'r') as infile:
        for line in infile:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) == 1:
                id_train_val = int(int(items[0])*ratio_train_val)
                id_train = int(id_train_val*ratio_train)
                print("id_train_val:"+str(id_train_val))
                print("id_train:" + str(id_train))
                continue
            if len(items) != 2:
                continue

            id_max = max(int(items[0]), int(items[1]), id_max)
            for it in items:
                n = int(it)
                if n not in G:
                    fea_vec = get_random_features(feature_size)
                    label_vec = [0 for i in range(flag_no)]
                    for f in node_flag[n]:
                        label_vec[f]=1
                    class_map[str(n)] = label_vec
                    id_map_unorder[str(n)] = fea_vec
                    if n <= id_train:
                        G.add_node(n, test=False, feature=fea_vec, val=False, label=label_vec)
                    elif n <= id_train_val:
                        G.add_node(n, test=False, feature=fea_vec, val=True, label=label_vec)
                    else:
                        G.add_node(n, test=True, feature=fea_vec, val=False, label=label_vec)
            G.add_edge(int(items[0]), int(items[1]), test_removed=False, train_removed=False)

    id_map_lst = sorted(id_map_unorder.iteritems(), key=lambda d:int(d[0]))
    id_map = collections.OrderedDict()
    # generate id_map, feats
    for it in id_map_lst:
        # print(it[0])
        feats.append(it[1])
        id_map[str(it[0])] = it[0]

    if self_loop=='yes':
        for i in range(id_max+1):
            G.add_edge(i, i, test_removed=False, train_removed=False)

    for edge in G.edges():
        if G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']:
            G[edge[0]][edge[1]]['train_removed']=True
        if G.node[edge[0]]['test'] or G.node[edge[1]]['test']:
            G[edge[0]][edge[1]]['test_removed']=True

    #random walks
    nodes = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    G_part = G.subgraph(nodes)
    walks = run_random_walks(G_part, nodes)
    
    print("gen intput after random walks")
    print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
    print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
    #save file
    dataname = dataname+"_"+str(id_train_val)+"_"+str(ratio_train_val)+"_nw"
    Gpath = DATA_PATH+'/'+str(dataname)+'-G.json'
    id_map_path=DATA_PATH+'/'+str(dataname)+'-id_map.json'
    class_map_path=DATA_PATH+'/'+str(dataname)+'-class_map.json'
    feats_path=DATA_PATH+'/'+str(dataname)+'-feats.npy'
    walks_path=DATA_PATH+'/'+str(dataname)+'-walks.txt'

    with open(Gpath, 'w') as outfile:
        outfile.write(json.dumps(json_graph.node_link_data(G)))

    with open(id_map_path, 'w') as outfile:
        outfile.write(json.dumps(id_map))

    with open(class_map_path, 'w') as outfile:
        outfile.write(json.dumps(class_map))

    np.save(feats_path, feats)

    with open(walks_path, 'w') as outfile:
        outfile.write("\n".join([str(p[0]) + "\t" +str(p[1]) for p in walks]))

# initialize the G
def init_G(initFile, flagFile, id_map, feats):
    # read node flag
    node_flag = {}
    flag_set = []
    flag_no = 0
    with open(flagFile, 'r') as infile:
        for line in infile:
            line = line.strip()
            items = line.split()
            if len(items) < 2:
                continue
            items = [int(it) for it in items]
            node_flag[items[0]] = items[1:]
            for it in node_flag[items[0]]:
                if it not in flag_set:
                    flag_set.append(it)
                    flag_no += 1
    
    edges_init = []
    with open(initFile, 'r') as infile:
        for line in infile:
            line = line.strip()
            items = line.split()
            if len(items) < 2:
                continue
            items = [int(it) for it in items]
            edges_init.append([items[0], items[1]])

    class_map = {}

    # generate networkx
    G = nx.Graph(name="disjoint_union(, )")
    
    update_G(G, feats, id_map, class_map, edges_init, node_flag, flag_no,  False)

    # random walks
    nodes = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    G_part = G.subgraph(nodes)
    walks = run_random_walks(G_part, nodes)

    for edge in G.edges():
        if G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']:
            G[edge[0]][edge[1]]['train_removed']=True
        if G.node[edge[0]]['test'] or G.node[edge[1]]['test']:
            G[edge[0]][edge[1]]['test_removed']=True

    return G, walks, class_map, node_flag, flag_no

def change_G_status(G):
    for n in G.nodes():
        if G.node[n]['val']:
            G.node[n]['val'] = False
        if G.node[n]['test']:
            G.node[n]['test'] = False
    for e in G.edges():
        if G[e[0]][e[1]]['test_removed']:
            G[e[0]][e[1]]['test_removed'] = False
        if G[e[0]][e[1]]['train_removed']:
            G[e[0]][e[1]]['train_removed'] = False

def get_random_features(fea_size):
    fea = [random.random() for i in range(int(fea_size))]
    return fea

# edges_added: (f1, node_id)...(fm, node_id)
def update_G(G, feats, id_map, class_map, edges_added, node_flag, flag_no, isTest=True):
    if not isTest:
        _test=False
        _val=False
        _test_removed=False
        _train_removed=False
    else:
        _test=True
        _val=False
        _test_removed=True
        _train_removed=True
    for e in edges_added:
        fn = e[0]
        tn = e[1]
        if fn not in G:
            fea_vec = feats[fn]
            label_vec = [0 for i in range(flag_no)]
            for f in node_flag[fn]:
                label_vec[f]=1
            G.add_node(fn, test=_test, feature=fea_vec, val=_val, label=label_vec)
            class_map[str(fn)]=label_vec
        if tn not in G:
            fea_vec = feats[tn]
            label_vec = [0 for i in range(flag_no)]
            for f in node_flag[tn]:
                label_vec[f]=1
            G.add_node(tn, test=_test, feature=fea_vec, val=_val, label=label_vec)
            class_map[str(tn)]=label_vec

        G.add_edge(fn, tn, test_removed=_test_removed, train_removed=_train_removed)

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

def test_randomwalk():
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

def test_generate_input():
    nwFile="/home/wangyun/repos/DNE/data/blog_nw.dat"
    flagFile="/home/wangyun/repos/DNE/data/blog_flag.dat"
    ratio=0.7
    feature_size=10
    dataname="blog"
    outputPath="/home/wangyun/repos/GraphSAGE/input_data"
    generate_traindata_for_SAGE(nwFile, flagFile, ratio, ratio, feature_size, dataname, outputPath)

def test():
    init_file="/home/wangyun/repos/GraphSAGE/input_data/dolphins_40_nw_init"
    flag_file="/home/wangyun/repos/GraphSAGE/input_data/dolphins_40_nw_flag"
    feature_size=10
    [G, feats, id_map, walks, class_map, node_flag, flag_no] = init_G(init_file, flag_file, feature_size)
    if False:
        with open("/home/wangyun/repos/GraphSAGE/output_data/G.json", 'w') as outfile:
            outfile.write(json.dumps(json_graph.node_link_data(G)))
        with open("/home/wangyun/repos/GraphSAGE/output_data/id_map.json", 'w') as ofile:
            ofile.write(json.dumps(id_map))

        with open("/home/wangyun/repos/GraphSAGE/output_data/class_map.json", "w") as ofile:
            ofile.write(json.dumps(class_map))

        with open("/home/wangyun/repos/GraphSAGE/output_data/walks.txt", "w") as ofile:
            ofile.write("\n".join([str(p[0])+"\t"+str(p[1]) for p in walks]))

if __name__=='__main__':
    test_generate_input()
    #test()
