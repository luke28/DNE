import os
import sys
import time
import networkx as nx
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib import colors
from matplotlib.patches import Ellipse, Circle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier


from env import *
from data_handler import DataHandler as dh
from lib_ml import MachineLearningLib as mll

class Metric(object):
    @staticmethod
    def cal_metric(TP, FP, TN, FN):
        res = {}
        res["acc"] = float(TP + TN) / float(TP + FP + FN + TN)
        res["precision"] = float(TP) / float(TP + FP) if TP + FP > 0 else 1.0
        res["recall"] = float(TP) / float(TP + FN) if TP + FN > 0 else 1.0
        try:
            res["F1"] = 1.0 / (1.0 / res["recall"] + 1.0 / res["precision"])
        except ZeroDivisionError:
            res["F1"] = 0.0
        return res

    @staticmethod
    def draw_pr(precision, recall, file_name = "pr.png"):
        index = np.array(range(len(precision)))
        width = 0.3
        tmplist1 = [(x, precision[x]) for x in precision]
        tmplist2 = [(x, recall[x]) for x in recall]
        tmplist1.sort()
        tmplist2.sort()
        X = [x[0] for x in tmplist1]
        y1 = [x[1] for x in Gtmplist1]
        y2 = [x[1] for x in tmplist2]
        plt.bar(index - width / 2, y2, width, color = "blue", label="recall")
        plt.bar(index + width / 2, y1, width, color = "red", label="precision")
        plt.grid(True, which='major')
        plt.grid(True, which='minor')
        plt.xticks(index, X, rotation = 45, size = 'small')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=5)

        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def draw_circle_2D(embeddings, drawer, draw_path, draw_cnt):
        font = {'family' : 'serif',
                'color'  : 'darkred',
                'weight' : 'normal',
                'size'   : 26}

        x = embeddings[:,0]
        y = embeddings[:,1]
        T = np.arctan2(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.scatter(embeddings[:, 0], embeddings[:, 1], c = T, alpha = 0.8, marker='o')
        delta_x = -0.01
        delta_y = 0.01
        line_id = 0
        for emb in embeddings:
            ax.text(emb[0]+delta_x, emb[1]+delta_y, line_id, ha='center', va='bottom')
            line_id += 1
        plt.axis('scaled')
        file_path = os.path.join(draw_path, drawer['func']+'_'+str(draw_cnt))
        pp = PdfPages(file_path)
        pp.savefig(fig)
        pp.close()


    @staticmethod
    def multilabel_classification_old(X, params):
        X_scaled = scale(X)
        y = getattr(dh, params['load_ground_truth_func'])(os.path.join(DATA_PATH, params["ground_truth"]))
        y = y[:len(X)]
        f1_micro = 0.0
        f1_macro = 0.0

        for _ in xrange(params['times']):
            for i in range(y.shape[1]):
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y[:, i], test_size=params['test_size'], stratify = y[:, i])
                clf = getattr(mll, params["classification"]["func"])(X_train, y_train, params["classification"])
                f1_micro += f1_score(y_test, mll.infer(clf, X_test), average='micro')
                f1_macro += f1_score(y_test, mll.infer(clf, X_test), average='macro')

        return f1_micro/(float(params['times']*y.shape[1])), f1_macro/(float(params['times']*y.shape[1]))

    @staticmethod
    def multilabel_classification(X, params):
        X_scaled = scale(X)
        y = getattr(dh, params['load_ground_truth_func'])(os.path.join(DATA_PATH, params["ground_truth"]))
        y = y[:len(X)]
        f1_micro = 0.0
        f1_macro = 0.0

        for _ in xrange(params['times']):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=params['test_size'])
            log = MultiOutputClassifier(SGDClassifier(loss='log'), n_jobs=params['n_jobs'])
            log.fit(X_train, y_train)

            for i in range(y_test.shape[1]):
                f1_micro += f1_score(y_test[:, i], log.predict(X_test)[:, i], average='micro')
                f1_macro += f1_score(y_test[:, i], log.predict(X_test)[:, i], average='macro')
        return f1_micro/(float(params['times']*y.shape[1])), f1_macro/(float(params['times']*y.shape[1]))
    
    @staticmethod
    def classification(X, params):
        X_scaled = scale(X)
        y = dh.load_ground_truth(os.path.join(DATA_PATH, params["ground_truth"]))
        y = y[:len(X)]
        acc = 0.0
        for _ in xrange(params["times"]):
             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"], stratify = y)
             clf = getattr(mll, params["classification"]["func"])(X_train, y_train, params["classification"])
             acc += mll.infer(clf, X_test, y_test)[1]
        acc /= float(params["times"])
        return acc

if __name__ == '__main__':
    X = np.random.uniform(-0.1, 0.1, 16).reshape(8, 2)
    drawer = {}
    drawer['func'] = 'abc'
    draw_cnt = 1
    Metric.draw_circle_2D(X, drawer, '', 1)
