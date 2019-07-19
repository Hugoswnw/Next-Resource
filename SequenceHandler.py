import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def genXY(flat, xy_index, transforming=lambda x: x):
    x, y = [], []
    for k, v in xy_index.items():
        x.append(transforming([flat[i] for i in v]))
        y.append(flat[k])
    return np.array(x), np.array(y)

def flatten(dataset, seq_len=1):
    xy_index = {}
    flat = []
    for _i, _seq in enumerate(dataset):
        for _j, _chap in enumerate(_seq):
            if(_j!=0):
                xy_index[len(flat)] = range(len(flat)-(min(seq_len, _j) if seq_len>0 else _j),len(flat))
            flat.append(_chap)
    return flat, xy_index

def ranks(flat, xy_index, transforming=lambda x: x[len(x)-1]):
    neigh = NearestNeighbors(len(flat)-1, metric="cosine")
    neigh.fit(flat)
    kneigh = neigh.kneighbors([transforming([flat[i] for i in x]) for x in xy_index.values()], return_distance=False)
    return pd.Series([ list(kneigh[i]).index(k) for i,k in enumerate(xy_index.keys())])

def evaluate(ranks, measures={"top1": (lambda x : int(x<1), "sum"), "top10": (lambda x : int(x<10), "sum"), "mean": (lambda x : x+1, "mean"), "count": (lambda x : x, "count")}):
    for k,v in measures.items():
        print("{}: {}".format(k, ranks.apply(v[0]).agg(v[1])))
        
def splitTrainTest(xy_index, sep=0.7):
    pos = np.random.permutation(list(xy_index.keys()))
    s = int(sep*len(pos))
    return getSubDict(xy_index, pos[:s]),getSubDict(xy_index, pos[s:])
    
def getSubDict (data, keys):
    return {x: data[x] for x in keys}

def gen_sequences(x, y):
    while True:
        for _i, _x in enumerate(x):
            yield np.array([x[_i]]), np.array([y[_i]])

def dist(a, b, distance):
    return pd.Series([distance(a[_i], b[_i]) for _i, _ in enumerate(a)]).mean()