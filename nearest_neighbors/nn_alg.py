import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neighbors import KNeighborsClassifier

def eval_neighbors_d2z(p,data,labels=None,k=1,use_email_dataset=True,p_is_matrix=True):
    distances = []
    import time
    if use_email_dataset:
        assert(type(labels) !=  type(None))

        p = p.astype(float)
        data = data.astype(float)
        start = time.time()
        if not p_is_matrix:
            p = p.reshape((1,len(p)))
            p = p.astype(float)
            for i in range(len(data)):
                p2 = data[i,:].astype(float)
                z = labels[i]
                # d = euclidean(p,p2)
                p2 = p2.reshape((1,len(p2)))
                
                
                d = cdist(p,p2,metric="euclidean")
                d = d.flatten()[0]
                distances.append((d,z))
            distances.sort(key=lambda x: x[0])
        else:
            d = cdist(p,data,metric="euclidean")
            ret = []
            labels = labels.astype(int)
            confidence = []
            
            for d_to_p in d:
                ls = labels[np.argsort(d_to_p)][:k]
                ls2 = labels[np.argsort(d_to_p)][:k]
                ls = np.bincount(ls)
                b = np.argmax(ls)
                tracker = [0,0]
                for i in ls2:
                    tracker[i]+=1
                conf = tracker[b]/sum(tracker)
                # print(b,conf)
                ret.append(b)
                confidence.append(1-conf)
            end = time.time()
            print(f"Done with distances. Time:{end-start} seconds")
            return ret,confidence                 
        
    else:
        for x,y,z in data:
            p2 = (x,y)
            d = euclidean(p,p2)
            distances.append((d,z))
        distances.sort(key=lambda x: x[0])

    
    if k>1:
        nearest = distances[:k]
        assert(len(nearest) == k)
        table = {}
        for d,l in nearest:
            if l not in table.keys():
                table[l] =1
            else:
                table[l]+=1
        its = list(table.items())
        its.sort(key=lambda x: x[1])
        return its[0][0]
    else:
        
        return distances[0][1]        
