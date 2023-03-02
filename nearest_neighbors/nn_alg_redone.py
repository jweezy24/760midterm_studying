import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

def get_datasets(n,features,div=8):
    X,y = make_blobs(n,features)
    split = int(n*((n*(div-1))/(n*div)))
    X_t = X[0:split,:]
    y_t = y[0:split]
    X_te = X[split:,:]
    y_te = y[split:]
    return X_t,y_t,X_te,y_te

def knn_alg_supervised(training,testing,training_labels,k=1,variation="",weights=[]):
    #Calculate Distances
    d = cdist(testing,training,metric="euclidean")
    #Intializes returns
    ret = []
    confidences = []

    #Checks variation
    if variation == "normal":
        #Each row represents the set of all distances in the training set to a single point
        for distance_to_point in d:
            #Sorts the results by then gets the ranked labels
            ls = training_labels[np.argsort(distance_to_point)][:k]
            #Bins each result the index is the value and the value at the index is the number of occurances
            ls = np.bincount(ls)
            #Grabs the predicted label
            b = np.argmax(ls)

            #Computes the confidence value
            tracker = [0 for i in range(len(set(training_labels)))]
            for i in ls:
                tracker[i] += 1
            confidence = tracker[b]/sum(tracker) 
            
            #Appended predicted label with its accociated confidence value
            ret.append(b)
            confidences.append(confidence)
        return ret,confidence

if __name__ == "__main__":
    variation = "normal"
    n = 1000
    features = 5
    training,training_labels,testing,testing_labels = get_datasets(n, features)
    knn_alg_supervised(training, testing, training_labels,variation=variation)


