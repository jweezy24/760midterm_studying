import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def mean(X):
    return sum(X)/len(X)

def standard_dev(X,mean):
    before_sum = (X-mean)**2
    before_div = sum(before_sum)
    before_root = before_div/len(X)
    return np.sqrt(before_root)

def standarize(X):
    mu = mean(X)
    std = standard_dev(X, mu)
    return (X - mu)/std

if __name__ == "__main__":
    X,y = make_blobs(100)
    print(X)
    scaler = StandardScaler().fit(X)
    X_std_sk = scaler.transform(X)
    my_X = standarize(X)
    print(X_std_sk-my_X)