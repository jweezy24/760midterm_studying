import numpy as np
from scipy.spatial.distance import euclidean
from sklearn import preprocessing


def sigma(w,x,l=None,return_probs=False):
    w = w.astype(float)
    x = x.astype(float)

    p = (w@x.T)
    res = (1/(1+ np.exp(-1*p)))
    if not return_probs:
        if len(x.shape)>1:
            for i in range(len(res)):
                pred = res[i]
                if pred >= 0.5:
                    res[i] = 1
                else:
                    res[i] = 0
        else:
            if res >= 0.5:
                return 1
            else:
                return 0
    else:
        return res
    return res

def step(w,x,y,lr):
    
    w = w -  lr*(x.T@(sigma(w,x,l=y)-y))
    return w

def logistic_regression_predict(x,w):
    validate = w@x.T
    p = [0 for i in range(5000)]
    c=0
    for pred in validate:
        if pred >= 0:
            p[c] = 1
        else:
            p[c]=0
    # print(p)
    return np.array(p)

def train_lr(training):
    w = np.ones(3000)
    w_old = np.zeros(3000)
    training,y = training
    training = preprocessing.normalize(training)
    lr = 0.01#2/5000
    max_iters = 10**8
    c = 0
    min_w = []
    min_v = 100
    hit_max = False
    while euclidean(w_old,w) > 0.001:

        w_old = w
        w = step(w,training,y,lr)
        preds = sigma(w,training)
        loss = sum(abs(y.flatten()-preds))/5000
        if loss < min_v:
            min_w = w
            min_v = loss 
        if c > max_iters:
            hit_max = True
            break
        print(f"Loss = {loss}, C = {c}")
        c+=1
    if hit_max:
        return min_w
    else:
        return w
