import numpy as np

#Simple implementation of sigmoid
def sigmoid(x,w):    
    return (1/(1+ np.exp(-1*(w.T@x))))

#Derivative of sigmoid
def sigmoid_p(x,w):
    res = sigmoid(x, w)

    for i in range(res.shape[1]):
        res[:,i] = res[:,i]*(1-res[:,i])
    return res

