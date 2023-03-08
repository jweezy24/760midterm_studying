import numpy as np

#Simple implementation of sigmoid
def sigmoid(x,w):
    return (1/(1+np.exp(-w.T@x)))

#Derivative of sigmoid
def sigmoid_p(x,w):
    return sigmoid(x, w)*(1-sigmoid(x, w))

