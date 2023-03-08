import numpy as np

#Tanh translation
def tanh(x,w):
    return (np.exp(w.T@x) - np.exp(-w.T@x))/(np.exp(w.T@x)+np.exp(-w.T@x))

#tanh derivative
def tanh_p(x,w):
    return 1- (tanh(x, w))**2