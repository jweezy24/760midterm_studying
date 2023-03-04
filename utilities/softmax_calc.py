import numpy as np

def softmax(x):
    exp_s = np.exp(x)
    exp_sum = np.sum(exp_s)

    soft_max_probs = exp_s/exp_sum
    return soft_max_probs


print(softmax([1,2,3,4,5]))