import numpy as np
from relu import *
from sigmoid import *
from tanh import *

class HiddenLayer:
    def __init__(self,alg,nodes):
        if alg.lower() == "relu":
            self.activation_function = relu
            self.derivative_func = relu_p
        elif alg.lower() == "sigmoid":
            self.activation_function = sigmoid
            self.derivative_func = sigmoid_p
        elif alg.lower() == "tanh":
            self.activation_function = tanh
            self.derivative_func = tanh_p
        else:
            raise("Activation function not supported.")
    
    def forward_pass(self,x,w):
        res = self.activation_function(x,w)
        return res
    
    def backward_pass(self,x):
        I = np.identity(len(x))
        res = self.derivative_func(x,I)
        return res


class NeuralNet:
    def __init__(self,layers,algs,node_sizes,input_size):
        #Initialize weights 
        w = np.zeros((input_size,node_sizes[0]))
        w = w+0.5
        self.ws = [w]
        #simple layers check
        self.layers = []
        if layers != len(algs):
            raise("Need to assign an activation function per layer")

        #Create layers  
        for i in range(layers-1):
            self.layers.append(HiddenLayer(algs[i],node_sizes[i]))
            w_new = np.zeros((node_sizes[i],node_sizes[i+1])) + 0.5
            self.ws.append(w_new)
        
        self.layers.append(HiddenLayer(algs[-1],node_sizes[-1]))
        self.ws.append(np.zeros((node_sizes[-2],node_sizes[-1])) + 0.5)
    
    def loss(self,y_hat,y):
        return y_hat-y        

    def train(self,X,y,lr=0.01):
        y = y.reshape((1,670))
        while True:
            c = 0
            losses = []
            caps = []
            print("FORWARD")
            #Forward pass
            res_results = []
            for layer in self.layers:
                if c == 0:
                    print(X.shape)
                    print(self.ws[c].shape)
                    res = layer.forward_pass(X.T, self.ws[c])
                    print(f"RESULTS SHAPE:{res.shape}")
                    res_results.append(res)
                else:
                    print(c)
                    print(res_results[c-1].shape)
                    print(self.ws[c].shape)
                    res = layer.forward_pass(res_results[c-1], self.ws[c])
                    print(f"RESULTS SHAPE:{res.shape}")
                    res_results.append(res)
                
                c+=1
            
            print("BACKPASS")
            deltas = []
            for i in range(len(res_results)-1,-1,-1):
                out = res_results[i]
                err = self.loss(out,y)
                layer = self.layers[i]
                tmp = layer.backward_pass(out)
                print(err.shape,tmp.shape)
                delta = err@tmp.T
                deltas.append(delta)

            
            print("UPDATING WEIGHTS")
            deltas.reverse()
            d = deltas
            print(res_results[-1])
            #update weights step
            ave_distance = 0
            for i in range(1,len(d)):
                # print(len(self.ws),len(res_results))
                diff = self.loss(res_results[i-1],y)
                print(diff)
                
                print(diff.shape,res_results[i].shape,d[i].shape)
                e = lr * diff@(res_results[i].T@d[i])
                self.ws[i] -= e
            

    def predict(self,X):
        c = 0
        res_results = []
        for layer in self.layers:
            if c == 0:
                print(X.shape)
                print(self.ws[c].shape)
                res = layer.forward_pass(X.T, self.ws[c])
                print(f"RESULTS SHAPE:{res.shape}")
                res_results.append(res)
            else:
                print(c)
                print(res_results[c-1].shape)
                print(self.ws[c].shape)
                res = layer.forward_pass(res_results[c-1], self.ws[c])
                print(f"RESULTS SHAPE:{res.shape}")
                res_results.append(res)
        
        return res_results[-1]

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
    # same as the previous section
    n_samples=1000, n_features=5, n_informative=3, n_classes=2, 
    # flip_y - high value to add more noise
    flip_y=0.1, 
    # class_sep - low value to reduce space between classes
    class_sep=0.5
    )

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

    nn = NeuralNet(3, ["sigmoid","relu","tanh"], [4,3,2], 5)
    
    nn.train(X_train, y_train)


