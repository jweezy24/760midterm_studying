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
        w = np.zeros((input_size,node_sizes[0]),dtype=float)
        print(w)
        self.ws = [w+0.5]
        #simple layers check
        self.layers = []
        if layers != len(algs)+1:
            raise("Need to assign an activation function per layer")

        #Create layers  
        for i in range(len(algs)-1):
            self.layers.append(HiddenLayer(algs[i],node_sizes[i]))
            w_new = np.zeros((node_sizes[i],node_sizes[i+1]),dtype=float)+0.5
            self.ws.append(w_new)
        
        self.layers.append(HiddenLayer(algs[-1],node_sizes[-1]))
        self.ws.append(
            np.zeros((node_sizes[-1],node_sizes[-1]),dtype=float)+0.5)

        
    
    def loss(self,y_hat,y):
        y_t = y.flatten()
        y_hat = y_hat.flatten()
        misses = 0
        for i in range(y_hat.shape[-1]):
            p = y_hat[i]
            if p!=y_t[i]:
                misses+=1
        print(misses)
        cur_loss = misses/len(y_t)
        return cur_loss

    def loss_all(self,y_hat,y):
        y_t = y.flatten()
        y_hat = y_hat.flatten()
        arr = []
        for i in range(y_hat.shape[-1]):
            arr.append(y_t[i]-y_hat[i])
        return np.array(arr)        

    def train(self,X,y,lr=0.001):
        y = y.reshape((1,-1))
        y = y.astype(float)
        cur_loss = 1
        old_loss = 0
        epochs = 0
        while epochs < 1000:
            c = 0
            print("FORWARD")
            #Forward pass
            res_results = []
            inputs = []
            for layer in self.layers:
                if c == 0:
                    inputs.append(X.T)
                    res = layer.forward_pass(X.T, self.ws[c])
                    res_results.append(res)
                else:
                    inputs.append(res_results[c-1])
                    res = layer.forward_pass(res_results[c-1], self.ws[c])
                    res_results.append(res)

                c+=1
            
            print("BACKPROP")
            print(y.shape,res.shape)
            o_err = (y-res).T
            print(o_err)
            print(o_err.shape)
            o_gradient = self.layers[-1].backward_pass(res)
            o_gradient = o_gradient@o_err
            deltas = [ (o_gradient,o_err) ]
            layer_err = o_err
            for i in reversed(range(len(self.ws)-1)):
                
                res = res_results[i]
                
                hidden_gradient = self.layers[i].backward_pass(res)
                print(hidden_gradient)
                print(layer_err.shape,self.ws[i].T.shape)
                
                layer_err= (layer_err@self.ws[i].T) 
                print(layer_err)
                print(hidden_gradient.shape,layer_err.shape)
                
                h_err = hidden_gradient@layer_err
            
                deltas.append( (h_err,layer_err) )
                print(i,layer_err.shape,hidden_gradient.shape)


                    

            
            print("UPDATING WEIGHTS")
            preds = self.predict(X)
            old_loss = cur_loss
            cur_loss = self.loss(preds,y)
            print(preds.shape)
            
            print(f"Current Loss:{cur_loss}")
            
            deltas.reverse()
            d = deltas

            #update weights step
            for i in range(len(d)):
                herr,gerr = d[i]
                if i == 0:
                    ins = X
                else:
                    ins = res_results[i-1]
                    
                
                print(i)
                if i == 0:
                    print(herr.shape,ins.T.shape,gerr.shape)
                    e = lr * (herr@ (ins.T@gerr)).T
                else:
                    print(herr.shape,ins.shape,gerr.shape)
                    e = lr * (herr@ (ins@gerr)).T
                
                
                    
                self.ws[i] -= e
                print(self.ws)
            
            # exit()
            epochs+=1
            print(epochs)

    def predict(self,X):
        c = 0
        res_results = []
        for layer in self.layers:
            if c == 0:
                res = layer.forward_pass(X.T, self.ws[c])
                res_results.append(res)
            else:
                res = layer.forward_pass(res_results[c-1], self.ws[c])
                res_results.append(res)
            c+=1
            print(self.ws[c])

        res_results[-1] = res_results[-1].flatten()
        res_results[-1][res_results[-1] >= 0.5] = 1
        print(res_results[-1])
        res_results[-1][res_results[-1] < 0.5] = 0
        print(res_results[-1])


        return res_results[-1]

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    X, y = make_classification(
    # same as the previous section
    n_samples=100, n_features=10, n_informative=6, n_classes=2, 
    # flip_y - high value to add more noise
    flip_y=0.01, 
    # class_sep - low value to reduce space between classes
    class_sep=0.5
    )

    # X = preprocessing.normalize(X)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

    nn = NeuralNet(4, ["sigmoid","sigmoid","sigmoid"], [4,3,1], 10)
    
    nn.train(X_train, y_train)
    

