import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:

    def __init__(self,X,y):
        #Sets classes 
        self.classes = np.unique(y)
        print(X.shape)
        #Do it all
        self.set_priors_and_likelihoods_and_fit(X,y)


    def set_priors_and_likelihoods_and_fit(self,X,y):
        #init priors
        self.priors = np.ones(len(self.classes))
        #init likelihoods
        self.likelihoods = []

        for i,c in enumerate(self.classes):
            #Gets a list of when y == c, only works for single labeled dataset
            X_c = X[y == c]
            #This sets the prior proabilities for each class in the dataset.
            #We can think about it as finding P(Y)
            self.priors[i] = X_c.shape[0]/X.shape[0]
            means = []
            vars = []
            for j in range(X.shape[1]):
                #Our likelihood parameters are the mean and std of the distribution of each feature in each class.
                means.append( X_c[:,j].mean())
                vars.append(X_c[:,j].var())
            
            self.likelihoods.append((np.array(means),np.array(vars)))
    
    def predict(self,X):
        preds = []
        for p in X:
            posteriors = []
            #Make column vector
            # p = p.reshape(len(p),-1)
            for i,c in enumerate(self.classes):
                # grab prior for ith class
                prior = np.log(self.priors[i])
                # Calculate the posterior
                posterior = np.sum(np.log(self.calculate_likelihood(p, i))) + prior
                # Save Result
                posteriors.append(posterior)
            #Save the class with the highest probability per point.
            preds.append(self.classes[np.argmax(posteriors)])
        
        #Return the class with the highest probability.
        return preds


    def calculate_likelihood(self, X, idx):
  
        
        #Get mean and std of class from earlier
        mean, std = self.likelihoods[idx]
        #Threshold for the std
        a = np.exp( - ((X - mean)**2 /(4*std) ))
        b = (1 / (np.sqrt(2 * np.pi * std)))

        return a*b

if __name__ == "__main__":
    
    X,y = make_classification(1000,n_classes=5,n_informative=5,n_features=5,n_redundant=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    nb_model = NaiveBayes(X_train,y_train)
    
    model = GaussianNB()
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)

    error = 0
    error2 = 0

    ps = nb_model.predict(X_test)
    for i in range(X_test.shape[0]):
        tmp = ps[i]
        tmp2 = predicted[i]
        if tmp != y_test[i]:
            error+=1
        if tmp2 != y_test[i]:
            error2 +=1

    error = error/X_test.shape[0]
    error2 = error2/X_test.shape[0]
    print(f"My Model Accuracy = {1-error}\tSklearn's Model Accuracy = {1-error2}")

