import numpy as np
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from cross_validation import *
import matplotlib.pyplot as plt

#loads the digits dataset
X, y = load_digits(return_X_y=True)
#Initalize a naive bayes classifier
naive_bayes = GaussianNB()
#Initalizes a multi-class svm model with a rbf kernal
svc = SVC(kernel="rbf", gamma=0.001)

#Setting up the x-axis
sample_sizes = [i for i in range(50,len(X),50)]

#Determin how many splits for CV
n=10
y_axis_nb = []
y_axis_svc = []
nb_std = []
svc_std = []
for sample_size in sample_sizes:
    #Split data for CV
    CV = CV_splits(X,y,n,sample_size)
    nb_ave_error = 0
    svc_ave_error = 0
    t_scores_nb = []
    t_scores_svc = []
    #Preform CV
    for xtr,ytr,xte,yte in CV:
        #Train Models
        naive_bayes.fit(xtr,ytr)
        svc.fit(xtr,ytr)

        #Test Models
        nb_preds = naive_bayes.predict(xte)
        svc_preds = svc.predict(xte)
        
        #Test Scores Estimations
        error_nb = 0
        error_svc = 0
        for i in range(len(nb_preds)):
            p = nb_preds[i]
            p2 = svc_preds[i]
            if p != yte[i]:
                error_nb+=1
            if p2 != yte[i]:
                error_svc+=1

        
        error_nb = error_nb/len(nb_preds)
        error_svc = error_svc/len(svc_preds)
        t_scores_nb.append(error_nb)
        t_scores_svc.append(error_svc)
        nb_ave_error+=error_nb
        svc_ave_error += error_svc
    
    nb_ave_error/=n
    svc_ave_error/=n
    y_axis_nb.append(1-nb_ave_error)
    y_axis_svc.append(1-svc_ave_error)
    nb_std.append(np.std(t_scores_nb))
    svc_std.append(np.std(t_scores_svc))

nb_mean = np.mean(y_axis_nb)
svc_mean = np.mean(y_axis_svc)


plt.errorbar(sample_sizes,y_axis_nb,yerr=nb_std,label="Naive Bayes")
plt.errorbar(sample_sizes,y_axis_svc,yerr=svc_std,label="SVC")
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.title("Naive Bays vs SVC Learning Curves")
plt.legend()
plt.show()