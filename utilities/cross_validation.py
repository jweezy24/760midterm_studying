import numpy as np
from sklearn.model_selection import train_test_split

## X is the dataset
## y is the labels of the datset
## n is the number of CV splits
def CV_splits(X,y,n,training_size):
    all_data = []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X)-training_size, random_state=42+i)
        all_data.append((X_train,y_train,X_test,y_test))
    return all_data
