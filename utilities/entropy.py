import numpy as np

def entropy(X):
    tracker = np.array([0 for i in range(max(set(X)))])
    for e in X:
        tracker[e-1]+=1
    probabilites = tracker/len(X)
    entropy = 0
    for p in probabilites:
        if p == 0:
            continue
        
        entropy += p*np.log2(p)
    return -entropy

def entropy_conditional(Y,X):
    joint = np.zeros((len(set(Y)),len(set(X))),dtype=float)
    tracker_X = np.array([0 for i in range(max(set(X)))])
    tracker_Y = np.array([0 for i in range(max(set(Y)))])
    for e in range(len(X)):
        x = X[e]
        y = Y[e]
        joint[y-1][x-1]+=1
        tracker_X[x-1]+=1
        tracker_Y[y-1]+=1

    events = len(set(Y))*len(set(X))
    joint_probabilites = joint/events
    X_probabilities = tracker_X/len(Y)
    Y_probabilities = tracker_Y/len(Y)
    entropy = 0

    for i in range(len(X_probabilities)):
        px = X_probabilities[i]
        for j in range(len(Y_probabilities)):
            py = Y_probabilities[j]
            joint_p = joint_probabilites[j,i]
            if px == 0 or joint_p==0:
                continue
            
            entropy += joint_p*np.log2(joint_p/px)

    return -entropy

if __name__ == "__main__":
    #Table example from readme
    X = [1,1,1,1,2,2,3,4]
    E = entropy(X)
    print(E)
    X = [1,1,2,2,1,2,1,2]
    Y = [1,1,1,1,2,2,3,4]
    E2 = entropy_conditional(Y,X)
    print(E2)
    print(E-E2)