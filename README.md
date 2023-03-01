# Midterm Overview
This repo is my studying resource where I attempt to code everything in the lectures to understand them.
It has worked from me in the past so I'm going to try again.

## ML Algorithms

This section will go over all of the ml algorithms coded from scratch.

### Decision Trees

Located in the `/dtrees` folder, is the algorithm for decision trees.
![dtrees](figures/Dtrees.png)
The above visualization demonstrates how the algorithm works at a high level.
Each node that is not a leaf node has a **split condition**.
These split conditions split the data set on a single feature based on a single bound.
These splits are deteremined using the **information gain ratio**(IGR).
To define IGR we first need to define **entropy** which is,
\[
    H(x) := -\sum_{x\in X} p(x) \log(p(x))\\
    = E[\log(p(x))]
\]
Where, $p(x)$ is the probability of an event $x$ happening and $E[x]$ is the expected value of x.
The entropy tells us how random a set of data is.
An $H[x] = 0$ tells us there is no randomness, i.e only one event exists in the set of points.
We can see that if there is only one event in the set $X$ that event will have a $p(x) = 1$ which will $\log(1) = 0$ thus having an entropy of 0.
