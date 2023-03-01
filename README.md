# Midterm Overview
This repo is my studying resource where I attempt to code everything in the lectures to understand them.
It has worked for me in the past so I'm going to try again.

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
```math
    H(x) := -\sum_{x\in X} p(x) \log(p(x))\\
    = E[\log(p(x))]
```
Where, $p(x)$ is the probability of an event $x$ happening and $E[x]$ is the expected value of x.
The entropy tells us how random a set of data is.
An $H[x] = 0$ tells us there is no randomness, i.e only one event exists in the set of points.
We can see that if there is only one event in the set $X$ that event will have a $p(x) = 1$ which will $\log(1) = 0$ thus having an entropy of 0.


## Concepts

##### Unsupervised Learning
This is the idea that a dataset does not have any labels for its training data.

#### Supervised Learning
This is the idea that a machine learning model is trained on data that has labels.
This is the primary focus of the first lecture.
We will formalize this concept below.
Let us say there is a set of all possible instances $X$, we want to find a function $f:X\rArr Y$ where $Y$ is the label space.
When I write, $f:X\rArr Y$ it means to map a set $X$ to a set $Y$.
To simplify this idea further, it means to find a function that maps one variable to the other.
The final component of the problem statement is the set of all possible mappings $H = \{h|h:X\rArr Y\}$.
Again this reads more complicated than it is, it is an abstract representation of all possible models.
The goal is to find an $h$ that best approximate $f$ using the labeled dataset.
Let us give examples of each problem component, for a **binary classification** problem, $Y=\{-1,1\}$.
$X$, on the other hand, can honestly be whatever but for example sake, we say that $X=\{1,2,3,4\}$.
$f$ will take $X$ and $Y$ and attempt to create a mapping that matches all known information.
So, $f=\{(1,1),(2,-1),(3,1),(4,-1)\}$ is a mapping. 

$X$ can also have multiple features per point rather than a single value.
In this case, $x\in X$ are **feature vectors**.
$Y$ can also have more than two possibilities, we can think of cases where $Y$ has multiple cases as $Y$ has multiple **classes**.

##### Classification Vs. Regression

Classification extracts labels from a finite set of possiblities.
Regression, on the other hand, can have an infinite number of possible outputs.
Classification's goal is to determine a point's class where as the goal of regression is to map a set of points to a continouous function.

##### Hypothesis Class

We discussed earlier that $H$ is the all possible $f$s for a given $X$ and $Y$.
However, as one may assume, that can be --and likely is-- infinite.
What people do is they pick a subclass of and attempt to work within that smaller region.
This subclass may still be infinite but due to the restriction on the type of function we are allowed within the subclass it makes finding said function easier.

Let $h_{\theta}(x) = \sum_{\theta_i \in h_{\theta}}{\theta_i x_i} $, $h_{\theta}$ is a linear subclass of the hypothesis class.
$h_{\theta}$ is simplier because we can control the amount of parameters and nothing special is applied to $x$.
$\theta_i$ is a weight and $x_i$ is a feature. 

##### Goal of Supervised Learning

Goal: **Find a model $h$ that best approximates $f$**
To find $f$ we have to work with a minimization problem:

$$
\hat{f} = \argmin_{h \in H} \frac{1}{n} \sum_{i=1}^{n}l(h(x^i),y^i)
$$

$l$ is a new symbol we have yet to define, $l$ is what we call a loss function.
Let us now describe what $\hat{f}$ really means.
We want to find $h$ such that $l$ is super small. 
The smaller that $l$ is, the better the selected model is. 
