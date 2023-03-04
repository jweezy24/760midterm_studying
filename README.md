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
    H(x) := -\sum_{x\in X} p(x) \log_2(p(x))\\
    = E[\log_2(p(x))]
```
Where, $p(x)$ is the probability of an event $x$ happening and $E[x]$ is the expected value of x.
The entropy tells us how random a set of data is.
An $H[x] = 0$ tells us there is no randomness, i.e only one event exists in the set of points.
We can see that if there is only one event in the set $X$ that event will have a $p(x) = 1$ which will $\log(1) = 0$ thus having an entropy of 0.
The information gain(IG) can now be defined below.
```math
    IG(Y,X) = H(Y) - H(Y|X)
```
So what this is saying intuitively is IG is the difference between the entropy of the even occuring over the data set and the entropy of the event happening when another event occurs.
Lets work through this example below,
![dtree_alg](figures/entropy_example.png)
First we calculate, 
```math
    H(Y) = -\sum_{x\in Y} p(x) \log(p(x))\\
    = (0.5(\log(0.5))) + (0.25(\log(0.25))) + (0.125(\log(0.125))) + (0.125(\log(0.125))) \\
    = 1.75
```
Next we calcuate,
```math
    H(Y|X) = -\sum_{x\in X}\sum_{y\in Y} p(x,y) \log(p(y|x))\\
    = 1.5
```
I wrote code in `utilities/entropy` calculating the results.
I just want to make a few notes, the notation does not account for conditional probability situations.
This again, is very annoying because the **conditional entropy is defined differently than regular entropy.**
As we can see from the equations above.
To get the result you have to apply Bayes rule to redefine the conditional within the log to be a product of the joint probabilities divided by the probability of $x$.

We can now finally define the IGR we discussed erlier.
```math
    IGR(X,Y) = \frac{IG(X,Y)}{H(Y)} 
```
Finally we have a formal definition of IGR. 

![dtree_alg](figures/dtree_algorithm.png)
Now that we have IGR we can walk through the decision tree algorithm.
The canidate splits are determined by the highest IGR when selecting a point to be conditioned over.
The stopping criteria will be when the IGR is zero or when the set of training instances is empty.
A leaf node will delegate a label once a testing sample is ran through the dtree.
Finding the best split is just using the maximal IGR we determined from when making $C$.
Once $C$ has been determined, we branch off into subsets.
If there are only two classes, then there will only be two subsets.
Each subset will get a node and the algorithm will be called recursively.
For an implementation example, see `dtrees/decision_trees.py`.


### k-Nearest Neighbors (kNN)

One of the simpler algorithms, the nearest neighbor algorithm will classify the points based on the point it is closest to in the dataset.

Below depicts a $k=1$ nearest neighbor algorithm visualized.
![vd](figures/voronoi_diagram.png)
As we can see, the regions around each determine which points will share that label based on the other point's location.
A little confusing but the idea here is that **a point is classified based on the closest point(when $k=1$).**

When $k>1$ we actually determine class based on the k nearest points.
For example, if $k=2$ we look at the 2 closest points and determin the label by the class that occurs the most frequently.
As you may have infered, ties can occur.
To solve this problem use an odd number for $k$ this solution will only work if there are only two classes.
If there are more than two classes than a tie can still occur.
Usually, if there is a tie, default to the closest point.
If there is still a tie, randomly pick one of the points to decide a label.
A formal description for what I am talking about is below:
```math
\hat{y} = \argmin_{y\in Y}\sum_{i=1}^{k}1(y = y^i)
```
Excuse the lack of an indicator function, I do not know how to write it in markup.
However, the equation above formalizes the idea of prediction.

Standardization is another topic that can be applied to multiple algorithms here but the lecture slides introduce the concept when discussing kNN.
To discuss the process, we first need to formalize the standard deviation and mean.
**Mean**
```math
    \mu_a = \frac{1}{n}\sum_{i=1}^{n}x_a^i
```
**Standard Deviation**
```math
    \sigma_a = (\frac{1}{n}\sum_{i=1}^{n}(x_a^i - \mu_a)^2)^{\frac{1}{2}}
```
Above is explicitly not in the notes, however I do believe the $\mu_a$ is supposed to be $\mu_a$.
In the notes the original definition defines it as $\mu_i$ which doesn't make any sense.
That would imply that there are multiple means, or a single mean per point.

The standardization formalization can now be written as.
$$
\hat{x}_a^j = \frac{x_a^j - \mu_a}{\sigma_a} 
$$
I implemented standardization in the file `utilities/standardization.py`.
I show that the sklearn standardization is gets the same result as my simple implementation.

I implemented a basic kNN algorithm in the file `nearest_neighbors/nn_alg.py`.
I have not implemented fancy nearest neighbors yet.
I think I eventually will.

There still remains the question, **How do you deal with irrelevant features?**
![irrelevant_features](figures/irrelevant_features.png)
Above is an example of how irrelevant features can affect the distances.
This is just a problem with kNN models in general.
They are weak to irrelevant features.

**Strengths**
1. Easy to explain predictions
2. Simple to implement
3. No training
4. Often a good solution in practice

**Weaknesses**
1. Sensitive to irrelevant features (see image above)
2. Prediction stage can be expensive
3. No "model" to interperet 

The **Inductive Bias** is defined as assumptions a learner uses to predict a label $y$ for a previosuly unseen instance $x_i$.
There are two components that make up inductive bias: **hypothesis space bias** and **preference bias**.
**Preference Bias** specifies a preference ordering within the space of models.
**Hypothesis Space Bias** determins the models that can be represented.
These definitions at face value are not good.
To simplify these abstract definitions, inductive bias is a set of assumptions that a machine learning algorithm makes about the relationship between the input and output data based on the training data.
Our preference bias is then defined by the neighborhoods seen in training data.
ChatGPT gave me a great example to explain this intuitively.
Imagine training a classifier based on movie reccomendations, if I prefer action movies, my training data will have a preference bias towards action movies.
Therefore, someone who does not enjoy action movies will have a less accurate rating prediction than me.
Hypothesis classes are actually a little easier to grasp.
Again, with the help of ChatGPT, a hypothesis class is more about boundry creation.
If a boundry is explicitly linear, then the kNN model will have trouble because kNN doesn't draw linear boundries well.
Therefore, kNN models have a hypothesis bias.

## Concepts

#### Unsupervised Learning
This is the idea that a dataset does not have any labels for its training data.
There are less formalities with this type of machine learning.
I don't know if that is applied through the entire study but this lecutre does not take equal amount of time to define this type of learning.

##### Setup

Given:
$$
\{x^1,x^2,...,x^n\}
$$
The **goal** is to discover interesting regularities/structures/patterns that characterize the instances.
For example, clustering, anomoly detection, and dimensionality reduction all fall under unsupervised learning.
To paraphrase a bit, unsupervised learning is designed for pattern identification with a lack of labels.

##### Clustering

The goal for clustering is to model *h* such that it divides training set into clusters with intra-cluster similarity amd inter-cluster dissimilarity.
To paraphrase again, the goal is to group *close* points and separate *far* points.
You may be asking yourself, how do you define distance for points where there is no clear definition of distnace?
Well, there are many ways, one can project points with no sense of distances into a space where distance is defined.
Or as my professor put it, just don't use the clusering algorithms.

![cluster](figures/clustering_example.png)

Above is a visualization of the concepts discussed. 


##### Anomaly Detection

The goal of anomaly detection is to detect anomalies.
Yes I know riviting.
But this problem is actually harder than it sounds.
Although simple to define what the goal is, defining what an anomoly is depends completely on the space you are working in.
An anomaly for a signal processing problem will look completely different than a computer vision anomaly.
Defining the anomaly to a computer is the most difficult component.

##### Dimensionality Reduction

The goal for dimensionality reduction is to find patterns in lower dimensional feature vectors.
In other words to detect correlations between lower dimensions within the dataset.
This concept again is harder than it sounds.
A easy example for those who know about it is **PCA** or Principal Component Analysis.
PCA defines what deminsions have the most information about the data.
I do not want to get into the definition here as this is a high level overview but the point is that it accomplishes the goal of dimensionality reduction which is find lower dimensional patterns within a dataset.



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

##### Types of Training

There are two types of learning defined in the lectures.
1. Batch Learning
2. Online Learning

**Batch Learning** trains over all instances at once.
Whereas, **Online Learning** trains the model on a specific group then updates are gotten per group.

The general idea behind these two concepts says that both methods for training work.
I am a bit confused as to why the separation is important but I digress.

##### Goal of Supervised Learning

Goal: **Find a model $h$ that best approximates $f$**
To find $f$ we have to work with a minimization problem:

```math
\hat{f} = \argmin_{h \in H} \frac{1}{n} \sum_{i=1}^{n}l(h(x^i),y^i)
```

$\hat{f}$ is known as **Empirical risk minimization**(ERM).

$l$ is a new symbol we have yet to define, $l$ is what we call a loss function.
Let us now describe what $\hat{f}$ really means.
We want to find $h$ such that the sum over all points $l$ is super small. 
The smaller that sum over $l$ is, the better the selected model is.
Intuitively this makes sense, the total error being low does imply that the model is the best.




