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

For the final note on dtrees, let us describe the inductive bias.
The **hypothesis space bias** is the decision tree algorithms favor trees with single features, axis parallel splits.
What this means is that the algorithm will tend toward designing models that are focused in a single dimension and place splits in parallel.
We can think about it as Dtrees are tend towards dividing up spaces with straight lines through where the lines are defined only in a single dimension.
The **preference bias** is indicitive of small trees identified by greedy search.
This means that the dtree algorithm will tend to try and make smaller trees with the greedy search algorithm.


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


### Linear Regression

Linear regression is the concept of finding a function $f$ that maps to a set of points and labels.

#### Setup

For the setup, I am going to be referencing the slide below.
![LR_setup](figures/LinearRegression_setup.png)
We are given the usual setup, real-numbered training data with $d$ features, and labels to go with each point.
Whats new here is the loss function.
We can rewrite the above function as,
```math
    l(f_{\theta}) = \frac{1}{n}||X\theta -y ||^{2}_2
```
Where $X$ is a matrix whos rows represent each point in our dataset and $|| * ||^{2}_{2}$ is the $l_2$ norm.
To optimize this loss function, we need to take the gradient and solve for when the gradient is $0$.
The work for is below.

![LR_gradient](figures/gradient_lr.png)

Notice how we can actually represent $\theta$ as an equation.
This formal definition means that training $\theta$ is simple.
However there is a caveat, **not all matricies are invertable**.
Meaning we have to account for the case when the matrix cannot be inverted.
This lack of inversion is actually more common then you may think.
Most datasets are actually not invertable by default meaning that it is more common to see the next forms of linear regression than the original.

##### Ridge Regression

The first regression type on our list is ridge regression.
Ridge regression has a loss function defined below.

![ridge_loss](figures/ridge_regression.png)

When we solve for $\theta$ now we get,

![ridge_theta](figures/theta_ridge.png)

Notice the $\lambda n I$ term.
All matricies are invertable when sumed with the identity matrix.
The reasoning for why the previous sentence is true is not needed to understand ridge regression's purpose.
Because the matrix is invertable, $\theta$ will always exist.

The natural question then becomes, how should one pick $\lambda$?
Normally, you would pick $\lambda$ during the cross validation phase.
Algorithmically changing it so that it minimizes the loss function. 

Ridge regression's **main goal stated in the slides is actually to prevent large weights**.
Making the matrix invertable may be an after thought for these lectures, but always having a inveratble matrix is a very important component of the ridge regression as well.

##### LASSO: The Weird Brother To Ridge Regression

LASSO looks very similar to ridge at first glance.
Equation below defines LASSO as,

![LASSO_eq](figures/LASSO_eq.png)

The only difference is that we are now using the $l_1$ norm.
As I said earlier, at a glance there doesn't seem to be a big difference.
However, there is: **LASSO has no closed form solution**.
Which means that the gradient has many possible solutions.
We will discuss how to work with this later in this section.

The **goal** of LASSO is to encourage *spase* soultions, meaning $\theta$ will contain more zeros.
With more zeros means more sparsity in the solutions to $f_{\theta}$.

##### Evaluating Linear Regression Models

There are several metrics we can look to when evaluating linear regression models.
We will start with **R-squared** which is defined as,

```math 
    R^2 = 1 - \frac{\sum_j (y^j - f_{\theta}(x^i))^2}{\sum_j(y^j - \bar{y})^2}
```

Where $\bar{y}$ is the emperhical mean of the labels.
R-squared tells us how much variance in $y$ is predictable by $x$. 

The next is **MSE** which is mean squared error.
There is also **RMSE** which is the same thing as MSE except you just take the root of the MSE.
These tell us how far in euclidean space the predicted value is from the actual.

There is also the **MAE** which is the mean average error.
This tells us the average error over all predictions.
MAE will likely tell is the same thing as MSE but with out squaring the error furst.

##### Gradient Descent for Linear Regression

We are finally discussing gradient descent.
Boy am I excited.
Below summarizes gradient desecent quite well.

![gd_simmary](figures/gd_summary.png)

The goal is to optimize $g(\theta)$, as we saw earlier when dealing with ridge regression, we can take the gradient to optimize the parameters.
However, rather than just setting the equation equal to zero, gradient desecent is an iterative method that goes until you say stop.
The general idea of gradient descent is to iteratively find a minimum without solving for zero.
There are several equations in machine learning that do not have a gradient that exists at zero.
Thus, the iterative method is the only way to minimize $g(\theta)$.
Gradient descent is one of the most widely adopted concepts in machine learning.
It opened the flood gates for several loss functions to be viable without needing the function to be continous.
If that last sentence did not make sense, **continuous** means that the function exists for all points, and the function has a derivative at every point.
Below is the gradient descent equation for linear regression.

![gd_lr](figures/gd_lr.png)

Gradient descent is actually faster.
Inverting a giant matrix is actually a very difficult task for a computer to do.
For example, inverting a square $m$ x $m$ has a compelxity of $m^3$.
Gradient descent, on the other hand, has a compexity of $dnt$ where $d$ is number of features in $X$, $n$ is how many points there are, and $t$ is how many epochs or gradient descent iters you evaluate.
Thus, gradient descent is much faster on average especially for large datasets.

So what are the drawback of gradient descent(GD)?
There are a couple, to take the GD you need to have a **convex function**.
How do we define a convex function?
The figure below is a good starting point.

![convexity_chart](figures/convexity_chart.png)

The above figure is a fancy way of saying that the function must have a local minima between two points.
The figure straight up says that you must be able to draw a line from $x_1$ to $x_2$ and all points of the function must be below that line.

### Logistic Regression

After a grueling time with the likelihood estimator, we can start discussing logistic regression.
If you are unfamiliar with the likelihood estimator, I would highly encourage you to read that section under concepts.
We define the logistice distribution as,

![log_reg_eq](figures/log_reg_eq.png)

Which has a conditional distribution defined as,

![log_reg_cond](figures/log_reg_cond.png)

We define the loss of the log likelihood estimation as 

$$
    log\;likelihood(\theta|x^i,y^i) = log(P_{\theta}(y^i,x^i))
$$

Thus, the optimization problem becomes,

![log_reg_opt](figures/log_reg_opt.png)

So, with all of the above, whats the big deal with logistic regression?

1. It is bounded on $(0,1)$
2. Its symmetric, $1 - \sigma(z) = \sigma(-z)$
3. Its gradient is super easy, $\sigma^{`}(z) = \sigma(z)(1-\sigma(z))$

I included an example logistic regression file in `logistic_regression/lr.py`.

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


#### Occam's Razor
Why is this a machine learning concept?
Well, the translated phrase is "**Entities should not be multiplied beyond necessity.**"
Further translation, "**When you have two competing threories that make the same prediction the simpler one is better.**"
This concept is applied to **decision tree learning** because the central hypothesis is, "the simplest tree that classifies the training stances accurately will generalize."
So how does Occam's razor apply?
Well, there are fewer small sized trees than long ones.
A short tree is unlikely to fit the data very well whereas a large tree would be more likely.
How hard is it to find the shortest decision tree?
It is **NP-Hard** to find the shortest possible decision tree.
Therefore, the general solution is to greedily choose splits based on entropy.
For more on how that is done, see the decision trees section.


#### Overfitting

Model overfitting is one of the most common problems in machine learning.
A model overfits under two conditions:
1. A low error on training data.
2. A high error over whole distribution.

Again, there is a great visualization in the slides below.
![overfitting_vis](figures/overfitting_view.png)

Another good visualization of overfitting comes again from the slides.
![dtree_overfitting](figures/dtree_overfitting_example.png)
The figure above has an x-axis of tree size and a y-axis of accuracy.
As the tree grows in size, the training data becomes more accurate where as the test set becomes less accurate.
Thus it is a great example of overfitting.

To describe the general phenomenon further, there is another figure from the slides.
![dtree_overfitting](figures/capacity_chart_overfitting.png)
Above is shows the optimal capacity is the boundry of the overfitting zone.
It is the moment that the model begins to overfit to the training data.

#### Evaluation Metrics

This section will be very important and I will try to have code examples for a lot of these.

##### Tuning Set

A tuning set is a set that is not used from primary training purposes but used to select among models.
Terrible description, a tuning set is a subset of the training data which is used to observe how the data will "tune" the model.
When I say "tune" I mean make slight adjustments to the parameters on data the model has not seen yet and see how the hyperparameters effect accuracy.


##### Training/Testing Set Evaluation Stratedgies

We first should discuss why using a single training/testing set is not the best option for model evaluation.
A single training set does not tell us how sensitive accuracy is to a particular training sample.
A single partition of data into training/testing misses out on several details about the dataset.
First, if the testing data is too large we get a reliable estimate of accuracy but we have a lower variance estimate.
If the training data is too large it will be a better representation of the whole distribution.
As you can see there is this trade off of making one set larger than the other.
Thus, we must partition the training data

###### Stratedgy 1: Random Resampling

To address the first problem of a lack of variation one should randomly resample their training and testing data using random resampling.
The process is visualized below.
![random_resampling](figures/random_resampling.png)
As you can see, the training and testing data have a fixed size and then we randomly partion the data into the sets.
However, this process could lead to issues where the distribution is not properly modeled.
For example, a random partion based on the figure above would be where all of the `+` values are in the test set but none are in the training set leading to a terrible model.
To address that concern you can apply stratified sampling.
![stratified_sampling](figures/stratified_sampling.png)
With stratified sampling, the class porportions are maintined when preforming the partition.
This preserves the distribution while maintaining the random value selection.

###### Stratedgy 2: Cross Validation

Cross validation is considered to be the industry standard.
I was doing cross validation in COMP 532, implying that this concept is widely used.
Let is first visualize it below.
![cross_validation](figures/cross_validation.png)
What this does is partition the training data into $n$ equally sized partitions.
Training on $n-1$ partitions then test using the partition that was left out.
The most common value of $n$ is usually $10$.
One could also apply the stratified sampling technique described in the prior subsection.
This would assure that the distributions of the data would be preserved upon creating the subsets.
Cross validation makes efficent use of the dataset which is one of the reasons why it is so commonly used.
It is important to note here that these stratedgies evaluate the **learning method** rather than the hypothesis.
The last statement is shrouded in abstract terms so to simplify it, we are evaluating training rather than evaluating the model itself.
As in, these stratedgies are designed to examine how different training data changes the model rather than examining how well the model will do once applied to unseen data.

##### Learning Curves

A learning curve is the accuracy of a method as a function training set size.
For the learning curve, I made a python script in `utilities/learning_curve_example.py`.
The script goes over the algorithm and compares a Naive Bayes model and a SVM model.
The result is below.
![learning_curve_example](figures/learning_curve_example.png)


##### Confusion Matrices and Formalized Metrics

Confusion matrices are a great way to examine model performence per class.
Below is an example a multi-class confusion matrix.
![confusion_matrix](figures/confusion_matrix.png)
The majority of this lecture discusses a 2-class confusion matrix which looks like this.
![2_class_cm](figures/2_class_cm.png)
From the above we can now formally define accuracy, recall, precision, error, and false positive rate.

**Accuracy**
```math
    Accuracy = \frac{TP+TN}{FP+TP+FN+TN}
```

**Error**
```math
    Error = \frac{FP+FN}{FP+TP+FN+TN}
```

**Recall**
```math
    Recall = \frac{TP}{TP+FN}
```

**False Positive Rate**
```math
    FP\;Rate = \frac{FP}{TP+FP}
```

**Precision**
```math
    Precision = \frac{TP}{TP+FP}
```

##### ROC Curves

Receiver Operating Characteristic(ROC) curves plot the TP rate(recall) vs the FP-rate.
The area under the ROC curve is sometimes used to evaluate the model.
ROC curves can be confusing, what they show is how sensitive the models are.
Below is a visual of what the curves can look like.
![ROC_Curve_expected](figures/ROC_Curve_expected.png)
The **ideal** curve shows what great performence should be you can think of the relationship as $FN=0$.
For the **alg1** and **alg2** curves, they show different types of patterns an ROC curve could show.
How the lines look are not that important, how the curve is formulated depends on the confidence of the false postive.
This isn't that imporant because the confidence of the model could be a result of the data rather than the model.
How researchers use these curves is by calculating the area under the curve.
Below is a screenshot from a paper discussing what constitutes a good model.
![ROC_curve_ranges](figures/ROC_curve_ranges.png)
A morve curved algorithm, like **alg1**, implies that the locations of the false positives are faily close to one another.
Whereas a more straight curve, like **alg2**, demonstrate the TP and FP are spread out.
This means that **alg2** has several false positives after seeing the first one.
Therefore, what really matters is the AUC as the tabel above suggests.
The closer the AUC is to 1, the better the model's performence.
Below is a visual of what the algroithm looks like.
![ROC_CURVE_VIS](figures/ROC_Curve_example.png)

##### Percision/Recall Curve

The final metric we are going to be discussing is the PR curve. 
PR curves show the **fraction of predictions** that are false positives.
At the recall increases, the false positives should decrease.
This is due to the denomenators of the two have a decreasing relationship.
All false predictions are either $FP$ or $FN$ being that percision and recall account for them both, as one increases the other will decrease.
An ideal curve starts at (0,1) and ends at (1,1) indicating that for all thresholds, the percision and recall are at 100% which implies that there are no errors.
Below is a picture of two curves demonstrating this relationship.
![pr_cuve](figures/PR_curve_example.png)

#### Maximum Likelihood Estimation

A likelihood function captures the probability of seeing some data as a function of model parameters.
This is weird and took me a ChatGPT conversation to really understand what it means.
A likelihood function determines how well some parameters model a given dataset.
When you want to maximize of a set of parameters, the liklihood function's peaks tell you which parameters best model the distribution of the dataset.
If you are able to find parameters that model the dataset well, then the parameters that you have found will become your predictor function.
There are two examples that I will use to attempt to explain this: a uniform distribution and a normal distribution.

The uniform distribution will have a likelihood function that looks like,
$$
    L(\theta;X) = \prod_j{p_{\theta}(x_j)}
$$

Where $X$ is a dataset that is distributed uniformally, $p_{\theta}(x_j)$ is the probability of event $x_j$ given parameters $\theta$.
Let us think about this result, a uniform distribution is can be defined as a set of events where all of the events are equally likely.
Given a set of parameters $\theta$ the is the product of the events happening within $\theta$ if $x_j$ is outside of the realm of possibility of $\theta$ then the whole product turns to zero.
Thus, if $\theta$ can describe the entirety of the $X$ the likelhood function will return a non-zero value.
An example of this can be found in `utilities/likelihood_function_example_uniform.py`.


The normal distribution example can be found in the `utilities/likelihood_function_example.py` script.
When given a set of points over a normal distribution, the likelihood function will find the values of $\mu$ and $\sigma$ of the unknown normal distribution.

Now with this understanding, we can finally define the maximum likelihood as, 
![likelihood_eq](figures/likelihood_eq.png)
