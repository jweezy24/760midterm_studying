## Practice Questions Solutions

This file will contain my answers to the practice questions.
The practice questions will be in a pdf in this folder.

### True / False

#### My Answers 
1. This question is **True**. The reason why it is true is that it is just the definition of unsupervised learning.
2. I believe this question is **False**. This statement needs to be more specific. It says "In cross validation, we train the classifier using all of the data, and predict the classification of the left out set." The part of the statement that is true is the fact that all of the data is used, eventually, when training a model using CV. However, how can there be a left out set if all of the data is used for training? Also, when we evaluate the classification, we do not use a single left out set to evaluate the classification.
3. This statement is **True**. We can see the relationship of capacity and error in one of the charts in the notes.
4. This one is just verifying if the node count in a setup is actually 30. There are two nodes for input, followed by a 4 node ReLU layer. Which will need 8 connections and thus 8 weights. There is then a second layer which will need an additional 12 weights 3 connections per node and there are 4 nodes. There is also a single output layer which will need another 3 weights. Bringing the total to 8+12+3=23. This is **False**.

#### Correct Answers
All correct.

### Neural Networks

#### My Answers
1. The first question is to derive the derivative of the ReLU function. Which we actually can't do by laws of math but we can derive something close. ReLU is defined as $ f(x) = x \; or \; 0 $ zero based on some threshold. Thus, the derivative of the function does not exist because it is a piece-wise function. However, and approximate derivate is $f^{`}(x) = 1 \; or \; 0$ depending on some threshold.
2. For this one I'm going to need some more space.

$$
    tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}\\
    tanh^{`}(x) = 1 - \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2}
$$
3. This question is annoying. What does it mean to compare the functions? ReLU is just a line at 0 then when it is greater then zero it is a line going up. Hyperbolic tangent and sigmoid have similar shapes but sigmoid has more values that are not 1 or -1. IDK what this question wants from me.

#### Correct Answers
All correct except for the third one. I am straight up supposed to compare them on every thing I can think of. Fuck that dude.

### Evaluation Metrics

#### My Answer
This one is just a table.

Based on the table if we set the confidence positive to be 0.5. We will have $TP = 4$, $TN = 3$, $FP = 1$, and $FN = 2$.

Thus, accuracy = $\frac{7}{10}$
fp-rate = $\frac{1}{5}$ This one was wrong due to me writing it down incorrectly
precision = $\frac{4}{5}$
recall = $\frac{4}{6}$


#### Correct Answer
I had the correct idea but wrote the false positive rate wrong.

### Problem 4

#### My Answer

MLE = $\frac{e^{\frac{-(x-\mu)^2}{2\sigma^2}}}{\sqrt{2\pi\sigma^2}}$

NLL = $\frac{1}{n}\sum_{x\in X} -\log(MLE(x))$

As $n$ goes to infinity NLL goes to 0. Because we take the log of the MLE in the NLL we get a straight line who's slope tends to zero as $n$ goes to infinity. A line is not convex thus there is no convexity to observe. 

#### Correct Answer
I got it very wrong. When deriving the MLE for $\mu$ and $\sigma$ I am supposed to take the partial of the thing I got. I wrote the initial thing correctly but I did not understand what it meant to get the MLE for the variables. I got the NLL correct. When n goes to infinity it actually will converge to the true value of the distribution.