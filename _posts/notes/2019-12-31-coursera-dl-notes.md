---
layout: post
title: "Coursera, deeplearning.ai: Deep Learning Notes"
author: "Larry Law"
categories: notes
tags: [notes, Deep Learning]
image: neural-network.jpeg
hidden: true
---
Lecturer: Professor Andrew Ng <br>
Course available [here](https://www.coursera.org/specializations/deep-learning).<br>

# Course 1: Neural Networks and Deep Learning
## Learning Outcomes
1. Defensive Programming with Matrixes
2. Activation Functions
3. Notable Quiz Questions

### Defensive Programming with Matrixes
```
a = np.random.randn(5) 
# a.shape = (5,) 
# rank 1 array (as it has only one axis) - don't use!

a = np.random.randn(5, 1) # a.shape = (5, 1)
a = np.random.randn(1, 5) # a.shape = (1, 5)
assert(a.shape == (5, 1))
```

### Activation Functions
Refer to my article [here](/articles/comparison-between-activation-functions.html)

### Notable Quiz Questions
> Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

True, Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector.

# Course 2: Improving Deep Neural Networks - Hyperparameter tuning, Regularization and Optimization
## Practical Aspects of Deep Learning
### Learning Outcomes
1. Dropout Regularisation

Intuition: Can't rely on any one feature, so we have to spread out weight

Prevent overfitting (putting all eggs in one basket)

> Why divide by `keep_prob`

Maintain expected value of _a3_

> Why do we remove dropout in at test time?

Ensure that our predictions are deterministic at test time.

Drawback: cost function J is less well defined (as every iteration we eliminate nodes at random)

Solution: Turn off dropout first (ie `keep_prob = 1`) and check that J is monotonely decreasing. This ensures that our gradient descent is implemented correctly. THen turn on dropout.

### Early stopping
> Not recommended as it couples both optimising cost function J and solving overfitting.

### Vanis