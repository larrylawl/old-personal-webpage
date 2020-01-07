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

<!-- omit in toc -->
## Table of Contents
- [Course 1: Neural Networks and Deep Learning](#course-1-neural-networks-and-deep-learning)
  - [Defensive Programming with Matrixes](#defensive-programming-with-matrixes)
  - [Activation Functions](#activation-functions)
  - [Notable Quiz Questions](#notable-quiz-questions)
- [Course 2: Improving Deep Neural Networks - Hyperparameter tuning, Regularization and Optimization](#course-2-improving-deep-neural-networks---hyperparameter-tuning-regularization-and-optimization)
  - [Practical Aspects of Deep Learning: Learning Outcomes](#practical-aspects-of-deep-learning-learning-outcomes)
    - [Setting Up Your Machine Learning Application](#setting-up-your-machine-learning-application)
      - [Decreasing Significance of Bias-Variance Tradeoff](#decreasing-significance-of-bias-variance-tradeoff)
      - [Train/Dev/Test split and tips](#traindevtest-split-and-tips)
    - [Regularising Your Neural Network](#regularising-your-neural-network)
      - [Dropout Regularisation](#dropout-regularisation)
      - [Early stopping](#early-stopping)
      - [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
      - [Weight Initialisation for Deep Networks](#weight-initialisation-for-deep-networks)

# Course 1: Neural Networks and Deep Learning
## Defensive Programming with Matrixes
```
a = np.random.randn(5) 
# a.shape = (5,) 
# rank 1 array (as it has only one axis) - don't use!

a = np.random.randn(5, 1) # a.shape = (5, 1)
a = np.random.randn(1, 5) # a.shape = (1, 5)
assert(a.shape == (5, 1))
```

## Activation Functions
Refer to my article [here](/articles/comparison-between-activation-functions.html)

## Notable Quiz Questions
> Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

True, Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector.

# Course 2: Improving Deep Neural Networks - Hyperparameter tuning, Regularization and Optimization

## Practical Aspects of Deep Learning: Learning Outcomes
1. Recall that different types of initializations lead to different results
2. Recognize the importance of initialization in complex neural networks.
3. Recognize the difference between train/dev/test sets
4. Diagnose the bias and variance issues in your model
5. Learn when and how to use regularization methods such as dropout or L2 regularization.
6. Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
7. Use gradient checking to verify the correctness of your backpropagation implementation

### Setting Up Your Machine Learning Application
#### Decreasing Significance of Bias-Variance Tradeoff
1. **To reduce bias (wo affecting variance):** Get more data and regularise
2. 1. **To reduce variance (wo affecting bias):** Get more data

#### Train/Dev/Test split and tips
1. **Smaller dataset (1K-10K):** 60/20/20 split.
2. **Large dataset (> 1M):** 98/1/1 split. Purpose of the dev/test set is to evaluate the algorithm, thus you don't need so much data.
3. **Obtain the training and test sets from the same distribution:** As more data wins, tendency is to get training data from different distributions from the test data, which will cause your model to learn wrongly.

### Regularising Your Neural Network
#### Dropout Regularisation
Intuition: Not put all your eggs in one basket. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.


```py
# l = 3, keep_prob = 0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3) # a3 *= d3
a3 /= keep_prob # Invert the dropout by maintaining the expected value of a3
```

> Why do we remove dropout in at test time?

Ensure that our predictions are deterministic at test time.

**Drawback**: cost function *J* is less well defined (as every iteration we eliminate nodes at random)

**Solution**: Turn off dropout first (ie `keep_prob = 1`) and check that J is monotonely decreasing. This ensures that our gradient descent is implemented correctly. Then turn on dropout.

#### Early stopping
Not recommended as it couples both objectives of optimising cost function J and solving overfitting (as opposed to gradient descent for J and regularisation for overfitting).

#### Vanishing and Exploding Gradients
Refer to the article [here](/articles/comparison-between-activation-functions.html)

#### Weight Initialisation for Deep Networks

<!-- Clarify how does increasing z relate to gradient? -->

<!-- Why does weight initialisation work?

Hi, I have 2 questions wrt this slide.

1. How does preventing the value of *z* from increasing/decreasing exponentially circumvent the vanishing gradient problem? Aren't we concern with the gradient of the activation function (ie g'(z)), rather than the input argument (ie z)? 
2. Suppose 1. is true, why does adjusting the value of variance prevent the value of z from increasing/decreasing exponentially? The expected value of z, E[z], will still be 0 no? Here's my working:

E[z] = E[w1x1 + ... + wnxn]
= E[w1x1] + ... + E[wnxn] (by linearity of expectations)
= E[w1]E[x1] + ... + E[wn]E[xn] (since w and x are independent)
= 0 (since E[w] = 0) -->

<!-- 1. Prevent z from blowing up (refer to graph on `activation function) --> As n increases, wi decreases
2. By keeping my variance inversely proportional to n, as n increases, w will spread v little from the mean which is 0, thus it decreases.  -->

```py
def initialize_parameters_he(layers_dims):
...
for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt([np.divide(2, layers_dims[l - 1])])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
```
This works as \$ Var(aX) = a^2Var(X) \$

1. Weight of Relu: \$ \frac{2}{n^{l - 1}} \$ (*He Initialisation*)
2. Weight of TanH: \$ \frac{1}{n^{l-1}} or \frac{2}{n^{l-1} + n^{l}} \$ (*Xavier Initialisation*)
