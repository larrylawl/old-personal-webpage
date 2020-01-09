---
layout: post
title: "Coursera, deeplearning.ai: Deep Learning Notes"
author: "Larry Law"
categories: notes
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
  - [Optimisation Alogrithms](#optimisation-alogrithms)
    - [Difference Between Batch, Mini-Batch, and Stochastic Gradient Descent](#difference-between-batch-mini-batch-and-stochastic-gradient-descent)
    - [Exponentially Weighted Average](#exponentially-weighted-average)
    - [Bias Correction](#bias-correction)
    - [Gradient Descent with Momentum](#gradient-descent-with-momentum)
    - [RMSprop](#rmsprop)
    - [Adam](#adam)
    - [Learning Rate Decay](#learning-rate-decay)
    - [The problem of local optima](#the-problem-of-local-optima)
    - [Notable Quiz Questions](#notable-quiz-questions-1)

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
2. **To reduce variance (wo affecting bias):** Get more data

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

## Optimisation Alogrithms
1. Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
2. Use random minibatches to accelerate the convergence and improve the optimization
3. Know the benefits of learning rate decay and apply it to your optimization

### Difference Between Batch, Mini-Batch, and Stochastic Gradient Descent

The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is *the number of examples* you use to perform one update step.

> With a well-turned mini-batch size, usually mini-batch gd outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large). Mini-batch size performs better than stochastic as it leverages vectorisation.

> Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

### Exponentially Weighted Average
Weighted average of all the previous velocities, where the coefficients (or probability) exponentially decreases as t decreases. 

$$
v_t = \beta v_{t - 1} + (1 - \beta)\theta_t
$$

![Exponentially Weighted Average](/assets/img/2019-12-31-coursera-dl-notes/exponentially-weighted-average.png)

> coefficient => \$ f(x) = 0.1 \times 0.9^{-x} \$, where x is \$ \v_t \$). 

Advantage of exponentially weighted average is that (a) its O(1) space complexity while (b) considering the weighted average of all the previous parameters, not jus the current parameter.

> Why is \$ v_t \$ approximately average over \$ \frac{1}{1-\beta} days? \$

$$
(1 - \epsilon)^{\frac{1}{\epsilon}} = \frac{1}{e} \\
0.9^{10} \approx 0.35 \approx \frac{1}{e}
$$

In the example above, when \$ \beta = 0.1 \$, it is an approximation for the weighted average over the past 10 days, as the coefficients for the days after that are too small for the number to be meaningful. 

### Bias Correction

$$
\begin{array}{l}{v_{d W^{[l]}}=\beta_{1} v_{d W^{[l]}}+\left(1-\beta_{1}\right) \frac{\partial \mathcal{J}}{\partial W^{[l]}}} \\ {v_{d W^{[l]}}^{c o r r e c t e d}=\frac{v_{d W}[l]}{1-\left(\beta_{1}\right)^{t}}}\end{array}
$$

- When _t_ is small, then bias correction scales up \$ v_t \$ (as the velocity, \$ v_{t - 1} \$, is very small)
- When _t_ is large, then bias correction has no effect on \$ v_t \$ (denominator ~= 1)

### Gradient Descent with Momentum

Intuitively, GD with Momentum (a) dampens oscillations in directions of high curvature by cancelling out gradients with opposite signs and (b) builds up velocity in directions (towards min) with a gentle but consistent gradient (since they don't cancel each other out).

$$
\begin{aligned}\left\{\begin{array}{l}{v_{d W^{[l]}}} & {=\beta v_{d W^{[l]}}+(1-\beta) d W^{[l]}} \\ {W^{[l]}} & {=W^{[l]}-\alpha v_{d W^{[l]}}}\end{array}\right.\\\left\{\begin{array}{l}{v_{d b^{[l]}}} & {=\beta v_{d b^{[l]}}+(1-\beta) d b^{[l]}} \\ {b^{[l]}} & {=b^{[l]}-\alpha v_{d b^{[l]}}}\end{array}\right.\end{aligned}
$$

\$ d W^{[l]} \$ is acceleration, \$ v_{d W^{[l]}} \$ is velocity, \$ beta \$ (being <1) is friction.

> Why is this called GD with momentum?

Its momentum makes it keep going in the previous direction.

**Comparison of Gradient Descent w/ various momentums**
1. Gradient Descent
2. Gradient Descent w smaller momentum
3. Gradient Descent w larger momentum

![Gradient Descent Comparison](/assets/img/2019-12-31-coursera-dl-notes/grad-descent-comparison.png)

> Why does gradient descent oscillate so sharply? 

Because you are taking the derivative only from the previous iteration (ie beta = 0), and not the exponential weighted average of all the previous iterations, thus it moves more sharply.


### RMSprop
TODO.
<!-- What if it's descending towards the correct direction? -->
<!-- Derivatives are much larger in the vertical direction than the horizontal direction -->
<!-- Gradient towards the cost function is near 0 (horizontal) -->
<!-- Why square? -->
<!-- Why mean? -->

### Adam
Combination of gradient descent with momentum and RMSprop.

$$
\left\{\begin{array}{l}{v_{d W^{[l]}}=\beta_{1} v_{d W^{[l]}}+\left(1-\beta_{1}\right) \frac{\partial \mathcal{J}}{\partial W^{[l]{l}}}} \\ {v_{d W^{[l]}}^{c o r r e c t e d}=\frac{v_{d W^{[l]}}}{1-\left(\beta_{1}\right\}^{t}}} \\ {s_{d W^{[l]}}=\beta_{2} s_{d W^{[l]}}+\left(1-\beta_{2}\right)\left(\frac{\partial J}{\partial W^{[l]}}\right)^{2}} \\ {s_{d W^{[l]}}^{c o r r e c t e d}=\frac{s_{d W l}[l]}{1-\left(\beta_{2}\right)^{t}}} \\ {W^{[l]}=W^{[l]}-\alpha \frac{v_{d W^{(l)}}^{r_{\text {diving }}}}{\sqrt{s_{d W[l]}^{s s w(l)}+\varepsilon}}}\end{array}\right.
$$

> Not different \$ beta_i \$ for _s_ (RMS prop) and _v_ (grad desc w momentum)


Some advantages of Adam include:
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except \$ \alpha \$)

Adam paper [here](https://arxiv.org/pdf/1412.6980.pdf)

### Learning Rate Decay

![learning rate decay](/assets/img/2019-12-31-coursera-dl-notes/learning-rate-decay.png)

1. When you just started training, it's okay to take bigger steps.
2. When nearing minimum point, take smaller steps so that you oscillate in a tighter region around this minimum. 

### The problem of local optima

![local min vs saddle points](/assets/img/2019-12-31-coursera-dl-notes/local-min-vs-saddle.png)

In higher dimensional space, it's more likely to obtain *saddle points* instead of *local minimums*. This is because to obtain a local minimum, **all** _n_ dimensions need to be a convex-like function. In contrast to obtain a saddle point, you only need a mixure of convex and concave functions. Consequently, we need not worry too much about gradient descent being stuck at local minimums.

![saddle point](/assets/img/2019-12-31-coursera-dl-notes/saddle.png)

However, the problem with saddle points is that it take very long to go down the plateau. Thus there's a need for gradient descent algorithms which work faster.

### Notable Quiz Questions

![Quiz on exponentially weighted average](/assets/img/2019-12-31-coursera-dl-notes/quiz-exponentially-weighted-avg.png)

When you increase \$ \beta \$ , you are taking into account more days, thus the graph adapts more slowly (consequently smoother), and hence the red line is shifted slightly to the right
