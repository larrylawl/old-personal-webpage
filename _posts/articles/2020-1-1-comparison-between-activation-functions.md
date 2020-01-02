---
layout: post
title: "Comparison between Logistic, TanH, and ReLU activation functions"
author: "Larry Law"
categories: articles
tags: [Machine Learning]
image: neural-network.jpeg
---
<div align="center">
    <i>"It's not who has the best algorithm that wins. It's who has the most data." - Andrew Ng</i>
</div>

<!-- omit in toc -->
## Learning outcomes
- [Definitions of activation functions](#definitions-of-activation-functions)
- [Why does TanH work better than logistic?](#why-does-tanh-work-better-than-logistic)
- [The Vanishing Gradient Problem of TanH and logistic](#the-vanishing-gradient-problem-of-tanh-and-logistic)
- [Why is ReLU preferred over TanH and logistic?](#why-is-relu-preferred-over-tanh-and-logistic)
- [The Dead Neuron Problem of ReLU](#the-dead-neuron-problem-of-relu)
- [Leaky Relu](#leaky-relu)

## Definitions of activation functions

![Activation Function cheatsheet](/assets/img/2020-1-1-comparison-between-activation-functions/activation-functions-cheatsheet.png)

## Why does TanH work better than logistic?
Notice that the mean of the output of the TanH is approximately 0. The centering of data makes learning for the next layer slightly easier. 

> Logistic function is usually only used in the final layer for _binary classification problems_, where the output of the network is a probability (that exists in _(0, 1)_).

## The Vanishing Gradient Problem of TanH and logistic

Notice that at the extreme ends of the domain of both TanH and logistic, the **gradient gradually approaches 0**. Without loss of generality to TanH, let us focus at the logistic function. 

The marginal change in gradient causes gradient-based learning methods to learn at a slower rate. More concretely, let us look at the partial derivative of the Cost Function used in Backpropagation.

$$
\frac{\partial{J(\Theta)}}{\partial \Theta^{(l)}_{i, j}} =a^{(l)}_{j} \delta^{(l+1)}_{i}
$$

where \$ \delta \$, also known as the "error term", is a recurrence relation such that

$$
\\ \delta^{(l+1)} := \frac{\partial{J(\Theta)}}{\partial z^{(l+1)}} = \delta^{(l+2)} \Theta^{(l+1)} g^{\prime}\left(z^{(l+1)}\right),
\\ \delta^{(L)} = a^{(L)} - y,
$$

and _a_ is the output of the logistic function.

$$
a = g(z) := \sigma(z)
$$

Since \$ g^{\prime}\left(z^{(l+1)}\right) \$ changes marginally, the partial derivative of the cost function and hence the parameters will also change marginally. The problem of the gradient approaching 0 leading to slower parameter learning is also known as the **vanishing gradient problem**.

The slower learning matters as performance is increasingly dependent on size of data (and the learning from it), rather than the type of algorithm used. To understand why, consider the graph below, which plots the performance of different algorithms over number of samples.

![Big Data](/assets/img/2020-1-1-comparison-between-activation-functions/big-data.png)

## Why is ReLU preferred over TanH and logistic?

The ReLU function is defined as 

$$
f(x) = max(0, x)
$$

![ReLU function](/assets/img/2020-1-1-comparison-between-activation-functions/relu.png)

The ReLU function learns faster because...
1. **It avoids the vanishing gradient problem:** For _z > 0_, the gradient is constant and hence it does not decrease.
2. **The gradient is easy to compute**: Gradient is either 0 or z, depending on the domain of z. (compare this with the computation of the gradient of the logistic function.)

> Another important property of the ReLU function is that it is non-linear. Specifically, it is a _piecewise linear function_. The non-linear property is important as, suppose not, the output of the activation function, _a_, will essentially be a linear regression, which reduces the problem to a linear regression problem (which can be solved using linear regression).

> The ReLU is the most commonly used activation function in neural networks, especially in CNNs.

## The Dead Neuron Problem of ReLU

However, notice that for the domain _z < 0_, the gradient is 0. This is problematic, for it causes the partial derivative of the cost function wrt that particular neuron to be zero, which essentially eliminates the neuron. This is also known as the **dead neuron problem.** 

To circumvent this, people use leaky ReLU, parametric ReLU, and SWISH activation functions. Let's consider the leaky ReLU to understand how it resolves the dead neuron problem.

## Leaky Relu

The leaky ReLU is defined as:

$$
a = max(\delta z, z), 
$$

where \$ \delta \$ is a very small positive real number.

![Relu vs Leaky Relu](/assets/img/2020-1-1-comparison-between-activation-functions/relu-vs-leaky-relu.png)

The advantage of the leaky ReLU (left) is that for the domain _z < 0_, the derivative will still be positive (as opposed to 0 for the ReLU (right)). This resolves the dead neuron problem. 

> In practice _p(z < 0)_ is very small, thus Relu and Leaky Relu should both work just fine.

<!-- omit in toc -->
## Credits
Sagar Sharma for the activation functions cheatsheet. Source [here.](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

Andrew Ng's Deep Learning course. Source [here](https://www.coursera.org/specializations/deep-learning).
