---
layout: post
title: "Why do we use ReLU functions over Sigmoid functions in Neural Networks?"
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
- [The Vanishing Gradient Problem of Sigmoid Function](#the-vanishing-gradient-problem-of-sigmoid-function)
- [Rectified Linear Unit (ReLU) Function Learns Faster](#rectified-linear-unit-relu-function-learns-faster)
- [The Dead Neuron Problem of ReLU](#the-dead-neuron-problem-of-relu)

## The Vanishing Gradient Problem of Sigmoid Function
Recall that the Sigmoid Function is defined as:

$$
h_\theta(x) = g(\theta^{\top}x) = g(z) = \frac{1}{1 + e^{-z}}
$$

![Sigmoid Function](/assets/img/2020-1-1-relu-vs-sigmoid/logistic.jpg)

Notice that at the extreme ends of the domain, the **gradient gradually approaches 0**. The marginal change in gradient causes gradient-based learning methods to learn at a slower rate. 

More concretely, let us look at the partial derivative of the Cost Function used in Backpropagation.

$$
\frac{\partial{J(\Theta)}}{\partial \Theta^{(l)}_{i, j}} =a^{(l)}_{j} \delta^{(l+1)}_{i}
$$

where \$ \delta \$, also known as the "error term", is a recurrence relation such that

$$
\\ \delta^{(l+1)} := \frac{\partial{J(\Theta)}}{\partial z^{(l+1)}} = \delta^{(l+2)} \Theta^{(l+1)} g^{\prime}\left(z^{(l+1)}\right),
\\ \delta^{(L)} = a^{(L)} - y,
$$

and _a_ is the output of the sigmoid function.

$$
a = g(z) := \sigma(z)
$$

Since \$ g^{\prime}\left(z^{(l+1)}\right) \$ changes marginally, the partial derivative of the cost function and hence the parameters will also change marginally. The problem of the gradient approaching 0 leading to slower parameter learning is also known as the **vanishing gradient problem**.

The slower learning matters as performance is increasingly dependent on size of data (and the learning from it), rather than the type of algorithm used. To understand why, consider the graph below, which plots the performance of different algorithms over number of samples.

![Big Data](/assets/img/2020-1-1-relu-vs-sigmoid/big-data.png)

## Rectified Linear Unit (ReLU) Function Learns Faster
The ReLU function is defined as 

$$
f(x) = max(0, x)
$$

![ReLU function](/assets/img/2020-1-1-relu-vs-sigmoid/relu.png)

The ReLU function learns faster because...
1. **It avoids the vanishing gradient problem:** For _z > 0_, the gradient is constant and hence it does not decrease.
2. **The gradient is easy to compute**: Gradient is either 0 or z, depending on the domain of z. (compare this with the computation of the gradient of the sigmoid function.)

> The ReLU is the most commonly used activation function in neural networks, especially in CNNs.

## The Dead Neuron Problem of ReLU

However, notice that for the domain _z < 0_, the gradient is 0. This is problematic, for it causes the partial derivative of the cost function wrt that particular neuron to be zero, which essentially eliminates the neuron. This is also known as the **dead neuron problem.** 

To circumvent this, people use leaky RELU, parametric RELU, and SWISH activation functions.

<!-- omit in toc -->
## Credits
Andrew Ng's Deep Learning course. Source [here](https://www.coursera.org/specializations/deep-learning).
