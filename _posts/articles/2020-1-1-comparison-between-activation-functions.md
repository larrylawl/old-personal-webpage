---
layout: post
title: "Comparison between Logistic, TanH, and ReLU activation functions"
author: "Larry Law"
categories: articles
tags: [machine-learning]
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
- [Leaky ReLU](#leaky-relu)

## Definitions of activation functions

![Activation Function cheatsheet](/assets/img/2020-1-1-comparison-between-activation-functions/activation-functions-cheatsheet.png)

## Why does TanH work better than logistic?
Notice that the mean of the output of the TanH is approximately 0. The centering of data makes learning for the next layer slightly easier. 

> Logistic function is usually only used in the final layer for _binary classification problems_, where the output of the network is a probability (that exists in _(0, 1)_).

## The Vanishing Gradient Problem of TanH and logistic
Notice that at the extreme ends of the domain of both TanH and logistic, the **gradient gradually approaches 0**. The marginal change in gradient causes gradient-based learning methods to learn at a slower rate. 

Let us look at the math to understand this more concretely. For simplicity, let me make the following assumptions

1. Neural network with 4 hidden layers with a single neuron each
2. Let us focus on the logistic function (heuristic applies to TanH as both their functions gradually approaches 0)
3. Weights are initialised using the gaussian method (ie mean = 0 and sv = 1).

Thus, the partial derivative of the cost function with respect to the weight of the first neuron will be:

$$
\frac{\partial J}{\partial \Theta_{1}}=a_0 \delta_{1} = a_0 \times \sigma^{\prime}\left(z_{1}\right) \times w_{2} \times \sigma^{\prime}\left(z_{2}\right) \times w_{3} \times \sigma^{\prime}\left(z_{3}\right) \times w_{4} \times \sigma^{\prime}\left(z_{4}\right) \times \frac{\partial J}{\partial a_{4}}
$$

Notice that the derivative of the logistic function is multiplied _l_ times, where _l = 4_ here. As the derivative is always less than _0.25_ (proof below), the multiplication of the small derivatives will lead to an even smaller partial derivative of the cost function. With the partial derivative being such a small number and the optimisation objective being to reduce the partial derivative to 0 (ie minimum point), the change in partial derivative caused by each step of the gradient descent will be very small, and hence the parameters will also change marginally. The problem of an activation function's gradient approaching 0 leading to slower parameter learning is also known as the **vanishing gradient problem**.

![Vanishing Gradient Graph](/assets/img/2020-1-1-comparison-between-activation-functions/vanishing-grad-graph.png)

> Why is the derivative of the logistic function lesser than 0.25? 

1. Since the initialised weights have standard normal distribution, \$ -1 < w_i < 1 \$. 
2. Since the derivative of the sigmoid function = _f(x)(1 - f(x))_, the maximum value of *f'(x) = 0.25* (when *f(x) = 0.5*) 

> If the derivative of the activation function is > 1, the compounding effect of this derivative will lead to a very large partial derivative of the cost function. This problem is known as the **exploding gradient problem.**

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

## Leaky ReLU

The leaky ReLU is defined as:

$$
a = max(\delta z, z), 
$$

where \$ \delta \$ is a very small positive real number.

![Relu vs Leaky ReLU](/assets/img/2020-1-1-comparison-between-activation-functions/relu-vs-leaky-relu.png)

The advantage of the leaky ReLU (left) is that for the domain _z < 0_, the derivative will still be positive (as opposed to 0 for the ReLU (right)). This resolves the dead neuron problem. 

> In practice _p(z < 0)_ is very small, thus Relu and Leaky Relu should both work just fine.

<!-- omit in toc -->
## Credits
Sagar Sharma for the activation functions cheatsheet. Source [here.](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

Andrew Ng's Deep Learning course. Source [here](https://www.coursera.org/specializations/deep-learning).
