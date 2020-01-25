---
layout: post
title: "Calculus of Backpropagation"
author: "Larry Law"
categories: articles
tags: [machine-learning]
image: neural-network.jpeg
---
## Learning outcomes
1. Intuition of Backpropagation
2. Proof for the Partial Derivative Term in Backpropagation
3. Calculus of the Gradient Vector of Backpropagation
4. Fitting the Calculus with the Intuition

<!-- Disclaimer that this explains Andrew Ng's course -->
_(This articles explains the calculus of the backpropagation algorithm in Andrew Ng's machine learning course, which can be found [here](https://www.coursera.org/learn/machine-learning).)_
![Andrew Ng's course: Backpropagation Algorithm](/assets/img/2019-12-18-calculus-for-backpropagation/backpropagation-algo.png)

## Prerequisite
Standard notations for neural networks as used in Andrew Ng's course. With some discrepancies, definition of the notations can be found in this [link](https://cs230.stanford.edu/files/Notation.pdf).

## Intuition of Backpropagation
Recall that the the purpose of Backpropagation is to _learn the parameters_ for a _neural network._ Backpropagation learns the parameters of layer \$ l \$, from the error terms of the layer \$ l + 1 \$. Thus, the parameter learning is propagated from the back.

## Proof for the Partial Derivative Term in Backpropagation
Suppose that the activation function chosen is the sigmoid function.
For simplicity, let us assume that the size of the training set is 1 (ie _m_ = 1).

$$
a = g(z) := \sigma(z)
$$

The claim: 

$$
\frac{\partial{J(\Theta)}}{\partial \Theta^{(l)}_{i, j}} =a^{(l)}_{j} \delta^{(l+1)}_{i}
$$

where \$ \delta \$, also known as the "error term", is a recurrence relation such that

$$
\\ \delta^{(l+1)} := \frac{\partial{J(\Theta)}}{\partial z^{(l+1)}} = \delta^{(l+2)} \Theta^{(l+1)} g^{\prime}\left(z^{(l+1)}\right),
\\ \delta^{(L)} = a^{(L)} - y,
$$

Proof can be found [here](https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/MVwN-LpLEeiBxhK7qjbMkg). Alternatively, I have included pictures of the proof at the bottom of the page.

## Calculus of the Gradient Vector of Backpropagation
Since the neural network needs to be trained on _m_ training samples, the partial derivative term needs to be averaged out over all these samples.

$$
{\frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{i, j}}} 
= D^{(l)}_{i, j}
= \frac{1}{m} \sum_{k=1}^{m} \frac{\partial J_{k}}{\partial \Theta^{(l)}_{i, j}}
$$

The summation term is precisely the update rule in the algorithm.

$$
\Delta_{i, j}^{(l)}:=\Delta_{i, j}^{(l)}+a_{j}^{(l)} \delta_{i}^{(l+1)} 
$$

Adding the regularisation terms give us


$$
\begin{aligned} \cdot & D_{i, j}^{(l)}:=\frac{1}{m}\left(\Delta_{i, j}^{(l)}+\lambda \Theta_{i, j}^{(l)}\right), \text { if } j \neq 0 \\ \cdot & D_{i, j}^{(l)}:=\frac{1}{m} \Delta_{i, j}^{(l)} \text { lf } j=0 \end{aligned}
$$

Finally, let's compute the gradient vector, which will be passed as an argument to parameter learning methods.

> Recall that gradient vector of a multivariable function packages the partial derivative of its parameters into a vector.

Fitting the partial derivatives into the gradient vector, we get

$$
\nabla J=\left[\begin{array}{c}{\frac{\partial J}{\partial \Theta^{(1)}_{1, 1}}} \\ {\frac{\partial J}{\partial \Theta^{(1)}_{1, 2}}} \\ {\vdots} \\ {\frac{\partial J}{\partial \Theta^{(L-1)}_{s_{l+1}, s_{l}}}}\end{array}\right]
$$

> For simplicity, bias terms are ignored.

And we are done :).

## Fitting the Calculus with the Intuition

Learning of a particular \$ \Theta^{(l)}_{i, j} \$ is dependent on the partial derivative of the cost function, which in turn is dependent on the error term.

$$
\frac{\partial{J(\Theta)}}{\partial \Theta^{(l)}_{i, j}} =a^{(l)}_{j} \delta^{(l+1)}_{i}
$$

This error term is a recurrence relation. In other words, the error term of layer _l+1_ is dependent on the error term of layer _l+2_, which ends at layer _L_.

$$
\\ \delta^{(l+1)} := \frac{\partial{J(\Theta)}}{\partial z^{(l+1)}} = \delta^{(l+2)} \Theta^{(l+1)} g^{\prime}\left(z^{(l+1)}\right)
\\ \delta^{(L)} = a^{(L)} - y,
$$

Thus Backpropagation is a process wherein the neural network _learns from the back (ie the last layer)._

## Credits
Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning).

Neil Ostrove for the proof of the partial derivative term.

![Back propagation proof 1](/assets/img/2019-12-18-calculus-for-backpropagation/backpropagation-proof-1.png)

![Back propagation proof 2](/assets/img/2019-12-18-calculus-for-backpropagation/backpropagation-proof-2.png)
