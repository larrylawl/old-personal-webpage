---
layout: post
title: "Gradient Descent"
author: "Larry Law"
categories: article
tags: [Machine Learning]
image: machine-learning.jpg
---
## Learning outcomes
1. Purpose of Gradient Descent
2. Intuition of Gradient Descent
3. Understand the math behind Gradient Descent
4. Understand simultaneous updates

## Prerequisite: Cost Function
Cost function is a function of our model's parameters that **measures the effectiveness of a machine learning model**. It is commonly denoted as,

$$ J(\theta_1, \theta_2, ..., \theta_n) $$

> Common cost functions include: root mean square error (RSME), Mean square error (MSE). 

## Purpose of Gradient Descent
**For parameter learning**. In other words, we are interested in finding the parameters such that the cost function of these parameters is minimum. 

## Intuition of Gradient Descent
Gradient descent achieves its purpose by _iteratively moving in the direction of steepest descent_.

In the context of the graph below, each cross marks the result of each iteration of the gradient descent, and they converge towards the (local) minimum of the cost function. 

![Gradient Descent](/assets/img/gradient-descent-3d.jpg)

## The math behind Gradient Descent

$$
\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right), where j = 0,1
$$

### Understanding the partial derivative term
Let's consider \$ R_2 \$ first. Suppose that we initialise \$ \theta_{1} \$ to the left of the minimum point of graph below. The derivative at that point is negative, thus the updated value of \$ \theta_{1} \$ will be increased (moves right) and hence _converges_ towards the local minimum, which is the purpose of gradient descent.

![Gradient Descent](/assets/img/gradient-descent-2d.jpg)

I intentionally used the word "_converges_". As can be seen from the graph, the derivative of each iteration as we approach the local minimum decreases, thus each update of \$ \theta_{1} \$ will correspondingly get smaller. This "natural" convergence is also the reason why the learning rate can be fixed, instead of decreasing as we approach the local minimum.

Now let's consider \$ R_3 \$. Recall that by taking the partial derivative of a multivariate function, I first need to fix the other variables as constants. This will give us the red line below.

![Partial Derivative of Single Variable](/assets/img/partial-derivative-sv.jpg)

This "reduces" the problem we had in \$ R_2 \$, where we have shown how the subtraction of the derivative eventually brings us to the local minimum.
> Local minimum as the cost function might have ≥1 minimums. The graph below has 2 minimums, as denoted by the red arrows.

![Geometric interpretation of partial derivative of a two variables](/assets/img/gradient-descent-3d.jpg)
> To learn more about partial derivatives, check out my post [here](./partial-derivatives.html).

### Understanding the learning rate, \$ \alpha \$
Intuitively, the learning rate denotes the _"size"_ of each descent. This makes sense as the learning rate is the coefficient of the partial derivative, which gives us the _direction of descent_. The image below summarises the implications of the different partitions of the chosen learning rate.

![Learning rate](/assets/img/learning-rate.jpg)

## Gradient Descent for multiple variables
Simultaneous updates brings us to the minimum of the cost function quicker.
![Simultaneous update](/assets/img/simult-update.jpg)

## Credits
Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning)

ML-cheat sheet for the definition of gradient descent. Source [here](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html).

Learning rate image. Source [here](https://medium.com/octavian-ai/how-to-use-the-learning-rate-finder-in-tensorflow-126210de9489)
