---
layout: post
title: "Math behind Large Margin Classifications for Support Vector Machines"
author: "Larry Law"
categories: journal
tags: [Machine Learning]
image: machine-learning.jpg
---
<div align="center">
    <i>"Go down deep enough into anything and you will find mathematics." - Dean Schlicter</i>
</div>

<!-- omit in toc -->
## Learning outcomes
- [Prerequisite: Geometric Interpretation of Dot Product](#prerequisite-geometric-interpretation-of-dot-product)
- [Large Margin Intuition](#large-margin-intuition)
- [Math Behind Large Margin](#math-behind-large-margin)

## Prerequisite: Geometric Interpretation of Dot Product

The dot product is defined for two vectors X and Y by

$$
X \cdot Y = |X||Y|cos\theta
$$

where \$ \theta \$ is the angle between the vectors and \$ \lvert X \lvert \$ is the length of the vector (aka the norm). Consequently, the dot product has the geometric interpretation of the length of projection X on Y, multiplied by the norm of Y (or vice versa, since dot product is commutative).

$$
X \cdot Y = p \times |Y|
$$

where p is the length of projection of X on Y.

## Large Margin Intuition
Support vector machines (SVM) are known as large margin classifiers. Intuitively, this is because the minimisation of the cost function will lead to large margins (ie the margin in the center).

![Support Vector Machine Margins](/assets/img/svm-margins.png)

Let us now look at the math justifying this intuition.

## Math Behind Large Margin
Recall that the cost function of SVM is as such:

$$
J(\theta) = C \sum_{i=1}^{m}\left[y^{(i)} cost_{1}\left(\theta^{T} x^{(i)}\right)+\left(1-y^{(i)}\right) cos t_{0}\left(\theta^{T} x^{(i)}\right)\right]+\frac{1}{2} \sum_{i=1}^{n} \theta_{j}^{2}
$$

> _C_ is a penalisation parameter that have the opposite role of the parameter \$ \lambda \$. Concretely, when C decreases, \$ \lambda \$ increases, the regularisation term increases, hence it mitigates overfitting.

If y = 1, the first function (ie `cost`) will be the graph on the left. If y = 0, it will be the graph on the right.

![Support Vector Machine](/assets/img/svm.png)

At the optimal minimisation of the cost function, the first term will equals zero. In order for the first term to be zero, 
1. If \$ y^{(i)} = 1 \$, \$ \theta^{T}x^{(i)} \$ ≥ 1 (refer to the left graph)
2. If \$ y^{(i)} = 0 \$, \$ \theta^{T}x^{(i)} \$ ≤ -1 (refer to the right graph)

Thus we can rewrite the cost function as 

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{n} \theta_{j}^{2}
$$

such that 
1. If \$ y^{(i)} = 1 \$, \$ \theta^{T}x^{(i)} \$ ≥ 1
2. If \$ y^{(i)} = 0 \$, \$ \theta^{T}x^{(i)} \$ ≤ -1

Since 

$$
\theta^{T}x = \theta \cdot x = p \times |\theta|
$$

We can further rewrite the optimised cost function as 

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{n} \theta_{j}^{2}
$$

such that 

1. If \$ y^{(i)} = 1 \$, \$ p^{(i)} \times \lvert \theta \lvert \$ ≥ 1
2. If \$ y^{(i)} = 0 \$, \$ p^{(i)} \times \lvert \theta \lvert \$ ≤ -1
3. where \$ p^{(i)} \$ is the projection of \$ x^{(i)} \$ on the vector \$ \theta \$

Since the optimal cost function is still dependent on \$ \lvert \theta \lvert \$, \$ \lvert \theta \lvert \$ will likely be small at the minimum point of the cost function. If \$ \lvert \theta \lvert \$ is small, then the projection _p_ has to be large, thus the decision boundary is large.

![Support Vector Margin](/assets/img/svm-margin-2.png)

This is why, the SVM is associated as a large boundary classifier.

<!-- omit in toc -->
## Credits
Mathworld for the interpretation of Dot Product. Source [here](http://mathworld.wolfram.com/DotProduct.html).

Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning).
