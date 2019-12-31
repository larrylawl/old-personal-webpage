---
layout: post
title: "Interpretation for Partial Derivatives"
author: "Larry Law"
categories: articles
tags: [Calculus]
image: calculus.jpg
---

(For a more detailed explanation, check out Khan Academy's post [here](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/introduction-to-partial-derivatives))

## Learning outcomes
1. Interpretation of partial derivatives wrt (a) single variable (b) multivariables
2. Geometric intepretation of (1)
3. Computation of partial derivatives, and its relation to (1)

## Interpretation of partial derivatives wrt single variable
Recall that derivatives tell us how a small change in x changes f(x).

$$ \frac{df(x)}{dx} $$

Similarly in the multivariate world, the partial derivative below tells us how a small change in \$ x_1 \$ changes the multivariate function. 

$$ \frac{df(x_1, x_2)}{dx_1} $$ 

In other words, we are interested in the derivative of the red line below for a fixed value of \$ x_2 \$.

![Geometric interpretation of partial derivative of a single variable](/assets/img/2019-12-09-partial-derivatives/partial-derivative-sv.jpg)

Note that the partial derivative of a function of \$ x_1 \$ is still a function of \$ x_1 \$. This function outputs the gradient of the multivariate function with respect to \$ x_1 \$, and with \$ x_2 \$ held constant. 

## Interpretation of partial derivatives wrt multiple variables

$$ \frac{df(x_1, x_2, ... , x_n)}{dx_1dx_2...dx_n} $$

Similarly, the partial derivatives wrt multiple variables tells us how a small change in all \$ x_i \$ changes the multivariate function. It is a function of all \$ x_i \$ that outputs the gradient of the multivariate function with respect to all \$ x_i \$. In \$ R_3 \$, this is the gradient of any surface we see.

![Geometric interpretation of partial derivative of a two variables](/assets/img/2019-12-09-partial-derivatives/gradient-descent-3d.jpg)

## Computation of partial derivatives wrt multiple variables
Recall that to compute partial derivatives wrt \$ x_i \$, we differentiate wrt to our variable of interest, and _treat the rest of the variables constant_. We iteratively repeat the partial differentiation until all variables have been partially differentiated. But can _treating the rest of the variables constant_ allow us to arrive at the interpretation we have above?

> Partial derivatives wrt multiple variables tells us how a small change in all \$ x_i \$ changes the multivariate function

To reconcile the computation with the interpretation, I like to think of each partial differentiation computation as an _expression._

$$ \frac{df(x_1, x_2, ... , x_n)}{dx_1dx_2...dx_n} $$

1. \$ \frac{\partial f}{\partial x_1} \$ describes how a small change in \$ x_1 \$ changes \$ f \$.
2. \$ \frac{1}{\partial x_2} \frac{\partial f}{\partial x_1} \$ describes how a small change in \$ x_2 \$ changes \$ \frac{\partial f}{\partial x_1} \$, which itself has captured how a small change in \$ x_1 \$ changes \$ f \$. Thus the expression as a whole describes how a small change in \$ x_1 \$ and \$ x_2 \$ changes \$ f \$.
3. Repeat until \$ x_n \$

## Application of partial derivatives: Gradient Descent
Learn more about Gradient Descent in my post [here](./gradient-descent.html).
