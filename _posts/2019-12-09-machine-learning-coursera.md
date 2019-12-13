---
layout: post
title: "Machine Learning Coursera Notes"
author: "Larry Law"
categories: journal
tags: [Machine Learning]
image: machine-learning.jpg
---
<!-- omit in toc -->
# Machine Learning Coursera Notes

Lecturer: Professor Andrew Ng <br>
Source: [here](https://www.coursera.org/learn/machine-learning/home/welcome)

<!-- omit in toc -->
## Table of Contents
- [Week 1](#week-1)
  - [Learning Outcomes](#learning-outcomes)
  - [Machine learning](#machine-learning)
  - [Cost Function](#cost-function)
  - [Gradient Descent](#gradient-descent)
- [Week 2: Multivariate Linear Regression](#week-2-multivariate-linear-regression)
  - [Learning outcomes](#learning-outcomes)
  - [Linear regression with multiple variables](#linear-regression-with-multiple-variables)
  - [Gradient Descent for multiple variables](#gradient-descent-for-multiple-variables)
  - [Feature Scaling](#feature-scaling)
  - [How to adjust learning rate?](#how-to-adjust-learning-rate)
  - [What is feature combination?](#what-is-feature-combination)
  - [What is polynomial regression?](#what-is-polynomial-regression)
- [Week 2: Computing Parameters Analytically](#week-2-computing-parameters-analytically)
  - [Learning Outcomes](#learning-outcomes-1)
  - [Why do we need normal equation?](#why-do-we-need-normal-equation)
  - [The math behind normal equation](#the-math-behind-normal-equation)
  - [Normal Equation vs Gradient Descent](#normal-equation-vs-gradient-descent)
  - [Causes for Normal Equation noninvertibility](#causes-for-normal-equation-noninvertibility)


## Week 1
### Learning Outcomes
1. What is (1) machine learning, (2) supervised learning and types of (2), (3) unsupervised learning and types of (3)
2. What is the cost function?
3. What is gradient descent?

### Machine learning
The field of study that gives computers the ability to learn without being explicitly programmed.
1. **Supervised Learning**: "Know the right answers.

    - **Regression**: Predict results within a _continuous output_ 
    - **Classification**: Map input variables to a _discrete output_

2. **Unsupervised Learning**: No idea what our results should look like
    - **Clustering**

### Cost Function
Refer to post on gradient descent [here](./gradient-descent.html). The below is the 0.5 * Mean-Squared-Error (MSE) cost function.

$$
J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}
$$

Vectorised form:

$$
\begin{array}{l}{\qquad J(\theta)=\frac{1}{2 m}(X \theta-\vec{y})^{T}(X \theta-\vec{y})} \\ {\text { where }} \\ {\qquad X=\left[\begin{array}{c}{-\left(x^{(1)}\right)^{T}-} \\ {-\left(x^{(2)}\right)^{T}-} \\ {\vdots} \\ {-\left(x^{(m)}\right)^{T}-}\end{array}\right] \quad \vec{y}=\left[\begin{array}{c}{y^{(1)}} \\ {\vdots} \\ {y^{(m)}}\end{array}\right]}\end{array}
$$

> Note: cost function is a function of the model parameters \$ h_\theta \$ while hypothesis function is a function of the variables \$ x \$...

### Gradient Descent
Refer to post on gradient descent [here](./gradient-descent.html). 

## Week 2: Multivariate Linear Regression
### Learning outcomes
1. What is multivariate linear regression? 
2. Gradient descent for multiple variables?
2. Feature scaling: why and how?
3. How to adjust the learning rate?
4. What is feature combination?
5. What is polynomial regression?

### Linear regression with multiple variables

Vectorised form:

$$
h_{\theta}(x)=\left[\begin{array}{llll}{\theta_{0}} & {\theta_{1}} & {\ldots} & {\theta_{n}}\end{array}\right]\left[\begin{array}{c}{x_{0}} \\ {x_{1}} \\ {\vdots} \\ {x_{n}}\end{array}\right]=\theta^{T} x
$$

### Gradient Descent for multiple variables
> Similar to gradient descent for single variable.
$$
\begin{array}{l}{\text { repeat until convergence: }\{} \\ {\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{j}^{(i)} \quad \text { for } j:=0 \ldots n} \\ {\}}\end{array}
$$


Vectorised form of Gradient Descent for linear regression (derivation [here](./gradient-descent-linear-regression.html)):

$$
\theta:=\theta-\frac{\alpha}{m} X^{T}(X \theta-\vec{y})
$$


### Feature Scaling
Refer to post on feature scaling [here](./feature-scaling.html)

### How to adjust learning rate?
1. **Debugging gradient descent**: Plotting a graph of the cost function over no. of iterations. If the cost function ever increases, the learning rate is likely to be too high.
> A high learning rate might cause divergence.
2. **Automatic convergence test**: Declare convergence if the cost function decreases by less than E in one iteration, where E is some small value such as 10^-3. 

### What is feature combination?
Eg \$ x_3 = x_1 * x_2 \$

### What is polynomial regression?
Change the hypothesis function to a polynomial equation that better models the data (ie lower cost function).

$$
Square root: h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} \sqrt{x_{1}}
$$

$$
Cubic: h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{1}^{2}+\theta_{3} x_{1}^{3}
$$

## Week 2: Computing Parameters Analytically
### Learning Outcomes
1. Normal equation: why and what?
2. Normal equation vs Gradient Descent
3. Causes for normal equation non-invertability

### Why do we need normal equation? 
**To obtain the optimum parameters _analytically_.** In other words, the parameters are computed explicitly, instead of being estimated iteratively in gradient descent.

### The math behind normal equation
Recall that parameters are _optimum_ when the cost function is minimum. Thus, we are interested in the value of the parameters at the minimum points of the cost function. _How do we obtain this value then?_

In \$ R_2 \$, we do so by first setting the derivative of the function to be 0 (to obtain the points of inflexion), then solve for the value of the parameter at the minimum point.

Similarly in \$ R_3 \$, we first set the _partial derivative_ with respect to \$ \theta_0 \$ to be 0, then we solve for \$ \theta_0 \$. Repeat iteratively till \$ \theta_n \$.

The result will arrive to the **normal equation formula**:

$$
\theta=\left(X^{T} X\right)^{-1} X^{T} y
$$

> There is **no need** to do feature scaling with normal equation as the purpose of feature scaling was to speed up gradient descent, which is a different method of obtaining the parameters.

### Normal Equation vs Gradient Descent

![Normal Equation vs Gradient Descent](/assets/img/comparison.jpg)

<!-- TODO: Normal equation only for linear regression? -->

### Causes for Normal Equation noninvertibility
If \$ X^{T} X \$ is **noninvertible**, the common causes include
1. **Redundant features:** Features are closely related, thus they are linearly dependent (lie on the same span), hence the matrix containing these vectors results in a linear transformation that squishes input vectors to the single dimension (ie determinant is zero), thus its a noninvertible matrix.
2. **Too many features**. Delete some features or use regularisation

Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning)