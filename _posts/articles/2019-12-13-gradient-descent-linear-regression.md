---
layout: post
title: "Derivation of vectorised form of Gradient Descent for Linear Regression"
author: "Larry Law"
categories: articles
tags: [Machine Learning]
image: machine-learning.jpg
---
## Learning outcomes
0. Understand purpose of vectorisation
1. Derive the vectorised form of gradient descent for linear regression

## Prerequisites: Gradient Descent
Read more about gradient descent in my post [here](./gradient-descent.html).

## Purpose of Vectorisation
![vectorisation](/assets/img/vectorisation.jpg)

Notice how the vectorised form is...

1. Easier to implement (thus having less bugs)
2. More efficient (since we are calling the optimised linear algebra library)

## From partial derivatives to summation...
Recall that gradient descent is defined as such:

$$
\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}, ..., \theta_{n}\right)
$$

It turns out that the partial derivative term can be simplified as followed:

$$
\begin{aligned} \frac{\partial}{\partial \theta_{j}} J(\theta) &=\frac{\partial}{\partial \theta_{j}} \frac{1}{2}\left(h_{\theta}(x)-y\right)^{2} \\ &=2 \cdot \frac{1}{2}\left(h_{\theta}(x)-y\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(h_{\theta}(x)-y\right) \\ &=\left(h_{\theta}(x)-y\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(\sum_{i=0}^{n} \theta_{i} x_{i}-y\right) \\ &=\left(h_{\theta}(x)-y\right) x_{j} \end{aligned}
$$

Thus our simultaneous update can be written as:

$$
\begin{array}{l}{\text { repeat until convergence: }\{} \\ {\qquad \begin{array}{l}{\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)} \\ {\theta_{1}:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(\left(h_{\theta}\left(x_{i}\right)-y_{i}\right) x_{i}^1\right)} \\ ... \\
\theta_{n}:=\theta_{n}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(\left(h_{\theta}\left(x_{i}\right)-y_{i}\right) x_{i}^n\right) \\
\}
\end{array}}\end{array}
$$
> Remeber to use simultaneous update!

## From summation to vector...
Claim that \$ \theta \$ can be rewritten from summation form (as above) to vector form (as below):

$$
\theta:=\theta-\alpha \delta, where\\
\delta := \frac{1}{m} X^{T}(X \theta-\vec{y})
$$

Let H denote the hypothesis matrix, 

$$
\begin{array}{l}{\dot{H}_{m \times 1}:=\left(\begin{array}{c}{h_{\theta}\left(x^{1}\right)} \\ {h_{\theta}\left(x^{2}\right)} \\ {\vdots} \\ {h_{\theta}\left(x^{m}\right)}\end{array}\right)=X \theta} & 

\end{array}
$$

Let E denote the error matrix,

$$
E_{m \times 1}:=\left(\begin{array}{c}{e_{1}} \\ {e_{2}} \\ {\vdots} \\ {e_{m}}\end{array}\right)=\left(\begin{array}{c}{h_{\theta}\left(x^{1}\right)-y^{1}} \\ {h_{\theta}\left(x^{2}\right)-y^{2}} \\ 
{\vdots} \\
{h_{\theta}\left(x^{m}\right)-y^{m}}\end{array}\right)_{m \times 1}=H-\vec{y}
$$

Let \$ \delta \$ denote the summation term

$$
\begin{aligned} \delta_{j} &:=\frac{1}{n} \sum_{i=1}^{n}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)} \\ &=\frac{1}{m}\left(e_{1} x_{j}^{1}+e_{2} x_{j}^{2}+\ldots+e_{m} x_{j}^{m}\right) \\ &=\frac{1}{m} x_{j}^{\top} E \\ \end{aligned}
$$

$$
\begin{array}{c}{\delta:=\left(\begin{array}{c}{\delta_{0}} \\ {\delta_{1}} \\ {\vdots} \\ {\delta_{n}}\end{array}\right)=\frac{1}{m} X^{\top} E}\end{array} 
$$

$$
{\theta=\theta-\alpha \delta} = {\theta-\frac{\alpha}{m} X^{T}(X \theta-\vec{y})} \text{ (shown)}\\
$$

> Note that this vectorised form applies for **logistic regression too.** Recall that logistic regression has the same gradient descent formula, with the only difference being that the hypothesis function of a logistic regression is a function of the linear regression (ie \$ h_{\theta}(x) = g(\theta^Tx) \$). 
> 
> Thus for logistic regression, we will need to substitute the hypothesis matrix to be a function of the linear regression.

$$
H = g(X\theta)
$$

> Vectorised formula will be as follow:

$$
{\theta=\theta-\alpha \delta} = {\theta-\frac{\alpha}{m} X^{T}(g(X \theta)-\vec{y})}\\
$$

## Credits
Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning)
