---
layout: post
title: "Learning Curves in Machine Learning"
author: "Larry Law"
categories: journal
tags: [Machine Learning]
image: machine-learning.jpg
---
<div align="center">
    <i>"Only half of programming is coding. The other half is debugging." - Unknown</i>
</div>

<!-- omit in toc -->
## Learning outcomes
- [Purpose of Learning Curves](#purpose-of-learning-curves)
- [Definition of Learning Curves](#definition-of-learning-curves)
- [Interpreting Learning Curves](#interpreting-learning-curves)
- [Implementation of Learning Curves in MatLab/Octave](#implementation-of-learning-curves-in-matlaboctave)

## Purpose of Learning Curves
Learning curves diagnose **underfitting/high bias** (left graph) and **overfitting/high variance** (right graph) problems.

![fitting](/assets/img/fitting.jpg)

> A model that underfits are biased for its assumption despite the training data saying otherwise.

> In statistics, variance can be interpreted as how far the data are spread out. A model that overfits make predictions on new data that are very far off the expected value, thus it has high variance.

## Definition of Learning Curves
Let _N_ denote the training set size, and _n_ denote the number of training data used in _N_ (thus _n < N_). Learning curve plots the training error and cross validation error over the training set size _N_. 

More concretely, it is a function of _n_. But what does this function do? This function first trains the parameters given _n_ training samples, and outputs (1) **\$ J_{train}(\Theta) \$**: the cost function of these parameters (y-coordinate) on the _training set of size n_ and (2) **\$ J_{CV}(\Theta) \$**: the cost function of these parameters on the _cross validation set of size N_.

![Learning Curve of High Bias](/assets/img/learning-curve-underfit.png) 

Note that we used _n_ for the \$ J_{train}(\Theta) \$ but _N_ for \$ J_{CV}(\Theta) \$. This is because the learning curve seeks to compare between the cost function that the model was trained on (which is over _n_ training samples) with the cost function that shows the accurate cost of the current model, (which is over _N_ cross validation samples).

> Example cost function (MSE) to illustrate the influence of _n_ on the cost function.

$$
J(\theta)=\frac{1}{2n} \sum_{i=1}^{n}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 n} \sum_{i=1}^{n}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}
$$

## Interpreting Learning Curves
![fitting](/assets/img/fitting.jpg)
![Learning Curve of High Bias](/assets/img/learning-curve-underfit.png) 

Learning Curve of model with _High Bias_. In particular, note that the **training error and test error converges.** Consider the underfitting example on the left. When _N_ is high, the model is not able to optimise the line much more as it is trying to fit a linear equation to a data points of quadratic/log nature. Since the model is not changing much, both the training and test error stagnates. Consequently, **collecting more training data** will not improve a model with high bias.

![Learning Curve of High Variance](/assets/img/learning-curve-overfit.png)

Learning Curve of model with _High Variance_. In particular, note that **test error decreases as N increases**. Consider the overfitting example on the right. With more data points, the overfitted model forces the polynomial equation to fit the data points. While it gets harder to fit the equation (thus training error increases), the equation also gets closer to the correct structure of the data points (thus test error decreases). Consequently, **collecting more training data** will improve a model with high variance.

## Implementation of Learning Curves in MatLab/Octave
```m
function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
...
for i = 1:m
    % Training on i training samples to obtain parameters.
    X_train_i = X(1:i, :);
    y_train_i = y(1:i);
    [theta] = trainLinearReg(X_train_i, y_train_i, lambda); 

    % Compute and store J_train and J_val.
    [error_train(i)] = linearRegCostFunction(X_train_i, y_train_i, theta, 0);

    [error_val(i)] = linearRegCostFunction(Xval, yval, theta, 0); % Note that it over the entire cross validation set.
end
```

Now simply plot `error_train` and `error_val` over `m` :).


<!-- omit in toc -->
## Credits
Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning).
