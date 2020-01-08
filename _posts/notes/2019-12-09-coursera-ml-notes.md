---
layout: post
title: "Coursera, Stanford: Machine Learning Notes"
author: "Larry Law"
categories: notes
image: machine-learning.jpg
hidden: true
---
Lecturer: Professor Andrew Ng <br>
Course available [here](https://www.coursera.org/learn/machine-learning).
Source code available [here](https://github.com/larrylawl/machine-learning-coursera-stanford)

<!-- omit in toc -->
## Table of Contents
- [Week 1](#week-1)
  - [Learning Outcomes](#learning-outcomes)
  - [Machine learning](#machine-learning)
  - [Cost Function](#cost-function)
  - [Gradient Descent](#gradient-descent)
- [Week 2: Multivariate Linear Regression](#week-2-multivariate-linear-regression)
  - [Learning outcomes](#learning-outcomes)
  - [Denoting multiple features](#denoting-multiple-features)
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
- [Week 3: Classification and Representation](#week-3-classification-and-representation)
  - [Learning Outcomes](#learning-outcomes-2)
  - [Sigmoid/Logistic Function: why and what?](#sigmoidlogistic-function-why-and-what)
  - [Decision boundary: what](#decision-boundary-what)
  - [Decision boundary: how to compute](#decision-boundary-how-to-compute)
- [Week 3: Logistic Regression Model](#week-3-logistic-regression-model)
  - [Learning Outcomes](#learning-outcomes-3)
  - [What is the Cost Function of the Logistic Regression Model](#what-is-the-cost-function-of-the-logistic-regression-model)
  - [Gradient Descent of the updated Cost Function](#gradient-descent-of-the-updated-cost-function)
  - [Advanced Optimisation: why and what](#advanced-optimisation-why-and-what)
  - [Advanced Optimisation: how](#advanced-optimisation-how)
- [Week 3: Multiclass Classification](#week-3-multiclass-classification)
  - [Learning Outcomes](#learning-outcomes-4)
  - [What is multiclass classification? How do we implement it?](#what-is-multiclass-classification-how-do-we-implement-it)
- [Week 3: Solving the Problem of Overfitting](#week-3-solving-the-problem-of-overfitting)
  - [Learning Outcomes](#learning-outcomes-5)
  - [Understanding Underfitting and Overfitting](#understanding-underfitting-and-overfitting)
  - [Implementation for Regularisation](#implementation-for-regularisation)
  - [The Tradeoff between Underfitting and Overfitting](#the-tradeoff-between-underfitting-and-overfitting)
  - [Regularised Linear Regression](#regularised-linear-regression)
  - [Regularised Logistic Regression](#regularised-logistic-regression)
- [Week 4: Neural Networks - Representation](#week-4-neural-networks---representation)
  - [Learning Outcomes](#learning-outcomes-6)
  - [Motivation: Non-linear hypotheses](#motivation-non-linear-hypotheses)
  - [Neurons and the Brain](#neurons-and-the-brain)
  - [Intuition for Neural Networks](#intuition-for-neural-networks)
  - [Model Representation](#model-representation)
  - [Multiclass Classification](#multiclass-classification)
- [Week 5: Neural Networks - Learning](#week-5-neural-networks---learning)
  - [Learning Outcomes](#learning-outcomes-7)
  - [Cost Function](#cost-function-1)
  - [Backpropagation: Intuition and Calculus](#backpropagation-intuition-and-calculus)
  - [Backpropagation: Algorithm](#backpropagation-algorithm)
  - [Gradient Checking](#gradient-checking)
  - [Random Initialisation](#random-initialisation)
- [Week 6: Evaluating a Learning Algorithm](#week-6-evaluating-a-learning-algorithm)
  - [Model Selection and Train/Valiation/Test sets](#model-selection-and-trainvaliationtest-sets)
- [Week 6: Bias and Variance](#week-6-bias-and-variance)
  - [Learning Outcomes](#learning-outcomes-8)
  - [Degree of polynomial and Bias/Variance](#degree-of-polynomial-and-biasvariance)
  - [Regularisation and Bias/Variance](#regularisation-and-biasvariance)
  - [Learning Curves](#learning-curves)
  - [Deciding What To Do Next Summary](#deciding-what-to-do-next-summary)
- [Week 6: Handling Skewed Data and Using Large Datasets](#week-6-handling-skewed-data-and-using-large-datasets)
  - [Learning Outcomes](#learning-outcomes-9)
  - [False Positives and Negatives](#false-positives-and-negatives)
  - [Error Metrics for Skewed Classes: Precision, Recall, F Score](#error-metrics-for-skewed-classes-precision-recall-f-score)
  - [Large Data Rationale](#large-data-rationale)
- [Week 7: Large Margin Classification](#week-7-large-margin-classification)
  - [Hypothesis Function](#hypothesis-function)
  - [Large Margin](#large-margin)
- [Week 7: Kernels](#week-7-kernels)
  - [Learning Outcomes](#learning-outcomes-10)
  - [What are Kernals: Similarity Functions](#what-are-kernals-similarity-functions)
  - [Choosing Landmarks](#choosing-landmarks)
- [Week 7: Using an SVM](#week-7-using-an-svm)
  - [Learning Outcomes](#learning-outcomes-11)
  - [SVM parameters](#svm-parameters)
  - [Multi-class classification](#multi-class-classification)
  - [Logistic vs SVMs](#logistic-vs-svms)
- [Week 8: Clustering](#week-8-clustering)
  - [Learning Outcomes](#learning-outcomes-12)
  - [Unsupervised Learning: K-Means Algorithm](#unsupervised-learning-k-means-algorithm)
  - [Optimisation Objective](#optimisation-objective)
  - [Random Initialisation](#random-initialisation-1)
  - [Choosing the number of clusters](#choosing-the-number-of-clusters)
- [Week 8: Principal Component Analysis](#week-8-principal-component-analysis)
  - [Learning Outcomes](#learning-outcomes-13)
  - [Motivations: Data Compression and Data Visualisation](#motivations-data-compression-and-data-visualisation)
  - [Principal Component Analysis Algorithm](#principal-component-analysis-algorithm)
  - [Reconstruction from Compressed representation](#reconstruction-from-compressed-representation)
  - [PCA vs Linear Regression](#pca-vs-linear-regression)
  - [Choosing the Number of Principal Components](#choosing-the-number-of-principal-components)
  - [Advice for applying PCA](#advice-for-applying-pca)
- [Week 9: Anomaly Detection](#week-9-anomaly-detection)
  - [Learning Outcomes](#learning-outcomes-14)
  - [Anomaly Detection Algorithm](#anomaly-detection-algorithm)
  - [Algorithm Evaluation](#algorithm-evaluation)
  - [Anomaly detecion vs Supervised Learning](#anomaly-detecion-vs-supervised-learning)
  - [Choosing what features to use](#choosing-what-features-to-use)
  - [Anomaly detection using Multivariate Gaussian Distribution](#anomaly-detection-using-multivariate-gaussian-distribution)
- [Week 9: Recommender Systems](#week-9-recommender-systems)
  - [Learning Outcomes](#learning-outcomes-15)
  - [Notations and Problem Formulation](#notations-and-problem-formulation)
  - [Optimisation Objective](#optimisation-objective-1)
  - [Collaborative Filtering Algorithm](#collaborative-filtering-algorithm)
  - [Finding Related Movies](#finding-related-movies)
  - [Mean Normalisation](#mean-normalisation)
- [Week 10: Large Scale Machine Learning](#week-10-large-scale-machine-learning)
  - [Learning Outcomes](#learning-outcomes-16)
  - [Motivation](#motivation)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
  - [Stochastic Gradient Descent Convergence](#stochastic-gradient-descent-convergence)
  - [Online Learning](#online-learning)
  - [Map-Reduce and Data Parallelism](#map-reduce-and-data-parallelism)


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
<!-- TODO: Add in convex vs non-convex cost function, cost functions of diff type of machine learning algo (MSE for linear regression, x for logistic etc) -->
The cost function below is the 0.5 * Mean-Squared-Error (MSE) cost function.

$$
J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}
$$

Vectorised form:

$$
\begin{array}{l}{\qquad J(\theta)=\frac{1}{2 m}(X \theta-\vec{y})^{T}(X \theta-\vec{y})} \\ {\text { where }} \\ {\qquad X=\left[\begin{array}{c}{-\left(x^{(1)}\right)^{T}-} \\ {-\left(x^{(2)}\right)^{T}-} \\ {\vdots} \\ {-\left(x^{(m)}\right)^{T}-}\end{array}\right] \quad \vec{y}=\left[\begin{array}{c}{y^{(1)}} \\ {\vdots} \\ {y^{(m)}}\end{array}\right]}\end{array}
$$

> Note: cost function is a function of the model parameters \$ h_\theta \$ while hypothesis function is a function of the variables \$ x \$...

### Gradient Descent
Refer to article on gradient descent [here](/articles/gradient-descent.html). 

## Week 2: Multivariate Linear Regression
### Learning outcomes
1. What is multivariate linear regression? 
2. Gradient descent for multiple variables?
2. Feature scaling: why and how?
3. How to adjust the learning rate?
4. What is feature combination?
5. What is polynomial regression?

### Denoting multiple features

$$
x^{i}_j \text{= value of feature j in the ith training example} \\

x^{i} \text{= input features of the ith training example}
$$

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


Vectorised form of Gradient Descent for linear regression (derivation [here](/articles/gradient-descent-linear-regression.html)):

$$
\theta:=\theta-\frac{\alpha}{m} X^{T}(X \theta-\vec{y})
$$


### Feature Scaling
Refer to post on feature scaling [here](/articles/feature-scaling.html)

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
\theta=\left(X^{T} X\right)^{-1} X^{T} y, where \\

X_{m\times(n+1)}=\left[\begin{array}{c}{1\left(x^{(1)}\right)^{\top}} \\ {1\left(x^{(n)}\right)^{\top}} \\ {\vdots} \\ {1\left(x^{(n)}\right)^{\top}}\end{array}\right]

$$

> There is **no need** to do feature scaling with normal equation as the purpose of feature scaling was to speed up gradient descent, which is a different method of obtaining the parameters.

> Note that \$ (X^{\top}X) \$ is non-invertible if m < n. Proof as follows:

1. rank(X) ≤ min(m, n+1)
2. rank(\$ X^{\top}X \$) = rank(X) (proof [here](https://math.stackexchange.com/questions/349738/prove-operatornamerankata-operatornameranka-for-any-a-in-m-m-times-n))
3. Since m < n, rank(\$ X^{\top}X \$) ≤ m ≤ n + 1.
4. rank(\$ X^{\top}X \$) = n + 1 for it to be invertible.
5. Hence \$ X^{\top}X \$ is non-invertible.

### Normal Equation vs Gradient Descent

![Normal Equation vs Gradient Descent](/assets/img/2019-12-09-coursera-ml-notes/comparison.jpg)

### Causes for Normal Equation noninvertibility
If \$ X^{T} X \$ is **noninvertible**, the common causes include
1. **Redundant features:** Features are closely related, thus they are linearly dependent (lie on the same span), hence the matrix containing these vectors results in a linear transformation that squishes input vectors to the single dimension (ie determinant is zero), thus its a noninvertible matrix.
2. **Too many features**. Delete some features or use regularisation

## Week 3: Classification and Representation
### Learning Outcomes
1. Sigmoid/Logistic function: why and what?
2. Decision boundary: why and how to compute?

### Sigmoid/Logistic Function: why and what?
We use the Logistic Function for **classification problems**. For now, let's focus on _binary_ classification problems (ie output is {0, 1}). 

The hypothesis function of the logistic function _outputs_ the _probability_ that our output is 1 conditioned on our input data (ie \$ x \$), parameterised by our model's parameter (ie \$ ;\theta \$).

$$
h_\theta(x) = P(y = 1|x; \theta)
$$
> `;` denote parameterised

Since the logistic regression is used for the binary classification problem, we need to _translate_ the output of the hypothesis function from (0, 1) to {0, 1}.

$$
h_\theta(x) ≥ 0.5 \rightarrow y = 1 \\
h_\theta(x) < 0.5 \rightarrow y = 0
$$

Vectorised form of this translation
```m
function p = predict(theta, X)
...
p = sigmoid(X * theta) >= 0.5 % size = [m, 1]
```

Internally, this hypothesis function is defined as such:

$$
h_\theta(x) = g(\theta^{\top}x) = g(z) = \frac{1}{1 + e^{-z}}
$$

Vectorised form of the hypothesis function
```m
function g = sigmoid(z)
...
g = 1 ./ (1 .+ e .^ -z) % size = [m, 1]
```

Graphically, it looks like this:
![logistic function](/assets/img/2019-12-09-coursera-ml-notes/logistic.jpg)
Note certain properties of it
1. **\$ h_\theta(x) \in (1, 0) \$.** This makes sense as the hypothesis function is a probability.
2. **\$ If x = 0, h_\theta(x) = 0.5 \$.** Intuitively, if the input data is null, the model will not have any information to make a prediction, thus the probability of the binary classification will be 0.5

### Decision boundary: what
The decision boundary is the **line** that partitions y (ie. {0, 1}). 

![decision boundary](/assets/img/2019-12-09-coursera-ml-notes/decision-boundary-fit.jpg)

### Decision boundary: how to compute
Let us start with an example.

We are interested in the inequality \$ \theta^{\top}x ≥ 0 \$ as it partitions the output to {0, 1}. The heuristic explaining why it partitions is shown below:

$$
\theta^{\top}x ≥ 0 \rightarrow g(\theta^{\top}x) ≥ 0.5 \rightarrow y = 1
$$

An example will clear things up. 

$$
\begin{array}{l}{\theta=\left[\begin{array}{c}{5} \\ {-1} \\ {0}\end{array}\right]} \\ {y=1 \text { if } 5+(-1) x_{1}+0 x_{2} \geq 0} \\ {5-x_{1} \geq 0} \\ {-x_{1} \geq-5} \\ {x_{1} \leq 5}\end{array}
$$

This inequality is useful as it tells us both the _equation of the decision boundary_ (ie \$ x_1 = 5 \$) and the _how the training examples were partitioned_ (ie \$ x_1 ≤ 5 \$ will be classified as 1).

> Notice how the boundary line is not dependent on \$ x_2 \$

> Notice that the decision boundary is a linear equation. Consequently, logistic regression only works well with a dataset of structure which is linearly separable. If a more complex decision boundary is needed, you should consider using neural networks.

## Week 3: Logistic Regression Model
### Learning Outcomes
1. What is the Cost Function of the Logistic Regression Model? What is the Gradient Descent of it?
2. Advanced optimisation: why, what and how?

### What is the Cost Function of the Logistic Regression Model

$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$

Vectorised form

```m
g = sigmoid(X * theta); % size = [m, 1]
J = (1 / m) * (-y' * log(g) - (1 - y') * log(1 - g));
```

Plotting the graph of the Cost Function,
![cost function graph](/assets/img/2019-12-09-coursera-ml-notes/cost-function-graph.jpg) 


Note the following properties
1. If y = 1 and the model predicts y = 0 (ie \$ h_\theta(x) \rightarrow 0 \$), the cost tends to infinity. Likewise for the opposite case (if y = 0 and the model predicts y = 1).
2. \$ {\operatorname{cost}\left(h_{\theta}(x), y\right)=0 \text { if } h_{\theta}(x)=y} \$

### Gradient Descent of the updated Cost Function
Exactly the same as linear regression (can be shown through calculus).

$$
{\theta=\theta-\alpha \delta} = {\theta-\frac{\alpha}{m} X^{T}(g(X \theta)-\vec{y})}\\
$$

### Advanced Optimisation: why and what
They **learn the model's parameters more quickly wo learning rate**. Examples include "Conjugate gradient", "BFGS", and "L-BFGS".

### Advanced Optimisation: how
Getting `costFunction`
```m
function [J, grad] = costFunction(theta)
g = sigmoid(X * theta);
...
J = (1 / m) * (-y' * log(g) - (1 - y') * log(1 - g));
grad = (1 / m) * X' * (g - y); % Partial derivative term
end
```
Using `fminunc`
```m
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```

## Week 3: Multiclass Classification
### Learning Outcomes
1. What is multiclass classification? How do we implement it?

### What is multiclass classification? How do we implement it?
Classifying data in > 2 categories (ie y = {0, 1, ... , n}). We do so by adopting the **one-vs-all** algorithm.

$$
\begin{aligned} y & \in\{0,1 \ldots n\} \\ h_{\theta}^{(0)}(x) &=P(y=0 | x ; \theta) \\ h_{\theta}^{(1)}(x) &=P(y=1 | x ; \theta) \\ \cdots & \\ h_{\theta}^{(n)}(x) &=P(y=n | x ; \theta) \\ \text { prediction } &=\max \left(h_{\theta}^{(i)}(x)\right) \\ & \end{aligned}
$$

To implement this, first we need to learn the parameters for each of the hypothesis functions.
```m
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
...
    for c = 1:num_labels
        initial_theta = zeros(n + 1, 1);

        % Set options for fmincg
        options = optimset('GradObj', 'on', 'MaxIter', 50);

        % Run fmincg to obtain the optimal theta
        % This function will return theta and the cost 
        [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
        all_theta(c, :) = theta; 
    endfor
```

> Use `fmincg` instead of `fminunc` as the former is more efficient when dealing with more parameters.

Next, we run the algorithm shown above.

```m
function p = predictOneVsAll(all_theta, X)
...
z = X * all_theta'; % size = [m, k]
g = sigmoid(z);
[max_values, indices] = max(g, [], 2); % size = [m, 1]
p = indices;
```
> Note use of `max` function.

![multiclass](/assets/img/2019-12-09-coursera-ml-notes/multiclass.jpg)

## Week 3: Solving the Problem of Overfitting
### Learning Outcomes
1. Understanding Underfitting and Overfitting
2. Implementing Regularisation and understand the tradeoff between Underfitting and Overfitting
3. Regularized linear regression
4. Regularized logistic regression


### Understanding Underfitting and Overfitting
![fitting](/assets/img/2019-12-09-coursera-ml-notes/fitting.jpg)

Underfitting (left) is when the hypothesis function, \$ h_\theta{x} \$, maps poorly to the trend of the data. Notice how the \$ h_\theta{x} \$ is a linear function while the data is quadratic (mid). Underfitting is usually caused by a function that is too simple.

Overfitting (right) is when the \$ h_\theta{x} \$ fits the available data but does not generalise well to predict new data. Notice how \$ h_\theta{x} \$ fits the test data well, but does not have any structure that shows that it can predict new data equally well.

To address overfitting, we can either (1) Reduce number of features or (2) implement **regularisation**.

### Implementation for Regularisation

$$
J(\theta)= E(\theta) + \frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}, where \\
E(\theta) = \text{error function}
$$

> Regularisation term starts from j = 1, not 0. We don't penalise \$ tetha_0 \$ as the term was introduced for neater notation.

> We regulate all parameters as we are not able to predict which parameter is more important than the other.

> Regularisation term is the euclidean norm. It is also known as the L2 regularisation, and weight decay. It's called the latter as in the gradient descent step, we are subtracting a very small multiple of W. 

$$
W^{l} := W^{l} - \frac{\alpha \lambda}{m}W^{l} - \alpha(\text{from backprop})
$$

### The Tradeoff between Underfitting and Overfitting

Recall that the error function (1st term) measures how different our model's prediction is from the expected value. Optimising the error function allows us to fit the training sample better at the risk of overfitting.

Regularisation (2nd term), on the other hand, reduces the magnitude of the parameters, thus smoothing out the hypothesis function, and _reduce overfitting_ at the risk of _underfitting_.

> While it reduces overfitting, implementing regularisation does not always mean the model will predict well for new training samples; too much regularisation will cause underfitting, which will lead to poorer prediction.

Thus the new cost function is the summation of both the error function and regularisation, as a better attempt to optimise the tradeoff between underfitting and overfitting.

### Regularised Linear Regression
Gradient Descent:
$$
\begin{array}{l}{\text { Repeat }\{} \\ {\qquad \begin{array}{l}{\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)}} \\ {\theta_{j}:=\theta_{j}-\alpha\left[\left(\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}\right)+\frac{\lambda}{m} \theta_{j}\right]}\\ \}\end{array}} \\ \end{array}
$$

Rearranging...
$$
\theta_{j}:=\theta_{j}\left(1-\alpha \frac{\lambda}{m}\right)-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$

> Note that \$ (1-\alpha \frac{\lambda}{m} \$ < 1. Intuitively you can see it as reducing the value of \$ \theta_j \$ by some amount on every update, which is the purpose of regularisation.

Normal Equation

$$
\begin{array}{l}{\theta=\left(X^{T} X+\lambda \cdot L\right)^{-1} X^{T} y} \\ {\text { where } L=\left[\begin{array}{cccc}{0} \\ {} & {1} \\ {} & {} & {1} \\ {} & {} & {} & {1}\end{array}\right]}\end{array}
$$
> Adding the term λ⋅L, then the matrix becomes invertible.

### Regularised Logistic Regression
$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}
$$

```m
function [J, grad] = costFunctionReg(theta, X, y, lambda)
...
g = sigmoid(X * theta);
e = (1 / m) * (-y' * log(g) - (1 - y') * log(1 - g)); % error term
newTheta = theta;
newTheta(1,1) = 0;
r = (lambda / (2 * m)) * (newTheta' * newTheta);
J = e + r;
grad = (1 / m) * X' * (g - y) + (lambda / m) * newTheta;
```

## Week 4: Neural Networks - Representation
### Learning Outcomes
1. Motivation: Non-linear hypotheses
2. Neurons and the Brain
3. Intuition for Neural Networks
4. Model Representation
5. Multiclass Classification

### Motivation: Non-linear hypotheses
It is expensive for logistic regression to add more polynomial features.
> 1. Represent all quadratic terms: O(\$ n^2 \$) (sum of arithmetic sequence)
> 2. Represent all cubic terms: O(\$ n^3 \$)

Neural networks are a faster way to represent non-linear hypothesis.

### Neurons and the Brain
Neural Network's initial purpose was to build learning systems, thus they modeled after the most amazing one - **human brains**.

Surprisingly, human brains have _one learning algorithm_ (instead of many distinct ones for each functionality). Thus it'll be valuable to model how the brain learns.

![neurons](/assets/img/2019-12-09-coursera-ml-notes/neurons.jpg)
_Dendrites_ (inputs) take in electrical inputs and channel them to _axons_ (outputs).

### Intuition for Neural Networks
Every additional layer allows the network to compute slightly more complex functions. Thus neural networks are able to compute complicated functions.

Consider this example of predicting digits (credits: 3B1B's video explaining neural networks [here](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). Each additional layer builts upon the previous layer in order to compute increasingly complicated functions.

|        | Layer 1 | Layer 2 | Layer 3  | Layer 4 |
|--------|---------|---------|----------|---------|
| Output | Pixels  | Edges   | Patterns | Numbers |

![neural networks intuition](/assets/img/2019-12-09-coursera-ml-notes/neural-networks-intuition.png)


### Model Representation
Notations
$$
\begin{array}{l}{a_{i}^{(j)} = \text{"activation" of unit i in layer j}} \\ 
{\Theta^{(j)} = \text{matrix of weights controlling function mapping from layer j to layer j+1}}\end{array}
$$

![Neural network model](/assets/img/2019-12-09-coursera-ml-notes/neural-network-model.png)

Expanding the terms

$$
\begin{aligned} a_{1}^{(2)} &=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\ a_{2}^{(2)} &=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\ a_{3}^{(2)} &=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\ h_{\Theta}(x)=a_{1}^{(3)}=& g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right) \end{aligned}
$$

`g` is the **logistic activation function,** whose argument is the **linear regression.** However, note the difference in vectorised form between neural networks and logistic regression.

```m
% Logistic Regression
g = sigmoid(X * theta); % size = [m, 1]

% Neural network
a_2 = g;
a_1 = X;
a_2 = sigmoid(a_1 * Theta1'); % size = [m, k], where k denote the number of units in the next layer.
```

<!-- Different theta size -->
This is because every neuron has its own set of parameters (refer to the arguments of `g` in the expanded equation above), thus the size of \$ \Theta \$ is \$ s_{j+1} \times (s_j + 1) \$ while that of \$ \theta \$ in logistic regression is \$ (n + 1) \times 1 \$.

> Thats why neural networks use \$ \Theta \$, not \$ \theta \$. 

Also, remember to add the bias nodes.
```m
a_2 = [ones(m, 1) a_2];
```

### Multiclass Classification
Recall that in multiclass classification, our output y \$ \in \$ {1, 2, ... , n}. In neural networks, y is represented as a matrix. 

$$
y^{(i)}=\left[\begin{array}{l}{1} \\ {0} \\ {0} \\ {0}\end{array}\right],\left[\begin{array}{l}{0} \\ {1} \\ {0} \\ {0}\end{array}\right],\left[\begin{array}{l}{0} \\ {0} \\ {1} \\ {0}\end{array}\right],\left[\begin{array}{l}{0} \\ {0} \\ {0} \\ {1}\end{array}\right]
$$

Neural networks follow the exact same algorithm as logistic regression for multiclass classification. The implementation is as follows:

```m
function p = predict(Theta1, Theta2, X)
...
a_1 = [ones(m, 1) X]; % size = [m = 500, n + 1 = 401]

a_2 = sigmoid(a_1 * Theta1'); % size = [500, 25] 
a_2 = [ones(m, 1) a_2]; % size = [500, 26]

a_3 = sigmoid(a_2 * Theta2'); % size = [500, 10]
[max_values, indices] = max(a_3, [], 2);

p = indices;
```

## Week 5: Neural Networks - Learning
### Learning Outcomes
1. Cost Function
2. Backpropagation: Intuition, Calculus, and Algorithm
3. Gradient Checking
4. Random Initialisation

### Cost Function
$$
J(\Theta)=-\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}\left[y_{k}^{(i)} \log \left(\left(h_{\Theta}\left(x^{(i)}\right)_{k}\right)+\left(1-y_{k}^{(i)}\right) \log \left(1-\left(h_{\Theta}\left(x^{(i)}\right)_{k}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_{l}} \sum_{j=1}^{s_{l+1}}\left(\Theta_{j, i}^{(l)}\right)^{2}\right.
$$

For the double summation,
1. It adds up the logistic regression costs calculated for each unit in the output layer
    - Inner loop: Loops through each unit in the output layer and computes the cost for a particular training sample. Returns cost of the training sample.
    - Outer loop: Loops through all training sample and computes the cost for the training set (containing the samples).

For the Triple Summation
1. the triple sum simply adds up the squares of all the individual Θs in the entire network, except the the bias term. (ie _i_ = 0) 
2. the _i_ in the triple sum does **not** refer to training example i.

Vectorised Implementation:
```m
% Compute a
function computeActivationFunction
    a_1 = [ones(m, 1) X];
    z_2 = a_1 * Theta1';
    a_2 = [ones(m, 1) sigmoid(z_2)];
    z_3 = a_2 * Theta2';
    a_3 = sigmoid(z_3);
endfunction

% Converts y (size = [m, 1]), where each entry in y is in [1, k]
% to label (size = [m, k])
function y_v = convertLabelsToVectors(y)
    % Logical arrays
    y_v = [1:num_labels] == y; 
endfunction

function computeCost
    computeActivationFunction;
    y_v = convertLabelsToVectors(y);

    % cost 
    c_each = -y_v .* log(a_3) - (1 - y_v) .* log(1 - a_3); % size = [m, s_L]
    c_all = (1 / m) * sum(sum(c_each));

    % regularisation term - rmb to not regularise bias terms!
    Theta1_squared = Theta1 .^ 2;
    Theta2_squared = Theta2 .^ 2;
    Theta1_squared_wo_bias = Theta1_squared(:, 2:end);
    Theta2_squared_wo_bias = Theta2_squared(:, 2:end);
    r = (lambda / (2 * m)) * ...
        (sum(sum(Theta1_squared_wo_bias)) + sum(sum(Theta2_squared_wo_bias)));

    % Cost 
    J = c_all + r;
endfunction
```

### Backpropagation: Intuition and Calculus
Refer to my post [here](/articles/calculus-for-backpropagation.html).

### Backpropagation: Algorithm
![Neural Network](/assets/img/2019-12-09-coursera-ml-notes/neural-network-model.png)

![Backpropagation algorithm](/assets/img/2019-12-09-coursera-ml-notes/backpropagation-algo.png)
Implementation Note
- Add bias term (ie 1) for all `a` except layer _L_. 
- Don't compute \$ \delta^{1} \$. Input data should not have error terms associated with them.
- Remove \$ \delta_0^{l} \$, where _l_ refers to any hidden layer. Bias unit of any layer is assumed to be 1 and independent of computation (ie not connected to previous layer), thus it should not have error term associated with it.
- Sanity check that \$ \Delta \$ is of the same dimensions as \$ \Theta \$. \$ \Delta \$ is the partial derivative of the cost function wrt each \$ \Theta \$, thus there is a one-to-one mapping from \$ \Theta \$ to \$ \Delta \$.
- Parameter unrolling

Vectorised Implentation (wo `for-loops`! :):
```m
function backprop 
    % Step 1: Compute a 
    computeActivationFunction;
    
    % Step 2: Compute error terms
    y_v = convertLabelsToVectors(y);
    d_3 = a_3 - y_v; % size = [5000, 10] 

    % Step 3: Delta 2
    d_2 = (d_3 * Theta2)(:, 2: end) .* sigmoidGradient(z_2); % size = [5000, 25]; remember to remove first error term

    % Step 4 and 5: Accumulate gradient and divide by sample size        
    Theta1_grad = (1 / m) .* (d_2' * a_1); % size = [25, 401]
    Theta2_grad = (1 / m) .* (d_3' * a_2); % size = [10, 25]

endfunction

function regularise
    r_1 = (lambda / m) .* (Theta1);
    r_1(:, 1) = 0; % Don't regularise bias term

    r_2 = (lambda / m) .* (Theta2);
    r_2(:, 1) = 0; % Don't regularise bias term

    Theta1_grad = Theta1_grad + r_1;
    Theta2_grad = Theta2_grad + r_2;
endfunction
```
### Gradient Checking

$$
\frac{\partial}{\partial \Theta_{j}} J(\Theta) \approx \frac{J\left(\Theta_{1}, \ldots, \Theta_{j}+\epsilon, \ldots, \Theta_{n}\right)-J\left(\Theta_{1}, \ldots, \Theta_{j}-\epsilon, \ldots, \Theta_{n}\right)}{2 \epsilon}
$$

$$
\text { difference }=\frac{\| g r a d-\text {gradapprox} \|_{2}}{\| \text {grad}\left\|_{2}+\right\| \text {gradaprox} \|_{2}}
$$

1. Numerator: Euclidean Distance between the two gradients
2. Denominator: Normalises the distance as a ratio of the difference and the length of the two gradients (in case any of these vectors are really small or large). 

Compare gradient of approximation with that of backpropagation.
Implementation tips
1. Approximating the cost function is slow (need to run forward propagation). For this reason, we don't run gradient checking at every iteration of descent during training. Just a few times to check if the gradient is correct.
2. Gradient checking works for any hypothesis function (linear, log etc).

Vectorised Implementation
```m
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

### Random Initialisation
Refer to my post [here](/articles/heuristic-for-random-init.html)

Implementation note:
1. Chooe epislon based on the number of units in the network. A good choice is 

$$
\epsilon_{i n i t}=\frac{\sqrt{6}}{\sqrt{L_{i n}+L_{o u t}}}
$$

Vectorised Implementation
```m
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

## Week 6: Evaluating a Learning Algorithm

### Model Selection and Train/Valiation/Test sets
How should we use the Train/Valiation/Test sets?
1. **Optimize the parameters Θ**  using the training set for each polynomial degree.
2. **Find the polynomial degree d with the least error** using the _cross validation set_.
3. **Estimate the generalization error** using the _test set_ with \$ J_{test}(\Theta^{(d)}) \$, (d = theta from polynomial with lower error)

Recommended breakdown:
- Training Set: 60%
- CV Set: 20%
- Test Set: 20%

> By having an extra CV set, the degree of the polynomial d has not been trained using the test set.

## Week 6: Bias and Variance
### Learning Outcomes
1. Degree of polynomial and Bias/Variance
2. Regularisation and Bias/Variance
3. Learning Curves
4. Deciding What To Do Next Summary

### Degree of polynomial and Bias/Variance
![Degree of polynomial and Bias/Variance](/assets/img/2019-12-09-coursera-ml-notes/polynomial-bias-variance.png)

In particular, note that
1. As polynomial degree increases, we shift from the problem of underfitting to overfitting.
2. High bias: \$ J_{train}(\Theta) \$ and \$ J_{CV}(\Theta) \$ will be high.
3. High variance: \$ J_{train}(\Theta) \$ will be low but \$ J_{CV}(\Theta) \$ will be high. This is because the model fit the data too incely, st it does not predict new data well.

### Regularisation and Bias/Variance
As λ increases, we regularise more, thus we shift from the issue of high variance to high bias. How do we get the optimal λ then?

1. Create a list of lambdas (i.e. λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
2. Create a set of models with different degrees or any other variants.
3. Iterate through the λs and for each λ go through all the models to learn some Θ.
4. Compute the cross validation error using the learned Θ (computed with λ) on the \$ J_{CV}(\Theta) \$ without regularization or λ = 0.
5. Select the best combination that produces the lowest error on the cross validation set.
6. Using the best combination Θ and λ, apply it on \$ J_{test}(\Theta) \$ to see if it has a good generalization of the problem.

### Learning Curves
Refer to my post [here](/articles/learning-curves.html)

### Deciding What To Do Next Summary 
1. Getting more training examples: Fixes high variance
2. Trying smaller sets of features: Fixes high variance
3. Adding features: Fixes high bias
4. Adding polynomial features: Fixes high bias
5. Decreasing λ: Fixes high bias
6. Increasing λ: Fixes high variance.

Recommended Approach for ML problems:
1. Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
2. Plot learning curves to decide if more data, more features, etc. are likely to help.
3. Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
   1. Use a single, numerical value to assess algorithm's performance (e.g. error rate).

## Week 6: Handling Skewed Data and Using Large Datasets
### Learning Outcomes
0. False Positives and Negatives
1. Error Metrics for Skewed Classes: Precision, Recall, F Score
2. Large Data Rationale


### False Positives and Negatives
![false positives and negatives](/assets/img/2019-12-09-coursera-ml-notes/false-positive-negative.png)

> **False positive:** Predicted positive incorrectly


### Error Metrics for Skewed Classes: Precision, Recall, F Score

Motivating example: Cancer
Suppose that only 0.5% of patients have cancer. A function that always outputs 0 will have an accuracy of 99.5%, but this model can surely be improved. Are there any other error metrics that evaluate skewed classes better?

1. Accuracy = (true positives + true negatives) / (total examples)
2. Precision = (true positives) / (true positives + false positives)
   1. Precision is how many of the returned hits were true positive i.e. how many of the found were correct hits.
3. Recall = (true positives) / (true positives + false negatives)
   1. Recall is how many of the true positives were recalled (found), i.e. how many of the correct hits were also found.
4. F score = (2 * precision * recall) / (precision + recall)
   1. Allows for a single numeric metric as it combines Precision and Recall. By multiplying it, it also effectively punishes the skewed cases (ie P = 1, R = 0, which can be easily achieved with a function that always outputs 1 and vice versa).

### Large Data Rationale
"It's not who has the best algorithm that wins. It's who has the most data."

![big data](/assets/img/2019-12-09-coursera-ml-notes/big-data.png)

1. Suppose feature _x_ has sufficient information to predict _y_ accurately 
   1. Useful test: Given the input _x_, can a human expert confidently predict _y_?
2. Able to use a _low bias_ algorithm
3. Using a very large training set will resolve issue of low variance
4. Most data wins :)

## Week 7: Large Margin Classification
### Hypothesis Function

$$
h_{\theta}(x)\left\{\begin{array}{ll}{1} & {\text { if } \theta^{\top} x \geqslant 0} \\ {0} & {\text { otherwise }}\end{array}\right.
$$

![SVM](/assets/img/2019-12-09-coursera-ml-notes/svm.png)

_C_ is a penalisation parameter that have the opposite role of the parameter \$ \lambda \$. Concretely, when C decreases, \$ \lambda \$ increases, the regularisation term increases, hence it mitigates overfitting.

Just as how multiplying by a constant does not change the x-coordinate of the minimum point of a graph, multiplying by constant _C_ does not change the theta where the cost funtion is minimm.

<!-- Not very convinced with the explanatin -->
When y = 1 and SVM hypothesis = 1, that means we predict correctly and thus the cost should be 0. We use the plot on the left above when y = 1. That's the plot for the input (to the sigmoid function) vs the cost when y = 1. When both y and the SVM hypothesis happen to be 1 (meaning the cost is 0), that can only be achieved when the input is to the right of 0 on the horizontal axis in that plot.


### Large Margin
Refer to my post [here](/articles/math-behind-large-margin-classification.html)

## Week 7: Kernels
### Learning Outcomes
1. What are Kernals: Similarity Functions
2. Choosing Landmarks
3. SVM parameters

### What are Kernals: Similarity Functions
The purpose of Kernels is to plot non-linear decision boundary. 

Kernels are **similarity functions**. Intuitively, given x, a kernel evaluates the similarity of x and the landmark _l_. (I will elaborate on how to choose landmarks later). 

![Support Vector Machine landmarks](/assets/img/2019-12-09-coursera-ml-notes/svm-landmarks.png)

> Kernel chosen here is the Gaussian kernel

But how does the similarity function help us plot non-linear decision boundaries?

This is the revised cost function with the kernel. In particular, note the argument to the _cost_ function. By using feature vector instead of the input x, we can plot non-linear decision boundaries.

$$
\min _{\theta} C \sum_{i=1}^{m} y^{(i)} \operatorname{cost}_{1} \left(\theta^{T} f^{(i)}\right)+\left(1-y^{(i)}\right) \operatorname{cost}_{0}({\theta^{T} f^{(i)})})+{\frac{1}{2} \sum_{j=1}^{m} \theta_{j}^{2}}
$$

Predict \$ y = 1 \$ if \$ \theta^{T} f ≥ 0 \$ 

### Choosing Landmarks

$$
l^{(i)} = x^{(i)}
$$
for 0 ≤ i ≤ n, where n is the dimension of \$ \theta \$. Thus the similarity function for the same points outputs 1.

Intuitively, this is nice because it is saying that my features are basically going to measure how close an example is to one of the things I saw in my training set.

## Week 7: Using an SVM
### Learning Outcomes
0. SVM parameters
1. Multi-class classification
2. Logistic vs SVMs

### SVM parameters
1. **C**: C is a penalisation parameter that have the opposite rote of the parameter \$ \lambda \$. Concretely, when C decreases, \$ \lambda \$ increases, the regularisation term increases, hence it mitigates overfitting.  
2. _(For Gaussian Kernels)_ \$ \sigma^2 \$: Larger sigma --> Similarity function becomes smoother (Feature decreases less quickly) -->  Not so dependent on x1 --> Higher bias --> lower variance (due to bias-variance tradeoff) 
![Gaussian kernel variance](/assets/img/2019-12-09-coursera-ml-notes/gaussian-kernel-variance.png)
3. **Perform feature scaling**: 

![Feature Scaling for SVM](/assets/img/2019-12-09-coursera-ml-notes/svm-feature-scaling.png)

### Multi-class classification
Either
1. Use built-in multiclass classification
2. Use one-vs-all method.
   1. Train K SVMs to get k \$ \theta \$.
   2. Pick class i with largest with \$ \theta^{T} f \$ (since it fits the hypothesis fn better)

### Logistic vs SVMs
Let _n_ = number of features and _m_ = number of training examples

1. If _n_ is large relative to _m_, use logistic regression or SVM with linear Kernel: data points are too little to have a complex decision boundary. 
> SVM with linear kernel have similar performance as logistic regression 
2. If _n_ is small but _m_ is intermediate, use SVM with Gaussian Kernel: Able to model complex boundary
3. If _n_ is small but _m_ is large: Create/add more features, then use logistic regression or SVM without a kernel. This is because SVM with Gaussian kernel might run too slowly on large _m_.

## Week 8: Clustering
### Learning Outcomes
1. Unsupervised Learning: K-Means Algorithm
2. Optimisation Objective
3. Random Initialisation
4. Choosing the number of clusters

### Unsupervised Learning: K-Means Algorithm
![K means algorithm](/assets/img/2019-12-09-coursera-ml-notes/k-means-algo.png)

### Optimisation Objective

$$
J\left(c^{(1)}, \ldots, c^{(m)}, \mu_{1}, \ldots, \mu_{K}\right)=\frac{1}{m} \sum_{i=1}^{m}  {\left\|x^{(i)}-\mu_{c^{(i)}}\right\|^{2}}
$$

> Cost function is also called Distortion

Intuitively, 
1. the cluster assignment step minimises J wrt the centroids, _c_, holding \$ \mu \$ fixed.
2. the moving of centroids minimises J wrt to \$ \mu \$ while holding _c_ fixed.
3. Steps 1 and 2 optimises the cost function J

> Cost function will never decrease. 

### Random Initialisation
1. Randomly pick _K_ training examples.
2. Set centroids, \$ \mu_{i} \$, to these K examples.
3. Repeat steps 1-2 for _n_ times, and pick clustering that gave lowest cost function

### Choosing the number of clusters
1. Elbow method

![Elbow method](/assets/img/2019-12-09-coursera-ml-notes/elbow-method.png)

2. Manually choosing _K_ (ie choosing T-shirt size)

## Week 8: Principal Component Analysis
### Learning Outcomes
1. Motivations: Data Compression and Data Visualisation
2. Principal Component Analysis Algorithm
3. Reconstruction from Compressed representation
4. PCA vs Linear Regression
5. Choosing the Number of Principal Components
6. Advice for applying PCA

### Motivations: Data Compression and Data Visualisation
Reducing dimensions allow for
1. Data Compression 
2. Data Visualisation (ie 2d - x,y axis etc)

### Principal Component Analysis Algorithm
1. Data preprocessing
   1. Mean normalisation and Feature Scaling
2. Algorithm
 
![PCA algorithm](/assets/img/2019-12-09-coursera-ml-notes/pca-algorithm.png)
1. Sigma is the covariance matrix
2. K is the top k eigenvectors of the covariance matrix
3. z is the reduced vector from dimension _n_ to _k_; it best represents x of _n_ dimensions in _k_ dimensions, where _k_ < _n_.

> Proof (here)[http://cs229.stanford.edu/notes/cs229-notes10.pdf] 

> Why is retaining variance equivalent to maximising variance? --> Original variance is without u --> To retain, u need a `u` st when multiplied with the dot product, it gets back the original variance --> Since u is a unit vector, the best u you can obtain is when they lie on the same span, thus cos(0) = 1 --> dot product will be x itself

### Reconstruction from Compressed representation

Recall when reducing,

$$
z = U^{\top}_{reduce}x
$$

Thus to reconstruct _x_, simply

$$
U_{reduce}z = x
$$

### PCA vs Linear Regression
1. Vertical vs Orthogonal distance (thus shortest)
   1. Interpretation: ppd. Formally: Dot product is 0
2. Predicting y vs focusing on a list of features (x1... xn)

### Choosing the Number of Principal Components
Choose _k_ to be the smallest value such that maximum variance is retained.

$$
\frac{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}-x_{a p p r o x}^{(i)}\right\|^{2}}{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}\right\|^{2}} ≤ 0.01\%
$$

It turns out that this is equivalent to

$$
\frac{\sum_{i=1}^{k} S_{i i}}{\sum_{i=1}^{m} S_{i i}} \leqslant 0.99
$$

where S is the diagonal matrix return from `svd`.


### Advice for applying PCA
1. **To prevent overfitting**: While PCA reduces dimension, thus reducing number of features thus reducing overfitting, you should use regularisation instead! PCA, unlike regularisation, loses information as it squishes dimension.
2. **Use raw data first**: Only use PCA if you need the speedup.

## Week 9: Anomaly Detection
### Learning Outcomes
1. Anomaly Detection Algorithm
2. Algorithm Evaluation
3. Anomaly detecion vs Supervised Learning
4. Choosing what features to use
5. Anomaly detection using Multivariate Gaussian Distribution

### Anomaly Detection Algorithm
![Anomaly detection algorithm](/assets/img/2019-12-09-coursera-ml-notes/anomaly-detection-algo.png)

### Algorithm Evaluation
1. Fit model _p(x)_ on training set.
2. On cross validation/test example _x_, predict 

$$
y=\left\{\begin{array}{ll}{1} & {\text { if } p(x) < \epsilon} \text{ (anomaly)}\\ 
{0} & {\text { if } p(x) ≥ \epsilon \text{ (normal)}}\end{array}\right.
$$

3. Evaluate based on `F-score` (not accuracy!). This is because anomalies represent skewed classes.

4. Iterate through values of \$ \epsilon \$ to determine best threshold value.

### Anomaly detecion vs Supervised Learning

|Anomaly Detection  |Supervised Learning  |
|---|---|
|Very small no. of positive examples. Large no. of negative examples.  |Large no. of positive and negative examples  |
|Many different "types" of anomalies. Hard for algo to learn from positive examples as future anomalies may be completely different. |Enough positive examples for algo to get a sense of what positive examples are like  |

Because 1) the mean and sv of positive examples are more consistent and 2) negative examples are rare, anomaly detection systems trains _p(x)_ on the positive examples (ie _y = 0_), and saves the negative examples for cv or test sets.

### Choosing what features to use
1. **Error Analysis:** manually looking at each anomaly that was wrong (ie has p(x) that is comparable w the normal eg). Come up w new feature to handle the anomaly better 
2. **Feature Engineering**
   1. _f(x)_: function of feature itself so as to get a normal distribution. Need to manipulate as your probability distribution _p_ is for gaussian distribution. (Use `histogram` to do so).
   2. _f(x, y)_: function of multiple features so as to capture anomalies better.

### Anomaly detection using Multivariate Gaussian Distribution
Instead of modeling \$ p(x_1), p(x_2),... \$ separately, mv gaussian distribution models \$ p(x_1, x_2, ..., x_n) \$.

1. Fit model _p(x)_ by setting

$$
\mu=\frac{1}{m} \sum_{i=1}^{m} x^{(i)}
$$

$$
\Sigma=\frac{1}{m} \sum_{i=1}^{m}\left(x^{(i)}-\mu\right)\left(x^{(i)}-\mu\right)^{T}
$$

2. Given a new example x, compute

$$
p(x; \mu, \Sigma)=\frac{1}{(2 \pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)
$$

Flag an anomaly if \$ p(x) < \epsilon \$.

If the features are independent of each other (ie the covariance matrix is zero off diagonal), then **the probability distribution of the gaussian distribution of the individual features equals that of the multivariate gaussian distribution.** That is, the two equations below are equal.

$$
p(x) = p(x_1 ; \mu_1; \sigma^2_1) \times p(x_2 ; \mu_2; \sigma^2_2) \times ... \times p(x_n ; \mu_n; \sigma^2_n)
$$

$$
p(x; \mu, \Sigma)=\frac{1}{(2 \pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)
$$

> Explanation: Zero off diagonal --> Each dimension is independent of the other --> joint prob distribution = the multiplication of the individual prob

|Original Model|Multivariate Gaussian  |
|---|---|
|Manually create features to capture anomalies (ie feature engineering) | Automatically captures correlations between features |
|Computationally cheaper |Computationally more expensive (as covariance matrix is nxn, thus calculating inverse will be n^2)  |
|Ok even if m is small  |Must have m > n (rec m ≥ 10n) in order for covariance matrix to be invertible  |

## Week 9: Recommender Systems
### Learning Outcomes
1. Notations and Problem Formulation
2. Optimisation Objective
3. Collaborative Filtering algorithm

### Notations and Problem Formulation
- _r(i, j) = 1_ if user _j_ has rated movie _i_ (0 otherwise)
- \$ y^{(i, j)} \$ = rating by user _j_ on movie _i_ (if defined)
- \$ \theta^{(j)} \$ = parameter vector for user _j_
- \$ x^{(i)} \$ = feature vector for movie _i_
- For user _j_, movie _i_, predicted rating: \$ (\theta^{(j)})^{\top}(x^{(i)}) \$

Objective is recommend the movies with highest predicted ratings to the user.

### Optimisation Objective

$$
J\left(x^{(1)}, \ldots, x^{\left(n_{m}\right)}, \theta^{(1)}, \ldots, \theta^{\left(n_{\omega}\right)}\right)=\frac{1}{2} \sum_{(i, j): r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)}\right)^{2}+\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n}\left(x_{k}^{(i)}\right)^{2}+\frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n}\left(\theta_{k}^{(j)}\right)^{2}
$$

Note that:
1. Do not include bias term (ie _x_ exists in _n_ dimension, not _n+1_).
2. Updates both _x_ and \$ \theta \$ simultaneously.
3. Cost function is the same as linear regression, except that it does not normalise by _1/m_ (which does not shift the minimum point of the cost function).

> Explanation: Normalizing by 1/m helps in regression because you may want to vary the size of the training set, but still have comparable cost values for analysis. Typically this isn't required in the recommender system. So we can remove the 1/m calculation and save a bit of computer processing time.

### Collaborative Filtering Algorithm
![Collaborative Filtering Algorithm](/assets/img/2019-12-09-coursera-ml-notes/collaborative-filtering-algo.png)

> Collaborative Filtering Algorithm is also called the _low rank matrix factorisation_.

> Why does initialising random values work? _(Disclaimer: This is my own rationalisation)_ 

This serves as symmetry breaking (similar to the random initialization of a neural network’s parameters) and ensures the algorithm learns all features (x) that are different from each other.

> How can we make predictions when we have neither _x_ (movie features) nor \$ \theta \$ (user preference)?

Notice that the cost function only computes the cost when there is a true value (ie _r(i, j) = 1_). For _i and j_ which have a true value, the cost function  sums up the squared difference between the predicted value (ie _thetax_) and the true value (ie _y_). Thus, the optimisation of the cost function will be based on minimising the squared difference above, only when there is a true value. After learning the parameters based on _y_, we can now predict the entries which are missing _x_ or \$ \theta \$.

> Why is it called collaborative filtering?
All these users are collaborating to help the system to learn better features. With every user rating some subset within the movies, every user is helping the algo a lil bit to learn the features better.

### Finding Related Movies
Find a movie with small euclidean distance, ie 

$$
||x^{(i)} - x^{(j)}||
$$

### Mean Normalisation
Important to perform (mean) normalisation.
> You can skip the variance if all the movies have the same ratings (ie 1-5)

Suppose not,
1. For entries without prediction, in order to minimise cost function, specifically the regularisation term, \$ \theta = 0 \$.
2. Which makes the prediction (ie \$ (\theta^{(j)})^{\top}(x^{(i)}) \$ = 0)
3. Not useful result

With mean normalisation, you can add the mean to 0, ie \$ (\theta^{(j)})^{\top}(x^{(i)}) + \mu_i \$

All the basis vectors are eigenvectors 

## Week 10: Large Scale Machine Learning
### Learning Outcomes
1. Motivation
2. Stochastic Gradient Descent
3. Mini-batch Gradient Descent
4. Stochastic Gradient Descent Convergence
5. Online Learning
6. Map-Reduce and Data Parallelism

### Motivation
"It's not who has the best algorithm that wins. It's who has the most data." - Finding from research by Banko and Brill, 2001. 

Thus we need an time-complexity efficient way of handling this data.

> Before performing faster gradient descent methods, first check if algo will perform just as well with a smaller subset of the data. Plot a learning curve for a range of values of m and verify that the algorithm has high variance when _m_ is small.

### Stochastic Gradient Descent
![Stochastic Gradient Descent Algorithm](/assets/img/2019-12-09-coursera-ml-notes/stochastic-gradient-descent-algo.png)

The term after alpha is the partial derivative term of the cost function of a single training sample.

$$
\operatorname{cost}\left(\theta,\left(x^{(i)}, y^{(i)}\right)\right)=\frac{1}{2}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$

### Mini-Batch Gradient Descent
1. **Batch Gradient Descent**: Use all _m_ examples in each iteration
2. **Stochastic Gradient Descent**: Use _1_ example in each iteration
3. **Mini-Batch gradient descent**: Use _b_ examples in each iteration

![Mini-batch gradient descent algo](/assets/img/2019-12-09-coursera-ml-notes/minibatch-gradient-descent-algo.png)

Mini-Batch vs Stochastic Gradient Descent
   1. Advantage: leverage vectorisation 
   2. Disadv: Need to find _b_

### Stochastic Gradient Descent Convergence
![Convergence](/assets/img/2019-12-09-coursera-ml-notes/convergence.png)

> Every data point is the cost function averaged above _m_ examples.

Learning rate is typically held constant. We can slowly decrease if we want theta to converge. However, 1) now we have more parameters to deal with and 2) we are usually satisfied so long as its in the region of local min.

$$
\alpha = \frac{\text{const1}}{\text{iterationNumber} + \text{const2}}
$$


### Online Learning
![Online Learning](/assets/img/2019-12-09-coursera-ml-notes/online-learning.png)

Data is continuously thrown away

### Map-Reduce and Data Parallelism
![Map Reduce](/assets/img/2019-12-09-coursera-ml-notes/map-reduce.png)

> `reduce` the partial derivatives to a single sum.