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
Source: [here](https://www.coursera.org/learn/machine-learning)

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

![Normal Equation vs Gradient Descent](/assets/img/comparison.jpg)

<!-- TODO: Normal equation only for linear regression? -->

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
![logistic function](/assets/img/logistic.jpg)
Note certain properties of it
1. **\$ h_\theta(x) \in (1, 0) \$.** This makes sense as the hypothesis function is a probability.
2. **\$ If x = 0, h_\theta(x) = 0.5 \$.** Intuitively, if the input data is null, the model will not have any information to make a prediction, thus the probability of the binary classification will be 0.5

### Decision boundary: what
The decision boundary is the **line** that partitions y (ie. {0, 1}). 

![decision boundary](/assets/img/decision-boundary-fit.jpg)

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
![cost function graph](/assets/img/cost-function-graph.jpg) 


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

![multiclass](/assets/img/multiclass.jpg)

## Week 3: Solving the Problem of Overfitting
### Learning Outcomes
1. Understanding Underfitting and Overfitting
2. Implementing Regularisation and understand the tradeoff between Underfitting and Overfitting
3. Regularized linear regression
4. Regularized logistic regression


### Understanding Underfitting and Overfitting
![fitting](/assets/img/fitting.jpg)

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

![neurons](/assets/img/neurons.jpg)
_Dendrites_ (inputs) take in electrical inputs and channel them to _axons_ (outputs).

### Intuition for Neural Networks
Every additional layer allows the network to compute slightly more complex functions. Thus neural networks are able to compute complicated functions.

Consider this example of predicting digits (credits: 3B1B's video explaining neural networks [here](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). Each additional layer builts upon the previous layer in order to compute increasingly complicated functions.

|        | Layer 1 | Layer 2 | Layer 3  | Layer 4 |
|--------|---------|---------|----------|---------|
| Output | Pixels  | Edges   | Patterns | Numbers |

![neural networks intuition](/assets/img/neural-networks-intuition.png)


### Model Representation
Notations
$$
\begin{array}{l}{a_{i}^{(j)} = \text{"activation" of unit i in layer j}} \\ 
{\Theta^{(j)} = \text{matrix of weights controlling function mapping from layer j to layer j+1}}\end{array}
$$

![Neural network model](/assets/img/neural-network-model.png)

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

$$
J(\Theta)=-\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}\left[y_{k}^{(i)} \log \left(\left(h_{\Theta}\left(x^{(i)}\right)_{k}\right)+\left(1-y_{k}^{(i)}\right) \log \left(1-\left(h_{\Theta}\left(x^{(i)}\right)_{k}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_{l}} \sum_{j=1}^{s_{l+1}}\left(\Theta_{j, i}^{(l)}\right)^{2}\right.
$$

Note that
1. The double sum simply adds up the logistic regression costs calculated for each cell in the output layer
    - Inner loop: Loops through each cell in the output layer and computes the cost for a particular training sample. Returns cost of the training sample.
    - Outer loop: Loops through all training sample and computes the cost for the training set (containing the samples).
2. the triple sum simply adds up the squares of all the individual Θs in the entire network.
3. the i in the triple sum does **not** refer to training example i

ALgo
1. Big delta is only until l-1
2. j = 0 corresponds to bias term --> Thus no regularisation for it