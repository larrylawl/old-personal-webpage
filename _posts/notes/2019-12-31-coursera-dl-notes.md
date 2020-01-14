---
layout: post
title: "Coursera, deeplearning.ai: Deep Learning Notes"
author: "Larry Law"
categories: notes
image: neural-network.jpeg
hidden: true
---
Lecturer: Professor Andrew Ng <br>
Course available [here](https://www.coursera.org/specializations/deep-learning).<br>

<!-- omit in toc -->
## Table of Contents
- [Course 1: Neural Networks and Deep Learning](#course-1-neural-networks-and-deep-learning)
  - [Defensive Programming with Matrixes](#defensive-programming-with-matrixes)
  - [Activation Functions](#activation-functions)
  - [Notable Quiz Questions](#notable-quiz-questions)
- [Course 2: Improving Deep Neural Networks - Hyperparameter tuning, Regularization and Optimization](#course-2-improving-deep-neural-networks---hyperparameter-tuning-regularization-and-optimization)
  - [Practical Aspects of Deep Learning: Learning Outcomes](#practical-aspects-of-deep-learning-learning-outcomes)
    - [Setting Up Your Machine Learning Application](#setting-up-your-machine-learning-application)
      - [Decreasing Significance of Bias-Variance Tradeoff](#decreasing-significance-of-bias-variance-tradeoff)
      - [Train/Dev/Test split and tips](#traindevtest-split-and-tips)
    - [Regularising Your Neural Network](#regularising-your-neural-network)
      - [Dropout Regularisation](#dropout-regularisation)
      - [Early stopping](#early-stopping)
      - [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
      - [Weight Initialisation for Deep Networks](#weight-initialisation-for-deep-networks)
  - [Optimisation Alogrithms](#optimisation-alogrithms)
    - [Difference Between Batch, Mini-Batch, and Stochastic Gradient Descent](#difference-between-batch-mini-batch-and-stochastic-gradient-descent)
    - [Exponentially Weighted Average](#exponentially-weighted-average)
    - [Bias Correction](#bias-correction)
    - [Gradient Descent with Momentum](#gradient-descent-with-momentum)
    - [RMSprop](#rmsprop)
    - [Adam](#adam)
    - [Learning Rate Decay](#learning-rate-decay)
    - [The problem of local optima](#the-problem-of-local-optima)
    - [Notable Quiz Questions](#notable-quiz-questions-1)
  - [Hyperparameter tuning, Batch Normalization and Programming Frameworks](#hyperparameter-tuning-batch-normalization-and-programming-frameworks)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
      - [Tuning Process](#tuning-process)
      - [Using an appropriate scale to pick hyperparameters](#using-an-appropriate-scale-to-pick-hyperparameters)
      - [Hyperparameters tuning in practice](#hyperparameters-tuning-in-practice)
    - [Batch Normalisation](#batch-normalisation)
      - [Implementation](#implementation)
      - [Why do we need batch normalisation, and why does it work?](#why-do-we-need-batch-normalisation-and-why-does-it-work)
      - [Batch norm at Test Time](#batch-norm-at-test-time)
    - [Multi-class classifiction](#multi-class-classifiction)
      - [Softmax Regression](#softmax-regression)
      - [Loss Function of Softmax](#loss-function-of-softmax)
    - [Deep Learning Frameworks](#deep-learning-frameworks)
- [Course 3: Convolutional Neural Networks (CNN)](#course-3-convolutional-neural-networks-cnn)
  - [Foundations of CNN](#foundations-of-cnn)
    - [Why learn CNN?](#why-learn-cnn)
    - [Convolution Operation](#convolution-operation)
    - [Padding](#padding)
    - [Strided Convolutions](#strided-convolutions)
    - [Summary of Notations for Convolution Operation](#summary-of-notations-for-convolution-operation)
    - [Pooling Layers](#pooling-layers)
    - [Fully Connected Layer](#fully-connected-layer)
    - [CNN Example](#cnn-example)

# Course 1: Neural Networks and Deep Learning
## Defensive Programming with Matrixes
```
a = np.random.randn(5) 
# a.shape = (5,) 
# rank 1 array (as it has only one axis) - don't use!

a = np.random.randn(5, 1) # a.shape = (5, 1)
a = np.random.randn(1, 5) # a.shape = (1, 5)
assert(a.shape == (5, 1))
```

## Activation Functions
Refer to my article [here](/articles/comparison-between-activation-functions.html)

## Notable Quiz Questions
> Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

True, Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector.

# Course 2: Improving Deep Neural Networks - Hyperparameter tuning, Regularization and Optimization

## Practical Aspects of Deep Learning: Learning Outcomes
1. Recall that different types of initializations lead to different results
2. Recognize the importance of initialization in complex neural networks.
3. Recognize the difference between train/dev/test sets
4. Diagnose the bias and variance issues in your model
5. Learn when and how to use regularization methods such as dropout or L2 regularization.
6. Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
7. Use gradient checking to verify the correctness of your backpropagation implementation

### Setting Up Your Machine Learning Application
#### Decreasing Significance of Bias-Variance Tradeoff
1. **To reduce bias (wo affecting variance):** Get more data and regularise
2. **To reduce variance (wo affecting bias):** Get more data

#### Train/Dev/Test split and tips
1. **Smaller dataset (1K-10K):** 60/20/20 split.
2. **Large dataset (> 1M):** 98/1/1 split. Purpose of the dev/test set is to evaluate the algorithm, thus you don't need so much data.
3. **Obtain the training and test sets from the same distribution:** As more data wins, tendency is to get training data from different distributions from the test data, which will cause your model to learn wrongly.

### Regularising Your Neural Network
#### Dropout Regularisation
Intuition: Not put all your eggs in one basket. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.


```py
# l = 3, keep_prob = 0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3) # a3 *= d3
a3 /= keep_prob # Invert the dropout by maintaining the expected value of a3
```

> Why do we remove dropout in at test time?

Ensure that our predictions are deterministic at test time.

**Drawback**: cost function *J* is less well defined (as every iteration we eliminate nodes at random)

**Solution**: Turn off dropout first (ie `keep_prob = 1`) and check that J is monotonely decreasing. This ensures that our gradient descent is implemented correctly. Then turn on dropout.

#### Early stopping
Not recommended as it couples both objectives of optimising cost function J and solving overfitting (as opposed to gradient descent for J and regularisation for overfitting).

#### Vanishing and Exploding Gradients
Refer to the article [here](/articles/comparison-between-activation-functions.html)

#### Weight Initialisation for Deep Networks

<!-- Clarify how does increasing z relate to gradient? -->

<!-- Why does weight initialisation work?

Hi, I have 2 questions wrt this slide.

1. How does preventing the value of *z* from increasing/decreasing exponentially circumvent the vanishing gradient problem? Aren't we concern with the gradient of the activation function (ie g'(z)), rather than the input argument (ie z)? 
2. Suppose 1. is true, why does adjusting the value of variance prevent the value of z from increasing/decreasing exponentially? The expected value of z, E[z], will still be 0 no? Here's my working:

E[z] = E[w1x1 + ... + wnxn]
= E[w1x1] + ... + E[wnxn] (by linearity of expectations)
= E[w1]E[x1] + ... + E[wn]E[xn] (since w and x are independent)
= 0 (since E[w] = 0) -->

<!-- 1. Prevent z from blowing up (refer to graph on `activation function) --> As n increases, wi decreases
2. By keeping my variance inversely proportional to n, as n increases, w will spread v little from the mean which is 0, thus it decreases.  -->

```py
def initialize_parameters_he(layers_dims):
...
for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt([np.divide(2, layers_dims[l - 1])])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
```
This works as \$ Var(aX) = a^2Var(X) \$

1. Weight of Relu: \$ \frac{2}{n^{l - 1}} \$ (*He Initialisation*)
2. Weight of TanH: \$ \frac{1}{n^{l-1}} or \frac{2}{n^{l-1} + n^{l}} \$ (*Xavier Initialisation*)

## Optimisation Alogrithms
1. Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
2. Use random minibatches to accelerate the convergence and improve the optimization
3. Know the benefits of learning rate decay and apply it to your optimization

### Difference Between Batch, Mini-Batch, and Stochastic Gradient Descent

The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is *the number of examples* you use to perform one update step.

> With a well-turned mini-batch size, usually mini-batch gd outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large). Mini-batch size performs better than stochastic as it leverages vectorisation.

> Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

### Exponentially Weighted Average
Weighted average of all the previous velocities, where the coefficients (or probability) exponentially decreases as t decreases. 

$$
v_t = \beta v_{t - 1} + (1 - \beta)\theta_t
$$

![Exponentially Weighted Average](/assets/img/2019-12-31-coursera-dl-notes/exponentially-weighted-average.png)

> coefficient => \$ f(x) = 0.1 \times 0.9^{-x} \$, where x is \$ v_t \$). 

Advantage of exponentially weighted average is that (a) its O(1) space complexity while (b) considering the weighted average of all the previous parameters, not jus the current parameter.

> Why is \$ v_t \$ approximately average over \$ \frac{1}{1-\beta} days? \$

$$
(1 - \epsilon)^{\frac{1}{\epsilon}} = \frac{1}{e} \\
0.9^{10} \approx 0.35 \approx \frac{1}{e}
$$

In the example above, when \$ \beta = 0.1 \$, it is an approximation for the weighted average over the past 10 days, as the coefficients for the days after that are too small for the number to be meaningful. 

### Bias Correction

$$
\begin{array}{l}{v_{d W^{[l]}}=\beta_{1} v_{d W^{[l]}}+\left(1-\beta_{1}\right) \frac{\partial \mathcal{J}}{\partial W^{[l]}}} \\ {v_{d W^{[l]}}^{c o r r e c t e d}=\frac{v_{d W}[l]}{1-\left(\beta_{1}\right)^{t}}}\end{array}
$$

- When _t_ is small, then bias correction scales up \$ v_t \$ (as the velocity, \$ v_{t - 1} \$, is very small)
- When _t_ is large, then bias correction has no effect on \$ v_t \$ (denominator ~= 1)

### Gradient Descent with Momentum

Intuitively, GD with Momentum (a) dampens oscillations in directions of high curvature by cancelling out gradients with opposite signs and (b) builds up velocity in directions (towards min) with a gentle but consistent gradient (since they don't cancel each other out).

$$
\begin{aligned}\left\{\begin{array}{l}{v_{d W^{[l]}}} & {=\beta v_{d W^{[l]}}+(1-\beta) d W^{[l]}} \\ {W^{[l]}} & {=W^{[l]}-\alpha v_{d W^{[l]}}}\end{array}\right.\\\left\{\begin{array}{l}{v_{d b^{[l]}}} & {=\beta v_{d b^{[l]}}+(1-\beta) d b^{[l]}} \\ {b^{[l]}} & {=b^{[l]}-\alpha v_{d b^{[l]}}}\end{array}\right.\end{aligned}
$$

\$ d W^{[l]} \$ is acceleration, \$ v_{d W^{[l]}} \$ is velocity, \$ beta \$ (being <1) is friction.

> Why is this called GD with momentum?

Its momentum makes it keep going in the previous direction.

**Comparison of Gradient Descent w/ various momentums**
1. Gradient Descent
2. Gradient Descent w smaller momentum
3. Gradient Descent w larger momentum

![Gradient Descent Comparison](/assets/img/2019-12-31-coursera-dl-notes/grad-descent-comparison.png)

> Why does gradient descent oscillate so sharply? 

Because you are taking the derivative only from the previous iteration (ie beta = 0), and not the exponential weighted average of all the previous iterations, thus it moves more sharply.


### RMSprop
TODO.
<!-- What if it's descending towards the correct direction? -->
<!-- Derivatives are much larger in the vertical direction than the horizontal direction -->
<!-- Gradient towards the cost function is near 0 (horizontal) -->
<!-- Why square? -->
<!-- Why mean? -->

### Adam
Combination of gradient descent with momentum and RMSprop.

$$
\left\{\begin{array}{l}{v_{d W^{[l]}}=\beta_{1} v_{d W^{[l]}}+\left(1-\beta_{1}\right) \frac{\partial \mathcal{J}}{\partial W^{[l]{l}}}} \\ {v_{d W^{[l]}}^{c o r r e c t e d}=\frac{v_{d W^{[l]}}}{1-\left(\beta_{1}\right\}^{t}}} \\ {s_{d W^{[l]}}=\beta_{2} s_{d W^{[l]}}+\left(1-\beta_{2}\right)\left(\frac{\partial J}{\partial W^{[l]}}\right)^{2}} \\ {s_{d W^{[l]}}^{c o r r e c t e d}=\frac{s_{d W l}[l]}{1-\left(\beta_{2}\right)^{t}}} \\ {W^{[l]}=W^{[l]}-\alpha \frac{v_{d W^{(l)}}^{r_{\text {diving }}}}{\sqrt{s_{d W[l]}^{s s w(l)}+\varepsilon}}}\end{array}\right.
$$

> Not different \$ beta_i \$ for _s_ (RMS prop) and _v_ (grad desc w momentum)


Some advantages of Adam include:
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except \$ \alpha \$)

Adam paper [here](https://arxiv.org/pdf/1412.6980.pdf)

### Learning Rate Decay

![learning rate decay](/assets/img/2019-12-31-coursera-dl-notes/learning-rate-decay.png)

1. When you just started training, it's okay to take bigger steps.
2. When nearing minimum point, take smaller steps so that you oscillate in a tighter region around this minimum. 

### The problem of local optima

![local min vs saddle points](/assets/img/2019-12-31-coursera-dl-notes/local-min-vs-saddle.png)

In higher dimensional space, it's more likely to obtain *saddle points* instead of *local minimums*. This is because to obtain a local minimum, **all** _n_ dimensions need to be a convex-like function. In contrast to obtain a saddle point, you only need a mixure of convex and concave functions. Consequently, we need not worry too much about gradient descent being stuck at local minimums.

![saddle point](/assets/img/2019-12-31-coursera-dl-notes/saddle.png)

However, the problem with saddle points is that it take very long to go down the plateau. Thus there's a need for gradient descent algorithms which work faster.

### Notable Quiz Questions

![Quiz on exponentially weighted average](/assets/img/2019-12-31-coursera-dl-notes/quiz-exponentially-weighted-avg.png)

When you increase \$ \beta \$ , you are taking into account more days, thus the graph adapts more slowly (consequently smoother), and hence the red line is shifted slightly to the right

## Hyperparameter tuning, Batch Normalization and Programming Frameworks
1. Master the process of hyperparameter tuning
2. TensorFlow

### Hyperparameter Tuning
#### Tuning Process
**Priortise parameters in the following order:**
1. Learning Rate (\$ \alpha \$)
2. Momentum Term (\$ \beta \$), no. hidden units, Mini-batch size
3. No. of layers, learning rate decay
4. Adam's parameters 

> Recommended by Andrew Ng

**Use random values instead of grid:** a 5x5 grid will only yield 5 distinct values of x or y, but 25 randomly chosen values will yield 25 distinct values of x and y.

**Coarse to fine scheme:** zooming into areas of interest

![Coarse to fine scheme](/assets/img/2020-1-1-comparison-between-activation-functions/coarse-to-fine.png)

#### Using an appropriate scale to pick hyperparameters
Scale according to orders of magnitude. Here's a great explanation from classmate Giovanna Roda.

> The problem with choosing numbers at random from an interval pops up when the two endpoints are of two different orders of magnitude. <br />
> For instance, it makes sense to select uniformly random numbers from the interval [3,10] because these two numbers have the same order of magnitude. <br />
> But if you select at random from [0.0001, 10] then 90% of your numbers will be between 1 and 10 and you will have only 10% between 0.0001 and 1. This might be a disadvantage for the choice of hyperparameters. If you use a very small value 0.0001 on the left, you probable want to try out many different small values.  <br />
> So what you actually want is an uniform random choice across orders of magnitude. But order of magnitude is nothing else but the logarithm (0.0001 has order of magnitude -4 in base 10, 10 has order 1). So instead of choosing a random number between [0.0001,10] you choose a random exponent between [-4, 1] (note: an exponent can be also a decimal number) and exponentiate 10 (the base) to it.

From the above explanation, we would expect,

$$
\alpha = 10 ^ r \text{(Learning rate)} \\
\beta = 1 - 10^r \text{(Exponentially weighted average)}
$$

> \$ \beta \$ is scaled as such because it is more sensitive when beta is closer to _1_. Plot \$ \frac{1}{1 - \beta} \$ to see this sensitivity.

#### Hyperparameters tuning in practice
How do we tune our hyperparameters in practice? Depends on **computational power**.
1. If you have it, then train multiple models in parallel and see which hyperparameters work best.
2. Else, babysite a model and manually tune the hyperparameters.

### Batch Normalisation
#### Implementation
For each hidden layer of the NN of each mini batch, normalise _z_.

> Don't normalise the input and output layers!

$$
\begin{aligned} \mu &=\frac{1}{m} \sum_{i} z^{(i)} \\ \sigma^{2} &=\frac{1}{m} \sum_{i}\left(z^{(i)}-\mu\right)^{2} \\ z_{\text {norm }}^{(i)} &=\frac{z^{(i)}-\mu}{\sqrt{\sigma^{2}+\varepsilon}} \\ \tilde{z}^{(i)} &=\gamma z_{\text {norm }}^{(i)}+\beta \end{aligned}
$$

> \$ \epsilon \$ is added for numerical stability (prevent division by 0)

> Parameters (\$ \mu \$ and \$ \gamma \$) are trained using normal learning methods (such as gradient descent).

![batch norm neural network](/assets/img/2020-1-1-comparison-between-activation-functions/batch-norm-nn.png)

#### Why do we need batch normalisation, and why does it work?
Batch normalisation makes **gradient descent quicker** because it...
1. **Normalises each layer st they have the same scale:** Having the same scale allows descent to be smoother as opposed to an oscillating descent. (same reason as feature scaling)
2. **Limits the covariate shift of the previous layers:** By normalising each layer, the mean and variance is fixed by \$ \mu \$ and \$ \gamma \$ respectively. This limits the covariate shift of each layer, thus the more stable changes allow the subsequent layers to have less adjustments to make, and hence learn faster.

> Covariate shift is simply the shift in the distribution of the input layer. If covariate shift occurs, our model needs to relearn the mapping of the input layer and the output.

> Because it limits covariate shift, **batch norm has a slight regularisation effect:** By adding noise to each layer, further layers are forced to not rely on the previous layer too much (similar to dropout). However, don't use batch norm to regularise - single role responsibility!

#### Batch norm at Test Time
Estimate \$ \mu \$ and \$ \gamma \$ using **exponentially weighted average from training set**. (no need to compute the parameters from the entire training set)
  
> Why do we need to batch norm at test time?

The weights are trained on normalised \$ z_{train} \$ of the input layers. In order to maintain the same distribution between \$ z_{train} \$ and \$ z_{test} \$, we do batch norm at test time too.

### Multi-class classifiction
#### Softmax Regression
Softmax regresion generalises logistic regression to *C* classes.

![softmax activation function](/assets/img/2020-1-1-comparison-between-activation-functions/softmax.png)

> Hardmax: Only one one, rest zeros

> If *C = 2*, softmax regression reduces to logistic regression

#### Loss Function of Softmax

$$
J(y, p)=-\sum_{i} y_{i} \log \left(p_{i}\right)
$$

Intuitively, this loss function looks at whatever is the ground true class in the training set, and tries to make the corresponding probability of that class as high as possible.

> This is a form of maximum likelihood estimation

### Deep Learning Frameworks
How does one pick a suitable framework?
1. Ease of programming (dev and dep)
2. Running spseed
3. Truly open (open source w good governance)
  
# Course 3: Convolutional Neural Networks (CNN)
## Foundations of CNN
1. Understand the convolution operation
2. Understand the pooling operation
3. Remember the vocabulary used in convolutional neural network (padding, stride, filter, ...)
4. Build a convolutional neural network for image multi-class classification

### Why learn CNN?
1. New Technology
2. Computer Vision research inspires other sub-domains (cross-fertilization)

### Convolution Operation
**Motivation:** Every image represents very large matrixes (3D and RGB; ie 1000x1000x1000x3); there are too many parameters to learn. There needs to be a more efficient way to learn these large matrixes. 

**Intuition:** Convolution operation is applying a *filter* on the original matrix. In doing so, it solves the problem of large matrixes by having...
1. **Parameter sharing:** A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image; able to reuse the feature detector across the image. (ie a filter is reused across the entire image)
2. **Sparsity of connections:** Each output value depends only on a small number of values (ie *-5* is dependent only on the *9/36* values of the input image and the filter image)

![Edge Detection](/assets/img/2019-12-31-coursera-dl-notes/edge-detection.png)
![Convolution Superimpose Example](/assets/img/2019-12-31-coursera-dl-notes/convolution-rgb.png)
![Convolution Layer Example](/assets/img/2019-12-31-coursera-dl-notes/convolution-layer.png)
Video available [here](/assets/img/2019-12-31-coursera-dl-notes/convolution-operation.mp4)

### Padding
**Motivation:** 
1. **Retain same dimensions of image**: It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.
2. **Retains information at the border:** Without padding, very few values at the next layer would be affected by pixels at the edges of an image.

> Padding does not affect channel of filter matrix (≠ output!!), only height and width. This is because you apply convolution filter across all channels. 
> ![Convolution Superimpose Example](/assets/img/2019-12-31-coursera-dl-notes/convolution-rgb.png)

> *f* is usually odd: 1) symmetric padding 2) Presense of central pixel so that you can talk about the position of the filter

### Strided Convolutions
Hyperparemeter which determines **how much the convolution window steps over**.

### Summary of Notations for Convolution Operation
![CNN notation summary](/assets/img/2019-12-31-coursera-dl-notes/cnn-notation-summary.png)
1. *f* refers to the filter layer (ie *f x f*)
2. The channel in each filter is dependent on the channel of the previous layer (ie \$ n_c^{[l-1]}\$) as it is applied across all channel at once (see picture)
> ![Convolution Superimpose Example](/assets/img/2019-12-31-coursera-dl-notes/convolution-rgb.png)
3. Each filter and bias term is equivalent to *z*
![Convolution Layer Example](/assets/img/2019-12-31-coursera-dl-notes/convolution-layer.png)
1. Activation will thus be the output of *g(z)* of point 3.
2. Parameters refer to each entry of the filter layers. Thus *weights = weights in each filter * no. of filters.*

### Pooling Layers
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. 

1. **Max-pooling layer:** slides a *(f, f)* window over the input and stores the max value of the window in the output.
   1. **Intuition:** Large number means that it has detected a particular feature
2. **Average-pooling layer:** slides an *(f, f)* window over the input and stores the average value of the window in the output.

![pooling layer](/assets/img/2019-12-31-coursera-dl-notes/pooling-layer.png)

Points to note
1. Hyperparameters: _f_ and _s_
2. **No parameters to learn:** You can't change these parameters as changing them means changing the size of each layers, which changes size of weights (which is fixed)
3. \$ N_c \$ does not change since pooling is applied layer by layer.

### Fully Connected Layer
Essentially a **normal neural network**, which allows the CNN to **train more parameters**. Usually used in the last few layers (wherein the no. of parameters are reduced via the pooling layer)

![Fully connected layer](/assets/img/2019-12-31-coursera-dl-notes/cnn-fully-connected-layer.png)

### CNN Example
![CNN Example](/assets/img/2019-12-31-coursera-dl-notes/cnn-example.png)
1.  Max pooling layers don't have parameters
2.  Parameters mainly come from FC
3.  Activation size drops gradually across the activation layer
4.  Conv -> Pool (to reduce image) -> FC

Notable Quiz Qns

> Because pooling layers do not have parameters, they do not affect the backpropagation (derivatives) calculation.

True. Back propagation does have to cross the pooling layers as well. Exactly what it does depends on whether it is a max pooling or average pooling layer. For a max layer, the gradient is passed only to the maximum of the inputs. For an average pooling layer, it gets equally distributed to the inputs. Mathematical explanation available [here.](https://datascience.stackexchange.com/questions/11699/backprop-through-max-pooling-layers)(Locally linear w slope 1 does not change the input gradient.)

> What is an Epoch?

One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
