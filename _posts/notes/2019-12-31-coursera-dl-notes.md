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
Standard notations for deep learning [here](/assets/img/2019-12-31-coursera-dl-notes/standard-notations-dl.pdf)

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
- [Course 4: Convolutional Neural Networks (CNN)](#course-4-convolutional-neural-networks-cnn)
  - [Foundations of CNN](#foundations-of-cnn)
    - [Why learn CNN?](#why-learn-cnn)
    - [Convolution Operation](#convolution-operation)
    - [Padding](#padding)
    - [Strided Convolutions](#strided-convolutions)
    - [Summary of Notations for Convolution Operation](#summary-of-notations-for-convolution-operation)
    - [Pooling Layers](#pooling-layers)
    - [Fully Connected Layer](#fully-connected-layer)
    - [CNN Example](#cnn-example)
  - [Deep Convolutional Models: Case Studies](#deep-convolutional-models-case-studies)
    - [Case Studies](#case-studies)
      - [Classic Networks](#classic-networks)
      - [ResNets: What and Why it Works](#resnets-what-and-why-it-works)
      - [ResNets: Implementation Details](#resnets-implementation-details)
      - [Networks in Networks and 1x1 Convolutions](#networks-in-networks-and-1x1-convolutions)
      - [Inception Modules: What and Why](#inception-modules-what-and-why)
      - [Inception Network](#inception-network)
    - [Practical advices for using ConvNets](#practical-advices-for-using-convnets)
      - [Transfer Learning](#transfer-learning)
      - [Data augmentation](#data-augmentation)
      - [State of Computer Vision](#state-of-computer-vision)
  - [Object Detection](#object-detection)
    - [Object localisation](#object-localisation)
    - [Landmark detection](#landmark-detection)
    - [Object Detection](#object-detection-1)
    - [Conv implementation of sliding windows](#conv-implementation-of-sliding-windows)
    - [Bounding Box prediction](#bounding-box-prediction)
    - [Intersection over Union (IOU)](#intersection-over-union-iou)
    - [Non-max suppression](#non-max-suppression)
    - [Anchor Boxes](#anchor-boxes)
    - [Yolo algo](#yolo-algo)
    - [Region Proposal](#region-proposal)
    - [Notable Quiz Questions](#notable-quiz-questions-2)
  - [Special Applications: Face Recognition & Neural Style Transfer](#special-applications-face-recognition--neural-style-transfer)
    - [Face Recognition](#face-recognition)
      - [Face Verification vs Face Recognition](#face-verification-vs-face-recognition)
      - [One shot learning](#one-shot-learning)
      - [Siamese Network](#siamese-network)
      - [Triple Loss](#triple-loss)
      - [Face Verification and Binary Classification](#face-verification-and-binary-classification)
    - [Neural Style Transfer](#neural-style-transfer)
      - [What are deep ConvNets learning?](#what-are-deep-convnets-learning)
      - [Cost Function](#cost-function)
      - [Convolutional Networks in 1D or 3D](#convolutional-networks-in-1d-or-3d)
- [Course 5: Sequence Models](#course-5-sequence-models)
  - [Recurrent Neural Networks](#recurrent-neural-networks)
    - [Why Sequence Models](#why-sequence-models)
    - [Notation](#notation)
    - [RNN Model](#rnn-model)
    - [Different types of RNNs](#different-types-of-rnns)
    - [Language modelling](#language-modelling)
    - [Sampling novel sequences](#sampling-novel-sequences)
    - [Vanishing Gradients with RNNs](#vanishing-gradients-with-rnns)
    - [Gated Recurrent Unit](#gated-recurrent-unit)
    - [Long Short Term Memory (LSTM)](#long-short-term-memory-lstm)
    - [Bidirectional RNNs](#bidirectional-rnns)
    - [Deep RNN](#deep-rnn)
  - [Natural Language Processing & Word Embeddings](#natural-language-processing--word-embeddings)
    - [Introduction to Word Embeddings](#introduction-to-word-embeddings)
    - [Learning Word Embeddings: Word2vec & GloVe](#learning-word-embeddings-word2vec--glove)
    - [Applications using Word Embeddings](#applications-using-word-embeddings)
  - [Various Sequence to Sequence Architectures](#various-sequence-to-sequence-architectures)
    - [Speech Recognition - Audio Data](#speech-recognition---audio-data)

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
Softmax regresion generalises logistic regression to *K* classes.

Given a test input x, we want our hypothesis to estimate the probability that P(y=k\|x) for each value of k=1,…,K.

$$
h_{\theta}(x)=\left[\begin{array}{c}{P(y=1 | x ; \theta)} \\ {P(y=2 | x ; \theta)} \\ {\vdots} \\ {P(y=K | x ; \theta)}\end{array}\right]=\frac{1}{\sum_{j=1}^{K} \exp \left(\theta^{(j) \top} x\right)}\left[\begin{array}{c}{\exp \left(\theta^{(1) \top} x\right)} \\ {\exp \left(\theta^{(2) \top} x\right)} \\ {\vdots} \\ {\exp \left(\theta^{(K) \top} x\right)}\end{array}\right]
$$

Here, \$ \theta^{(i)} \$ are the parameters of our model. Notice that the term \$ \frac{1}{\sum_{j=1}^{K} \exp \left(\theta^{(j) \top} x\right)} \$ normalises the distribution so that it sums to one.

> Hardmax: Only one one, rest zeros

> If *C = 2*, softmax regression reduces to logistic regression

#### Loss Function of Softmax

$$
J(\theta)=-\left[\sum_{i=1}^{m} \sum_{k=1}^{K} 1\left\{y^{(i)}=k\right\} \log \frac{\exp \left(\theta^{(k) \top} x^{(i)}\right)}{\sum_{j=1}^{K} \exp \left(\theta^{(j) \top} x^{(i)}\right)}\right]
$$

In this equation, 1{.} is the indicator function.

Intuitively, this loss function looks at whatever is the ground true class in the training set, and tries to make the corresponding probability of that class as high as possible.

> This is a form of maximum likelihood estimation

### Deep Learning Frameworks
How does one pick a suitable framework?
1. Ease of programming (dev and dep)
2. Running spseed
3. Truly open (open source w good governance)
  
# Course 4: Convolutional Neural Networks (CNN)
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
5.  \$ n_H \$ and \$ n_W \$ decreases, while \$ n_C \$ increases.

Notable Quiz Qns

> Because pooling layers do not have parameters, they do not affect the backpropagation (derivatives) calculation.

True. Back propagation does have to cross the pooling layers as well. Exactly what it does depends on whether it is a max pooling or average pooling layer. For a max layer, the gradient is passed only to the maximum of the inputs. For an average pooling layer, it gets equally distributed to the inputs. Mathematical explanation available [here.](https://datascience.stackexchange.com/questions/11699/backprop-through-max-pooling-layers)(Locally linear w slope 1 does not change the input gradient.)

> What is an Epoch?

One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.

## Deep Convolutional Models: Case Studies
**Learning Objectives**
1. Understand multiple foundational papers of convolutional neural networks
2. Analyze the dimensionality reduction of a volume in a very deep network
3. Understand and Implement a Residual network
4. Build a deep neural network using Keras
5. Implement a skip-connection in your network
6. Clone a repository from github and use transfer learning

### Case Studies
#### Classic Networks
**Why Look at Classic Networks?** They serve as boilerplates. An architecture that has worked well on one computer vision task often works well on other tasks.

**Classic Networks Examples**

![LeNet](/assets/img/2019-12-31-coursera-dl-notes/LeNet.png)

**AlexNet**
![AlexNet](/assets/img/2019-12-31-coursera-dl-notes/AlexNet.png)
1. Similarity to Lenet, but much bigger
2. ReLU instead of sigmoid/Tanh

**VGG**
![VGG](/assets/img/2019-12-31-coursera-dl-notes/VGG.png)
1. Simplified architecture
2. \$ n_h, n_w \$ decreases by *1/2* but \$ n_c \$ increases by *1/2*

#### ResNets: What and Why it Works
![ResBlock](/assets/img/2019-12-31-coursera-dl-notes/resblock.png)
![ResNet](/assets/img/2019-12-31-coursera-dl-notes/resnet.png)

**Why does ResNets work?**
*How does it achieve monotonely decrease training error even with no. of layers? (ie graph on right instead of left)*

1. **Skip-connection does not worsen training:** 
   1. What goes wrong in very deep plain nets in very deep network without this residual of the skip connections is that when you make the network deeper and deeper, it's actually very difficult for it to choose parameters that learn even the identity function which is why a lot of layers end up making your result worse rather than making your result better.
   2. ResNet circumvents that by making it easy to learn the identity function. If using L2 regularisation, *w* and *b* will be minimised. If *w* and *b = 0*, *g(al) = al* (identity function).
2. **Opportunity to learn a more complex function:** More layers, thus more complex function.
3. **Skip-connection speeds up backpropagation:** 
   1. **Propagating through fewer layers:** If grad ~= 0, it'll be the identity function. Consequently, the current layer and the upstream layer will essentially be the same layer, thus gradient descent will be propagating through fewer layers
   2. **Avoids the problem of vanishing gradient:** if the output from the adjacent layer ~= 0 (thus grad ~= as we are using a ReLU activation fn), you'll use the gradient of the upstream layer's activation instead. (similar explanation to the learning of identity function when *w* and *b* ~= 0)

#### ResNets: Implementation Details
![Identity Block](/assets/img/2019-12-31-coursera-dl-notes/identity-block.png)
**Identity Block:** The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say \$ a^{[l]} \$ ) has the same dimension as the output activation (say \$ a^{[l+2]} \$)

![Convolution Block](/assets/img/2019-12-31-coursera-dl-notes/conv-block.png)
**Convolution Block:** resize the input *x* to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. (This plays a similar role as the matrix \$ W_s \$ discussed in lecture.). The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step.

#### Networks in Networks and 1x1 Convolutions
![1x1 convolutions](/assets/img/2019-12-31-coursera-dl-notes/1x1-convolution.png)

> You can use a 1x1 convolutional layer to reduce \$ n_C \$ but not \$ n_H \$ and \$ n_W \$

#### Inception Modules: What and Why
![Inception Module](/assets/img/2019-12-31-coursera-dl-notes/inception-module.png)

**Inception Network Motivation:**
Use all types of permutation of layering techniques, and let network choose which one to learn. However, it is computationally expensive to combine these different techniques (ie channel concat). 

![Inception Module Problem](/assets/img/2019-12-09-coursera-ml-notes/inception-module-problem.png)


This is resolved by using the 1x1 convolutions to reduce the input data volume's size, before applying larger convolutions.

![Inception Module Solution](/assets/img/2019-12-09-coursera-ml-notes/inception-module-solution.png)


#### Inception Network
![Inception Network](/assets/img/2019-12-09-coursera-ml-notes/inception-network.png)
Inception Network

> Fun Fact: this meme is actually cited in the research paper. 
![Inception Meme](/assets/img/2019-12-09-coursera-ml-notes/inception-meme.png)

### Practical advices for using ConvNets
#### Transfer Learning
**Use other's initialisation** which they have spent many mths training. Concretely,

1. Use architectures of networks published in the literature
2. Use open source implementations if possible
3. Use pretrained models and finetune on your dataset

Implementation Notes:
1. Simply train softmax layer and freeze parameters of other layers
2. Since weights are frozen in the hidden layers, the activations are (almost) deterministic, thus you can save the activations into the disk to reuse them. 
3. If you have data, then you can freeze fewer layers (ie keep the weights of the later layers unfrozen).

#### Data augmentation
Computer Vision is often limited by the lack of data. Thus we use data augmentation. Common techniques include

1. Mirroring
2. Random Cropping
3. Color Shifting (which uses PCA)
4. Implementing distortions during training

#### State of Computer Vision
![Computer Vision State](/assets/img/2019-12-09-coursera-ml-notes/computer-vision-state.png)

Interesting Points
1. Lack of data in computer vision, which leads to more hand engineering.

Given the relative importance of hand engineering, cv focuses more on doing well on benchmarks and on winning competitions. (likelier to get published)
1. Ensembling - Train several networks independently and average their outputs
2. Multicrop - make sure you get it right by averaging 10 results

## Object Detection
1. Understand the challenges of Object Localization, Object Detection and Landmark Finding
2. Understand and implement non-max suppression
3. Understand and implement intersection over union
4. Understand how we label a dataset for an object detection application
5. Remember the vocabulary of object detection (landmark, anchor, bounding box, grid, ...)

### Object localisation
**Image Classification:** 
**Classification with Localisation:** Localise the object and Identify it
**Detection:** Localise and detect multiple objects

![localisation and detection](/assets/img/2019-12-31-coursera-dl-notes/localisation-and-detection.png)

### Landmark detection
![landmark detection](/assets/img/2019-12-31-coursera-dl-notes/landmark-detection.png)

> Need to (labourously) specify each landmark

### Object Detection
**Sliding Windows Detection Algorithm:** 
![Sliding Window Detection Algorithm](/assets/img/2019-12-31-coursera-dl-notes/sliding-window.png)

1. Take these windows and slide them across the entire image and classify every square region with some stride if it contains a car or not.
3. Increase window size with each iteration
4. Drawback: Computation cost of individually calculating each sliding window.

### Conv implementation of sliding windows
![conv implementation of sliding window](/assets/img/2019-12-31-coursera-dl-notes/conv-imp-sliding-windows.png)

1. Run forward propagation on the entire image
2. **Shared computation:** Entries of output (computed once) corresponds to respective sliding window (computed 9 times for 3x3)

### Bounding Box prediction
Output vector contains the bounding boxes
1. \$ b_x \$: x-coordinate of midpoint wrt top left corner of the boundary box
2. \$ b_y \$: y-coordinate of midpoint wrt top left corner of the boundary box
3. \$ b_h \$: Height of box
4. \$ b_w \$: Width of box

> Possible range: 0 < magnitude of \$ b_x, b_y \$ < 1; magnitude of \$ b_h, b_w \$ > 1 (this is why boundary box sizes can differ)

> \$ b_x, b_y, \$ can exist anywhere on the image; it's a normal coordinate.

### Intersection over Union (IOU)
![intersection over union](/assets/img/2019-12-31-coursera-dl-notes/iou.png)

### Non-max suppression
![non max suppression](/assets/img/2019-12-31-coursera-dl-notes/non-max-supp.png)

![non max suppression algo](/assets/img/2019-12-31-coursera-dl-notes/non-max-supp-algo.png)

1. Motivation: Multiple detections of the same object. We want the best dection of the object.
2. Pick the box with highest prob in the region.
3. Those with high IOU with the highest prob will be suppressed

### Anchor Boxes
![anchor box](/assets/img/2019-12-31-coursera-dl-notes/anchor-boxes.png)

1. Motivation: One grid box want to detect multiple objects
2. Output vector _y_ contains prediction for different anchor boxes.


### Yolo algo
This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

![yolo training](/assets/img/2019-12-31-coursera-dl-notes/yolo-training.png)

> Yolo converts the image (of shape *(m, l, h, 3)*) to output vector *y*, which is also known as the encoding architecture for Yolo

![yolo predictions](/assets/img/2019-12-31-coursera-dl-notes/yolo-prediction.png)

Output 1
![yolo output 1](/assets/img/2019-12-31-coursera-dl-notes/yolo-output-1.png)

Output 2
![yolo output 2](/assets/img/2019-12-31-coursera-dl-notes/yolo-output-2.png)

Output 3
![yolo output 3](/assets/img/2019-12-31-coursera-dl-notes/yolo-output-3.png)

### Region Proposal
![region proposal](/assets/img/2019-12-31-coursera-dl-notes/region-proposal.png)

1. Select just a few blobs, and Run continent classifier
2. Drawback: R-CNN is still slower than yolo algos

### Notable Quiz Questions

> Suppose you are using YOLO on a 19x19 grid, on a detection problem with 20 classes, and with 5 anchor boxes. During training, for each image you will need to construct an output volume yy as the target value for the neural network; this corresponds to the last layer of the neural network. (yy may include some “?”, or “don’t cares”). What is the dimension of this output volume?

grid length x grid height x anchor classes x (5 + classes) = 19 x 19 x (5 x 25) <br />
5: \$ p_c \$, midpoint x, midpoint y, height and width

## Special Applications: Face Recognition & Neural Style Transfer
### Face Recognition
#### Face Verification vs Face Recognition
Verification
1. **Input:** image, name/ID
2. **Output:** whether the input image is that of the claimed person
3. 1:1 problem

Recognition
1. Has a database of *K* persons
2. **Input:** Image
3. **Output:** ID if the image is any of the K persons (or "not recognised")
4. 1:k problem

#### One shot learning
**Motivation:** Difficult to train CNN from just one training example (employee image) <br />
**Solution:** Learn a similarity function, *d(img1, img2)*, which is used to compare between input image and database of image.

$$
d(img1, img2) = \text{degree of difference between images} \\
\text{if } d(img1, img2) ≤ \tau, \text{same} \\
\text{else diff}
$$

#### Siamese Network
![siamese-network](/assets/img/2019-12-31-coursera-dl-notes/siamese-network.png)

#### Triple Loss
1. A: Anchor
2. P: Positive
3. N: Negative

**Learning Objective**
$$
d(A, P) + \alpha ≤ d(A, N) \\
\lVert f(A) -f(P) \rVert^2 + \alpha ≤ \lVert f(A) -f(N) \rVert^2 \\
\lVert f(A) -f(P) \rVert^2 - \lVert f(A) -f(N) \rVert^2 + \alpha ≤ 0
$$

> \$ \alpha \$, called margin, is used to avoid the network learning trivial solns to the inequality (ie *d(A,P) = d(A,N)*)

**Loss Function**

$$
L(A,P,N) = max(\lVert f(A) -f(P) \rVert^2 - \lVert f(A) -f(N) \rVert^2 + \alpha, 0)
$$

During training, if *A, P, N* are chosen randomly, \$ d(A,P) + \alpha ≤ d(A, N) \$ is easily satisfied as the probability of *d(A,P)* and *d(A, N)* being different is high. Consequently, the network does not need to train the weights much to satisfy the inequality.

Solution: Choose triples that're "hard" to train on (ie *d(A,P) ~= d(A, N)*)

#### Face Verification and Binary Classification
Use sigmoid or chi-square function
![Face Verification and Binary Classification](/assets/img/2019-12-31-coursera-dl-notes/face-ver-binary-classification.png)

Implementation notes:
1. Precompute activations of last layer for existing employees to save computation time for face recognition
2. Loop through all images in existing database. Pick the one that has the least distance from the input image. Doing so reduces the problem to a Face verification problem, wherein we simply compare the best image with the input image.

### Neural Style Transfer
> Form of unsupervised, not supervised learning! There is no labelled data.

#### What are deep ConvNets learning?
![Deep Layer](/assets/img/2019-12-31-coursera-dl-notes/deep-layer-2.png)

1. 9 patches constitute 1 hidden unit to be activated (but it's a larger patch of the picture!)
2. Each unit computes increasingly complex features as we go deeper

#### Cost Function
$$
J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G), where
$$

1. **C:** Content image
2. **S:** Style image

Algorithm:
1. Initiate G randomly.
2. Use gradient descent to minimize *J(G)*

> Q: Shouldn't cost be a function of the parameters? 

*G* is the pixels of the generated image. Hence, optimising *J* wrt *G* will change the values of the pixels such that it'll look closer like the original image (thus minimising cost).

**Content Cost Function**

$$
J_{content}(C, G) = 0.5 * \lVert a^{[l](c)} - a^{[l](G)} \rVert^2
$$

> Use a pre-trained ConvNet like VGG

**Style Cost Function** <br />
**Style:** correlation between activations across channels. 

> Highly correlated measures the degree to which high level texture components tend to occur tgt

Let \$ a^{[l]}_{i, j, k}\$ = activation at *(i, j, k)*. \$ G^{[l]}\$ is \$ n^{[l]}_c \times n^{[l]}_c \$

$$
G_{k k^{\prime}}^{[l](G)}=\sum_{i=1}^{n_{H}} \sum_{j=1}^{n_{W}} a_{i, j, k}^{[l](G)} a_{i, j, k^{\prime}}^{[l](G)}
$$

G is also called the [gram matrix](http://mathworld.wolfram.com/GramMatrix.html). Given a set *V* of *m* vectors, the gram matrix G is the matrix of all possible inner products of *V*

$$
g_{i,j} = v_i^{T}v_j
$$

1. Set of *m* vectors: {1, 2, ..., \$ n_c \$} (ie all channels)
2. G includes all possible pairs of channels 
3. Gij compares how similar vi is to vj: If they are highly similar, you would expect them to have a large dot product, and thus for Gij to be large.
4. The diagonal elements Gii measure how "active" a channel i is.

For a particular pair of channels, G measures how high the activations are to each other (thus correlated), across the entire height and width of the channel.

$$
J_{s t y l e}^{[l]}(S, G)=\frac{1}{\left(2 n_{H}^{[l]} n_{W}^{[l]} n_{C}^{[l]}\right)^{2}} \sum_{k} \sum_{k^{\prime}}\left(G_{k k^{\prime}}^{[l](S)}-G_{k k^{\prime}}^{[l](G)}\right)^{2}
$$

> Coefficient is the normalisation constant. However it does not matter that much as the style cost function is premultiplied by the hyperparameter \$ \beta \$ anyway

Overall style cost function is

$$
J_{s t y l e}(S, G) = \sum_l \lambda^{[l]}J_{s t y l e}^{[l]}(S, G)
$$

> We get even better results by combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.

#### Convolutional Networks in 1D or 3D
![conv net in 1d](/assets/img/2019-12-31-coursera-dl-notes/conv-1d.png)

![conv net in 3d](/assets/img/2019-12-31-coursera-dl-notes/conv-3d.png)

# Course 5: Sequence Models
## Recurrent Neural Networks
### Why Sequence Models
![Sequence Examples](/assets/img/2019-12-31-coursera-dl-notes/seq-eg.png)

### Notation
1. \$ a^{(2)[3]\langle 4 \rangle}_5 \$: denotes the activation of the 2nd training example (2), 3rd layer [3], 4th time step , and 5th entry in the vector.
2. \$ T_x^{i} \$: Total number of words for the *ith* example
3. \$ T_y^{i} \$: Total number of output labels for the *ith* example

> \$ X^{(i)\langle t \rangle} \$ can be computed using one hot encoding

### RNN Model
**Why not a standard network?**
1. Doesn't share features learned across different positions of text
> Suppose that network learns that \$ X^{\langle 1 \rangle} \$ is a name. If \$ X^{(i)\langle j \rangle} \$ = \$ X^{(i)\langle t \rangle} \$, then the network should also learn that \$ X^{(i)\langle j \rangle} \$ is a name.
2. Inputs, outputs can be of different lengths in different examples

**RNN Model**
![Basic RNN cell](/assets/img/2019-12-31-coursera-dl-notes/basic-rnn-cell.png)

![Basic RNN](/assets/img/2019-12-31-coursera-dl-notes/basic-rnn.png)

Points to note
1. This RNN can only learn from previous activation functions and not after. Solution: Bidirectional RNN (BRNN)
2. This RNN suffers from vanishing graident problem. Solution: LSTM or GRU network.
3. The RNN works best when each output can be estimated using "local" context. "Local" context refers to information that is close to the prediction's time step *t* .

> The weights and biases (Waa,ba,Wax,bx) are re-used each time ste

$$
a^{\langle t \rangle} = g(W_a[a^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b_a) \\
\hat{y}^{\langle t \rangle} = g(W_{y}a^{\langle t \rangle} + b_y)
$$

**Loss Function**

$$
L^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) = -y^{\langle t \rangle}log\hat{y}^{\langle t \rangle} - (1 - y^{\langle t \rangle})log(1-\hat{y}^{\langle t \rangle}) \\
L(\hat{y}, y) = \sum_{t = 1}^{T_y} L^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})
$$

> As the RNN model is a sequence model over time, the backpropagation process is also called backpropagation through time.

### Different types of RNNs
![Rnn Type Summary](/assets/img/2019-12-31-coursera-dl-notes/rnn-types-summary.png)

1. **Standard RNN:** Many-To-Many (\$ T_x = T_y \$)
2. **Sentiment Classification:** Many-To-One
3. **Music Generation:** One-To-Many
4. **Machine Translation:** Many-To-Many ( \$ T_x ≠ T_y \$)

### Language modelling
**Motivation:** Differentiate between sentences which sound the same (eg "the apple and pair salad" vs "the apple and pear salad").

We do so by comparing the probability of the different sentences. 

$$
P(sentence) = P(y^1, y^2, ..., y^{T_y}) = P(y^1)P(y^2 \mid y^1)P(y^3 \mid y^1, y^2)
$$ 

![RNN model language model](/assets/img/2019-12-31-coursera-dl-notes/rnn-model-lang-model.png)

> Input \$ x^i \$ is of the correct previous word.

> Output is a softmax layer of the entire corpus

> Corpus: large set of eg text

> Most computational music algorithms use some post-processing because it is difficult to generate music that sounds good without such post-processing. The post-processing does things such as clean up the generated audio by making sure the same sound is not repeated too many times, that two successive notes are not too far from each other in pitch, and so on. *One could argue that a lot of these post-processing steps are hacks; also, a lot of the music generation literature has also focused on hand-crafting post-processors*, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the RNN. 

### Sampling novel sequences
**Motivation.** Informally get a sense of what the sequence model has learnt by generating a sentence from an input word.

**Implementation.**
1. After generating \$ y^{\langle t+1 \rangle} \$, we want to sample the next word. 
2. If we select the most probable, the model will always generate the same result given a starting word. Thus to make the results more interesting, we will use `np.random.choice` to select a next letter that is likely, but not always the same. (ie selecting from the probability distribution from the *softmax* layer)
3. Sampling is the selection of a value from a group of values, where each value has a probability of being picked.
4. Sampling allows us to generate random sequences of values.
 
![Sampling Novel Sequence](/assets/img/2019-12-31-coursera-dl-notes/sampling-novel-sequences.png)

> Input (\$ x^{i} \$) comes from previous layer's output (ie \$ \hat{y}^{i-1} \$) instead of output labels (ie \$ y^{i-1} \$)

> Randomly sample according to this soft max distribution.

**Character-level language model**

Adv:
1. Need not deal with unknown words

Disadv:
1. Much longer sequences
   1. Not as good at capturing long range dependencies
   2. Expensive computation

### Vanishing Gradients with RNNs
"The *cat*, which already ate ..., *was* full."

Due to the need to capture long range dependencies (ie *"cat"* and *"was"*), the earlier layers need to learn from the later layers via backprop. However, with so many layers in between, the gradient of the later layers will have a hard time affecting the gradient of the earlier layers.

> Exploding gradients can be addressed using gradient clipping, wherein we "clip" the max value.

### Gated Recurrent Unit 
**Motivation:** NN to capture long range dependencies (ie *"cat"* and *"was"*) within the sentence

"The *cat*, which already ate ..., *was* full."

*c* represents the memory cell, which for the GRU it represents the output value *a*. Intuitively, *c* provides memory for the network to remember down the layers.

$$
c^{\langle t \rangle} = a^{\langle t \rangle}
$$

Candidate \$ \tilde{c} \$

$$
\tilde{c}^{\langle t \rangle} = tanh(W_c[\Gamma_r^{\langle t \rangle} \times c^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b_c), where
$$

\$ \Gamma_r \$ denotes the relevant gate, which determines if the previous memory cell is useful or not. If it is not, then we simply use the input value \$ x \$.

$$
\Gamma_r = \sigma(W_r[c^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b_r)
$$

Gate \$ \Gamma_u \$ decides if it should update with \$ c \$ or \$ \tilde{c} \$ (thus usage of sigmoid function). Ideally, \$ \Gamma_u = 1 \$ for *cat* and *was*, and \$ \Gamma_u = 0 \$ for words in between them.

$$
\Gamma_u = \sigma(W_u[c^{\langle t - 1 \rangle}, X^{\langle t \rangle}] + b_u) \\
c^{\langle t \rangle} = \Gamma_u \times \tilde{c}^{\langle t \rangle} + (1- \Gamma_u) \times c^{\langle t - 1 \rangle}
$$

### Long Short Term Memory (LSTM)
![LSTM](/assets/img/2019-12-31-coursera-dl-notes/LSTM.png)

1. \$ \Gamma_f \$: Forget gate. If the subject changes its state (from a singular word to a plural word), the memory of the previous state becomes outdated, so we "forget" that outdated state.
2. \$ \Gamma_o \$: Output gate. The output gate decides what gets sent as the prediction (output) of the time step.

> No consensus wrt which is better - GRU or LSTM

### Bidirectional RNNs
**Motivation.**
1. He said, "Teddy bears are on sale!"
2. He saisd, "Teddy Roosevelt was a great President!"
3. The meaning of "Teddy" is determined by the word that comes **after** (ie "bears" or "Roosevelt"); the current iteration of sequence model only allows a neuron to learn from the words before it. To resolve this, we have **bidirectional RNNs**.

![BRNN](/assets/img/2019-12-31-coursera-dl-notes/BRNN.png)

### Deep RNN
![Deep RNN](/assets/img/2019-12-31-coursera-dl-notes/deep-rnn-eg.png)

> \$ a^{[2]\langle 3 \rangle} \$ is determined both by its preceding layer of the same word (ie \$ a^{[1]\langle 3 \rangle} \$) and the word before of the same layer (ie \$ a^{[2]\langle 2 \rangle} \$)

> Every column represents one RNN cell

## Natural Language Processing & Word Embeddings
### Introduction to Word Embeddings
**Problem.** One-hot encoding a) does not capture sementic relationship between words. (ie inner product between any pair of word is 0) and b) is large in size

**Solution.** Featurized representation. (Every dimension represent a feature)

**Visualising word embeddings:** *iD* -> *2D* (t-SNE algorithm)

![Visualise Word Embeddings](/assets/img/2019-12-31-coursera-dl-notes/visualise-word-embeddings.png)

**Transfer learning and word embeddings**
1. Learn word embeddings from large text corpus (1-100B words)
2. Transfer embedding to new task with smaller training set (100k words)
3. Optional: Continue to finetune the word embeddings with new data

> Relation to face encoding: Face encoding takes in an image and outputs a 128D vector (and compares this 128D vector with another 128D vector to determine if the images are the same). Word embedding takes in a word and outputs a fixed vector with each dimension capturing a sementic meaning.

**Comparing between Word Embeddings**

**Cosine similarity**

$$
sim(u, v) = \frac{u \cdot v}{\lvert u \lvert \lvert v \lvert} = \frac{\lvert u \lvert \lvert v \lvert cos\theta}{\lvert u \lvert \lvert v \lvert} = cos\theta
$$

where \$ \theta \$ is the angle between the two vectors *u* and *v*,

1. \$ cos\theta = 1 \implies \theta = 0 \$: Perfectly similar
2. \$ cos\theta = 0 \implies \theta = 90 \$: No similarity
3. \$ cos\theta = -1 \implies \theta = 180 \$: Completely opposite directions

**Euclidean distance between vectors**

$$
disim(u, v) = \lvert u - v \lvert^2
$$

**Embedding Matrix**

$$
E_{d, m}
$$

1. *d*: Number of word embedding dimensions
2. *m*: Number of examples

### Learning Word Embeddings: Word2vec & GloVe

<!-- **Learning Word Embeddings** -->
<!-- TODO: insert neural language model -->

**Skip-gram model**
Picking a random context word *c* and outputs a random target word *t* around *c*.

$$
p(t | c)=\frac{e^{\theta_{t}^{T} e_{c}}}{\sum_{j=1}^{10,000} e^{\theta_{j}^{T} e_{c}}}
$$

1. \$ \theta_t =\$ parameter associated with output *t*
2. 10,000 refers to the no. of words in the dictionary.

> This is at the softmax layer

**Problems with softmax classification**
Denominator (summation over *i* number of dictionary words) is computationally expensive. 

Solution: 
1. Hierarchal softmax (similar to hierarchal addressing for IP addresses). 
> Common words are placed at the top of the hierarchal softmax classifier, as opposed to having a BBST


<!-- **Negative Sampling** -->
<!-- Q: Don't understand-->

<!-- **GloVe word vectors**
GloVe: Global vectors for word representation

$$
X_{i,j} = \text{# times j appears in context of i}
$$

> i: context, j: target

How related are i and j, depending on how closely they are to each other. (but what about distance ... ... ...)

Why is theta and ej symmetrical for glove but not for the others?

Individual rows can't be assigned indiv rows of english description. -->

### Applications using Word Embeddings

**Sentiment Classification** 
**Definition.** Task of looking at a piece of text and telling if someone likes or dislikes the thing they're talking about.

![RNN sentiment classfication](/assets/img/2019-12-31-coursera-dl-notes/rnn-sentiment-classification.png)

**Debiasing word embeddings**

**Problem.** Man as Computer Programmer and Woman as Homemaker -> Enforces gender stereotype.

**Solution.**
1. Identify bias direction (ie \$ e_{he} - e_{she} \$)
2. Neutralize: For every word that is not definitional (ie doctor/nurse), project to get rid of bias. (ie shift towards non-bias direction)

![Neutralize Bias](/assets/img/2019-12-31-coursera-dl-notes/neutralize-bias.png)

$$
\begin{aligned} e^{\text{bias-component}}&=\frac{e \cdot g}{\|g\|_{2}^{2}} * g \\ e^{\text {debiased}} &=e-e^{\text {bias-component}} \end{aligned}
$$

> \$ e^{bias_component} is the projection of e onto the direction g.

1. Equalize pairs of words that you might want to have differ only through the bias property (st they are equidistant from the non-bias axis)

> As a concrete example, suppose that "actress" is closer to "babysit" than "actor." By applying neutralizing to "babysit" we can reduce the gender-stereotype associated with babysitting. But this still does not guarantee that "actor" and "actress" are equidistant from "babysit." The equalization algorithm takes care of this.

**Confusion Matrix**
1. Printing the confusion matrix can also help understand which classes are more difficult for your model.
2. A confusion matrix shows how often an example whose label is one class ("actual" class) is mislabeled by the algorithm with a different class ("predicted" class).

![Confusion Matrix](/assets/img/2019-12-31-coursera-dl-notes/confusion-matrix.png)

## Various Sequence to Sequence Architectures
![Language Model vs Machine Translation](/assets/img/2019-12-31-coursera-dl-notes/language-model-vs-machine-translation.png)

**Language Model vs Machine Translation**
1. Input is different (Random initialisation vs Conditional Language)
2. Probability of output \$ y^{\langle i \rangle} \$ vs Probability of output \$ y^{\langle i \rangle} \$ conditioned on input \$ x^{\langle i \rangle} \$ vs 
3. Sample \$ y^{\langle i \rangle} \$ randomly vs Deterministic output (intuitively, consistent translation) by taking the max **joint** probability (as opposed to the max prob of each word - greedy search)

**Beam Search Algorithm**
Instead of taking the max prob of each word, beam search takes the max joint probability conditioned on the input x

$$
\arg \max _{y} \prod_{t=1}^{T_{y}} P\left(y^{<t>} | x, y^{<1>}, \ldots, y^{<t-1>}\right)
$$

In order to take the max joint prob, it needs to store the words with highest probability at each step, or Beam Width \$ \beta \$.

**Definition.** Beam Width stores a predetermined number, \$ \beta \$, of best states at each level.

> If \$ \beta \$ = 1, beam search = greedy search.

> \$ \beta \$ = 3 in the example below.

![Beam Search Algo](/assets/img/2019-12-31-coursera-dl-notes/beam-search-algo.png)

Beam Width \$ \beta \$: large \$ \beta \$ results in better result, but slower (as you need to keep more words in memory)

**Refinements to Beam Search**
The joint probability is a product of probabilities which are much less than 1, thus it is a very tiny number that can result in a numerical underflow.

**Definition.** Number is too small to be represented by a float.

$$
\arg \max _{y} \prod_{t=1}^{T_{y}} P\left(y^{<t>} | x, y^{<1>}, \ldots, y^{<t-1>}\right)
$$

A neat trick is to insert \$ log \$, such that it'll be a sum of log probabilities.

$$
\arg \max _{y} \sum_{y=1}^{T_{y}} \log P\left(y^{<t>} | x, y^{<1>}, \ldots, y^{<t-1>}\right)
$$

Doing so will likely give you the same result. This is because \$ log(P(y \mid x)) \$ is a non-decreasing monotone graph, thus a value of y that maximises \$ P(y \mid x) \$ will also maximise \$ log(P(y \mid x)) \$.

Lastly, most of the terms are in the range 0 < x < 1, thus they are exponentially negative. Thus the more terms there are, the more negative the joint probability. To nullify this, we can **normalise the joint probability**, which reduces the penalty of outputting longer translations.

$$
\frac{1}{T_y^{\alpha}}\arg \max _{y} \sum_{y=1}^{T_{y}} \log P\left(y^{<t>} | x, y^{<1>}, \ldots, y^{<t-1>}\right)
$$

where \$ \alpha \$ is a parameter (often tuned to 0.7, but with no theoretical justification).

**Error Analysis in Beam Search** <br />
**Motivation.** Does the error lie with Beam Search or RNN?

Assume that \$ y^* \$ is the better sentence than \$ \hat{y} \$, and the algorithm wrongly chose \$ \hat{y} \$.

![Error Analysis on Beam Search](/assets/img/2019-12-31-coursera-dl-notes/error-analysis-beam-search.png)

Repeat this on the errors of the deveopment set, and figure out what fraction of errors are "due to" beam search vs RNN model.

**Bilingual Evaluation Understudy (Bleu) Score** <br />
Automatic evaluation of machine translation (instead of having a human to do it)

Probability for n-gram

> **n-grams:** n number of words appearing next to each other

$$
P_n = \frac{\sum_{\text{n-grams in y}}Count_{clip}(n-gram)}{\sum_{\text{n-grams in y}}Count(n-gram)}
$$

> Count-clip: Clips the specific part of the sentence. If countclip(1-gram), it'll return the individual words of the sentence.

Combined Bleu Score:

$$
BP\exp^{\frac{1}{k}\sum_{n=1}^{k}p_n}
$$

> Brevity Penalty: Penalises translations which are too short (easier to have higher precision with shorter translation)

**Attention Model Intuition** <br />
**Intuition.** An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. (similar to how a human would translate portion by portion of the sentence, instead of translating the entire sentence.)

$$
\alpha^{<t, t^{\prime}>}=\text { amount of attention } y^{<t>} \text { should pay to  } a^{<t^{\prime}>}
$$

$$
\alpha^{<t, t^{\prime}>}=\frac{\exp \left(e^{<t, t^{\prime}>}\right)}{\sum_{t^{\prime}=1}^{T_{x}} \exp \left(e^{<t, t^{\prime}>}\right)}
$$

The terms \$ e(t, t') \$ are passed into a softmax layer to ensure that for every fix value of t, \$ \alpha^{\langle t, t^{\prime} \rangle} \$ will sum to one. 

The terms \$ e(t, t') \$ are computed from \$ s^{\langle t - 1 \rangle} \$ and \$ a^{\langle t^{\prime} \rangle} \$.

![Attention Model](/assets/img/2019-12-31-coursera-dl-notes/attention-model.png)

### Speech Recognition - Audio Data
**Problem.** Given input audio clip *x*, output transcript *y*.

> Audio clips are plots with the air pressure against time. Human ear doesn't process raw wave forms, but has physical structures that measures the amounts of intensity of different frequencies.

![CTC cost speech recognition](/assets/img/2019-12-31-coursera-dl-notes/ctc-cost-speech-recognition.png)

**Trigger Word Detection** <br />
![Trigger Word Detection Algo](/assets/img/2019-12-31-coursera-dl-notes/trigger-word-algo.png)

> E.g.: Apple Siri, Google Home

Implementation Details:
1. Dataset usually comprises of synthesised data: keyword overlayed with background noises. Because speech data is hard to acquire and label, synthesizing your training data using the audio clips of activates, negatives, and background is easier.
2. Unidirectional RNN rather than a bidirectional RNN.
This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said.


