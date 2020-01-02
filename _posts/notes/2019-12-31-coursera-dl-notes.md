---
layout: post
title: "Coursera, Stanford: Deep Learning Notes"
author: "Larry Law"
categories: notes
tags: [notes, Deep Learning]
image: neural-network.jpeg
hidden: true
---
Lecturer: Professor Andrew Ng <br>
Course available [here](https://www.coursera.org/specializations/deep-learning).<br>
Course notations available [here](https://d3c33hcgiwev3.cloudfront.net/_106ac679d8102f2bee614cc67e9e5212_deep-learning-notation.pdf?Expires=1578096000&Signature=GpoeRBwFaUWIr5ryWYRyovABkIaqsTUJiplq4Fh-5BSyeHNZ~8hVirdtULblmZFvyYdVcWvZnC-soZilkuXf0rGgMs~uCGjPwoK7TFTcG6l5AaHen-86m-teuS37zFdeR0doeUzqX-jkAlhkqwZjPr7epdFBrnjrgiFp9GZXE7k_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A).


<!-- ## Table of Contents
Week 1:

1. Rectify: Taking a max of 0 which is why you get a function shape like this
2. Online advertising: lucrative app
3. Why sigmoid over relu? 
<!-- Relu function greater than 1? 
https://www.coursera.org/learn/neural-networks-deep-learning/discussions/weeks/1/threads/bJAuE_qXSgqQLhP6l9oKnQ-->


# Course 1: Neural Networks and Deep Learning
## Learning Outcomes
1. Defensive Programming with Matrixes
2. Activation Functions
3. Why do we need non-linear activation functions?

### Defensive Programming with Matrixes
```
a = np.random.randn(5) # rank 1 array - don't use!

a = np.random.randn(5, 1) # a.shape = (5, 1)
a = np.random.randn(1, 5) # a.shape = (1, 5)
assert(a.shape == (5, 1))
```

### Why do we need non-linear activation functions?
![non linear activation function](/assets/img/2019-12-31-coursera-dl-notes/non-linear.png)

Suppose not, the _a_ will essentially be a linear regression, which reduces the problem to a linear regression problem (which can be solved using linear regression).

> ReLU is a piecewise linear function.

