---
layout: post
title: "Intuition for Linear Independence"
author: "Larry Law"
categories: articles
tags: [linear-algebra]
image: linear-algebra.jpg
---

## Learning Outcomes
1. Intuition for Linear Independence
2. How (1) fits the definition of Linear Independence
3. How to determine if a set of vectors are linearly independent

## Intuition for Linear Independence
Consider the vectors 

$$
\quad \vec{v}_{1}=\left(\begin{array}{c}{1} \\ {0} \\ {0}\end{array}\right) 
\quad \vec{v}_{2}=\left(\begin{array}{c}{0} \\ {1} \\ {0}\end{array}\right) 
\quad \vec{v}_{3}=\left(\begin{array}{c}{0} \\ {0} \\ {1}\end{array}\right)
$$

Without \$ v_1 \$ which captures the x-axis, the linear combination of \$ v_2 \$ and \$ v_3 \$ can never express a vector in the x-axis (other than the origin). Likewise for \$ v_2 \$ (y-axis) and \$ v_3 \$ (z-axis).

Each vector in this example captures a unique _direction_ that cannot be expressed as a linear combination of the other vectors.

## How the intuition fits the definition
More formally, a set of vectors is defined as linearly independent if and only if **for any vector in the set, it cannot be expressed as linear combination of the other vectors in the set.**

More concretely, this set of vectors are linearly independent if and only if **the only scalars that satisfy the equation below are all zeros. (ie \$ c_1 = c_2 = ... = c_n = 0\$)**

$$
c_1v_1 + c_2v_2 + ... + c_nv_n = [v_1, v_2, ..., v_n][c_1,c_2, ..., c_n]^{\top} = 0
$$

## How to determine if a set of vectors are linearly independent

First, perform Gaussian Jordan Elimination to solve for the scalars (or matrix) c.
> The scalars can be expressed as a matrix. 

Consider the example below where there are no zero rows. Observe that from the 3rd equation, \$ c_3 = 0 \$. Thus \$ c_2 = 0 \$, and hence \$ c_1 = 0 \$. Since the only scalars that satisfy the equation are all zeroes, this set of vectors is linearly independent.

![linear-independence](/assets/img/2019-12-14-linear-independence/linearly-independent.jpg)

Next consider the example below where there is a zero row. \$ c_3 \$ can take on any real value. Since there are non zero scalars that satisfy the equation, this set of vectors is not linearly independent.

![linear-dependence](/assets/img/2019-12-14-linear-independence/linearly-dependent.jpg).

More generally, **the maximum number of linearly independent vectors is equal to the number of non-zero rows.**

## Credits
Cliff notes for the examples. Source [here](https://www.cliffsnotes.com/study-guides/algebra/linear-algebra/real-euclidean-vector-spaces/linear-independence).
