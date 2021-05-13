---
layout: post
title: "Eigenvectors and Eigenvalues"
author: "Larry Law"
categories: articles
tags: [linear-algebra]
image: linear-algebra.jpg
---
<div align="center">
    <i>"Eigen: a german translation of 'Own'"</i>
</div>

<!-- omit in toc -->
## Learning outcomes
- [Eigenvectors and Eigenvalues](#eigenvectors-and-eigenvalues)
- [Intuition for Solving for Eigenvalues](#intuition-for-solving-for-eigenvalues)
- [Eigenbasis](#eigenbasis)
- [Application: Matrix Diagonalisation](#application-matrix-diagonalisation)

## Eigenvectors and Eigenvalues
Recall that linear transformation is a function that moves a vector from its representation in its coefficient matrix to the standard basis.

$$
Av = \lambda  v
$$

If the linear transformation of this vector _v_ **retains its span** then _v_ is called an __eigenvector__.

> Recall that span of a vector represents all possible linear combinations of a vector.

Since this transformation of _v_ retains its span, it is scalar transformation, \$ \lambda \$. **The factor by which the vector is scaled** is the **eigenvalue.**

## Intuition for Solving for Eigenvalues

$$
Av = \lambda  v \\
Av - I\lambda  v = 0 \\
(A - I\lambda) v = 0
$$

In order for the above linear transformation to hold, the determinant of the coefficient matrix must equals 0.

$$ 
det(A - I\lambda) = 0
$$

This is because the determinant computes the **factor by which the space was changed by this linear transformation**. Thus if determinant is zero, the factor by which the space was changed will be 0, which equals the RHS. 

> Read more about determinants in my post [here.](./determinant.html)

Solving for the determinant of the coefficient matrix gives us our eigenvalues.

With the eigenvalues, we can proceed to solve for eigenvectors.

## Eigenbasis
An eigenbasis is a **set of basis vectors wherein every vector is an eigenvector**.

## Application: Matrix Diagonalisation
Matrix diagonalisation is the process of taking a square matrix and converting it to a diagonal matrix.

$$
P^{-1}AP = D,
$$

where _A_ is a square matrix, _P_ is an invertible matrix composed of the eigenvectors of A, and D is a diagonal matrix containing the corresponding eigenvalues.

**Why is matrix diagonalisation useful?**

Consider the case where we want to find \$ A^{100} \$. Suppose _A_ is not a diagonal matrix, doing matrix multiplication directly will be tedious. However, matrix diagonalisation allows us to convert A to a diagonal matrix.

$$
A = PDP^{-1} \\
A^{100} = PD^{100}P^{-1}
$$

Computing the power of a diagonal matrix is simply computing the power of each diagonal entries.

**So why does matrix diagonalisation work?**

$$
P^{-1}AP = D
$$

Here's the proof sketch:
1. *AP* represents a scalar transformation of any vector as *P* is chosen to use the eigenvectors of A. The scalar values will be the corresponding eigenvectors.
2. When we premultiply this scalar transformation with \$ P^{-1} \$, we can take the scalar values out, thus it'll be equivalent to a matrix multiplication of the scalar matrix and identity matrix (ie \$ ZP^{-1}P = ZI \$, where *Z* denote the scalar matrix).
3. This matrix multiplication will return precisely the diagonal matrix, with each entry denoting the corresponding eigenvector.

> The set of basis vectors which we can choose _P_ from is also called the **eigenbasis**

Formal proof of diagonalisation is beautifully explained by ST. Lawrence [here](https://www.youtube.com/watch?v=D93lYjmR-7A).

<!-- omit in toc -->
## Credits
3Blue1Brown's video on Eigenvectors and Eigenvalues. Source [here](https://www.youtube.com/watch?v=PFDu9oVAE-g&t=887s).
