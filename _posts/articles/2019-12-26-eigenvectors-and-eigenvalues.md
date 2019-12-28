---
layout: post
title: "Eigenvectors and Eigenvalues"
author: "Larry Law"
categories: article
tags: [Linear Algebra]
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

With the eigenvalues, we can proceed to solve for eigenvectors. Concrete steps can be found [here](https://www.scss.tcd.ie/Rozenn.Dahyot/CS1BA1/SolutionEigen.pdf).

## Eigenbasis
An eigenbasis is a **basis wherein every vector is an eigenvector**. 

<!-- omit in toc -->
## Credits
3Blue1Brown's video on Eigenvectors and Eigenvalues. Source [here](https://www.youtube.com/watch?v=PFDu9oVAE-g&t=887s).
