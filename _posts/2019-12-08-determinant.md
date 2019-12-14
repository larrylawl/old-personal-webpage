---
layout: post
title: "Interpretation for Determinant"
author: "Larry Law"
categories: journal
tags: [Linear Algebra]
image: linear-algebra.jpg
---

(For a more detailed explanation, check out 3Blue1Brown's video [here](https://www.youtube.com/watch?v=Ip3X9LOh2dk&t=57s))

## Interpretation for Determinant

Recall that a linear transformation _moves_ the vector. The determinant computes the **factor by which the space was changed by this linear transformation** (space := area in \$ R_2 \$, volume in \$ R_3 \$).

If the determinant, d, is...

≥0, the linear transformation changes the space by a factor of d.

≤0, the linear transformation _flips_ and changes the space by a factor of d.

= 0, since the factor by which the space was changed is 0, this linear transformation squishes the space to one dimension. Thus the matrix is _non-invertible_, and hence it is _singular_

## Why does this work?

$$ det(AB) = det(A)det(B) $$

AB is equivalent to

1. first applying linear transformation of B, which moves the vector. The factor by which this movement changes space is \$ det(B) \$.
2. then applying linear transformation of A. The factor by which this movement changes space is \$ det(A) \$.

Thus determinant of the linear transformation of B then A equals the multiplication of the individual determinant values.
