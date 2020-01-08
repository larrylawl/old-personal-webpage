---
layout: post
title: "Interpretation for Matrix Multiplication and Linear Transformation"
author: "Larry Law"
categories: articles
tags: [linear-algebra]
image: linear-algebra.jpg
---
Without loss of generality to \$ R_n \$, let us consider \$ R_2 \$.

## Prerequisite: Basis Vectors

Suppose I have two vectors, \$ \overrightarrow{v_i} \$ and \$ \overrightarrow{v_j} \$. So long as these vectors are linearly independent, any vector can be represented as the linear combination of these two vectors. These vectors are also defined as the basis vectors of a vector space \$ V \$.

In the "typical" world, the basis vectors are \$ \overrightarrow{e_x} = (0, 1) \$, \$ \overrightarrow{e_y} = (1, 0) \$ in \$ R_2 \$. These special vectors are also called the standard basis.

## Interpretation of Matrix Multiplication

Now suppose I have any vector \$ \overrightarrow{x} \$ \$ \in \$ in \$ R_2 \$ and is represented by my basis vectors \$ \overrightarrow{v_i} \$ and \$ \overrightarrow{v_j} \$.

**How will \$ \overrightarrow{x} \$ look like in the standard basis?**

\$ \overrightarrow{x} \$ in the standard basis will equal the summation of the scalar multiple of \$ x_1 \$ with \$ \overrightarrow{v_i} \$ and the scalar multiple of \$ x_2 \$ with \$ \overrightarrow{v_j} \$. Notice that this is precisely the definition of matrix multiplication.

$$
T(\overrightarrow{v}) :=
\begin{bmatrix}
i_1 & j_1 \\
i_2 & j_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
i_1 \\
i_2
\end{bmatrix} x_1 +
\begin{bmatrix}
j_1 \\
j_2
\end{bmatrix} x_2
=
\begin{bmatrix}
i_1x_1 & j_1x_2 \\
i_2x_1 & j_2x_2
\end{bmatrix}
$$

## Interpretation of Linear Transformation

Matrix multiplication is the main example of a linear transformation. This linear transformation _moves_ \$ \overrightarrow{v} \$ from the representation of the basis vectors in the coefficient matrix to the standard basis. Parsing the term Linear Transformation will help us understand it better.

**Transformation:** A function that takes in an input vector and _moves_ it.
**Linear:** The _movement_ described above is linear if and only if

1. the origin remains
2. lines stay as lines. (In other words, the set of vectors that have the same span remains the same after the transformation)

Notice how matrix multiplication does not violate the two requirements of linear transformation

## Interpretation of Inverse Matrix Multiplication

If _A_ is a _function_ that moves the input vector from the representation of the basis vectors in A to the standard basis, then \$ A^{-1} \$, being the _inverse function_, will do the opposite: move the input vector from the standard basis back to the basis vectors in A.

<!-- omit in toc -->
## Credits
For a more detailed explanation, check out 3Blue1Brown's video [here](https://www.youtube.com/watch?v=kYB8IZa5AuE).
