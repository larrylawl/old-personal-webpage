---
layout: post
title: "Heuristic Justifying Random Initialisation In Neural Networks"
author: "Larry Law"
categories: journal
tags: [Machine Learning]
image: neural-network.jpeg
---
## Learning outcomes
1. Purpose for Random Initialisation in Neural Networks
2. Heuristic Justifying (1)

## Purpose for Random Initialisation in Neural Networks
In Backpropagation, parameters are randomly initialised in order to avoid the case wherein they are _symmetrical_ (aka _breaking symmetry_). More concretely, they are symmetrical in the sense that all parameters of a particular layer _l_ are the same.

> Parameters of different layers need not necessarily be the same for the symmetric property to hold.

### What is wrong with symmetrical initialisation?

The claim is that _the symmetrical property largely persists even after backpropagation_. Assuming that this claim holds, for all units of layer _l_, \$ \Theta^{(l - 1)} \$ is the same (by assumption) and \$ a^{(l - 1)} \$ is the same as well (by the neural networks model). Thus \$ a^{(l)} \$ will be the same, which makes the units of layer _l_ **redundant.**

Let us look at the heuristic justifying the claim made above, and explain the qualifier _"largely"_ later.

## Heuristic Justifying Random Initialisation
Suppose symmetrical intialisation of parameters.

Let's find out the effect of backpropagation on the parameters using _first-step analysis_. That is, we analyse the effect of one iteration, and make a generalised conclusion for _n_ iterations from there.

Recall that the Backpropagation learns parameters using the partial derivative of the cost function with respect to that parameter. More concretely,

$$
\frac{\partial{J(\Theta)}}{\partial \Theta^{(l)}} =a^{(l)} \delta^{(l+1)}, \delta \text{ to be defined}
$$

Since the parameters are the same in the first iteration, \$ a^{(l)} \$ will be the same for all units in layer _l_. Let us take a look at \$ \delta \$, which is also known as the "error term".

$$
\\ \delta^{(l+1)} := \frac{\partial{J(\Theta)}}{\partial z^{(l+1)}} = \delta^{(l+2)} \Theta^{(l+1)} g^{\prime}\left(z^{(l+1)}\right),
\\ \delta^{(L)} = a^{(L)} - y,
$$

1. _Suppose that the neural network has only 1 output._ \$ \delta^{(L)} \$ will thus be a real number. 
2. \$ \Theta^{(l+1)} \$ is the same (by assumption) 
3. Derivative of activation function `g` is the same across all layers. As explained earlier, z of any layer _l_ will be the same. Thus this derivative term will be the same.
4. From (1) - (3), all error terms of layer _l_ will be the same.

Putting it all together, we have shown that the partial derivative for units of the same layer will be the same. Consequently, the output from the parameter learning function will be the same, thus the symmetrical property persists after _1_ iteration. Since the parameters are still symmetrical, we can expect them to still be symmetric after _n_ iterations.

## Explaining the qualifier - "largely persists"
We have made an assumption that the neural network only has 1 output. What if there are more than 1 output? \$ \delta^{(L)} \$ will now be a matrix, with all entries being the same except for the entry wherein y = 1. 

However, since all other entries remain the same, the symmetrical property _largely_ persists after Backpropagation.

## Credits
Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning).
