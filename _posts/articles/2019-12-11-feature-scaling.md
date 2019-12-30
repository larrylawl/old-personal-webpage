---
layout: post
title: "Feature Scaling: why and how"
author: "Larry Law"
categories: articles
tags: [Machine Learning]
image: machine-learning.jpg
---

## Learning outcomes
1. Why do we need feature scaling?
2. How is feature scaling implemented? 

## Why do we need feature scaling?
**To speed up gradient descent**. Recall that the purpose of gradient descent is to _learn our model's parameters._ Suppose that the range of our parameters are uneven. A larger range will lead to a slower descent, while a smaller range will lead to a faster descent. The uneven simultaneous update of our parameters will cause the descent to _oscillate_ which is inefficient. 
> Not sure what is gradient descent? Simultaenous update? Read more in the post about it [here](./gradient-descent.html)

<!-- TODO: Include example -->

## How do we implement feature scaling?
One common way is to **normalise the feature** (ie subtracting the mean and dividing by the standard deviation). 

$$
x_{i}:=\frac{x_{i}-\mu_{i}}{s_{i}}
$$

## Credits
Andrew Ng's Machine Learning course. Source [here](https://www.coursera.org/learn/machine-learning?utm_source=gg&utm_medium=sem&utm_content=93-BrandedSearch-INTL&campaignid=1599063752&adgroupid=58953588605&device=c&keyword=coursera%20courses&matchtype=b&network=g&devicemodel=&adpostion=1t1&creativeid=303554599611&hide_mobile_promo&gclid=EAIaIQobChMIvfCauaSo5gIVF4iPCh1U1gK3EAAYASABEgLY6vD_BwE)
