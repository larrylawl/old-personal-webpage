---
layout: post
title: "Winning Kaggle Competitions with Deep Learning"
author: "Tao Sheng"
categories: takeaways
hidden: true
---
*(Here's the youtube [link](https://www.youtube.com/watch?v=8YTUpMY2dos))*

# Prepare for a DL Competition
1. GPU
2. Read a lot of papers

# Five Steps to win a DL Competition
1. Understand the data
2. Build a strong baseline
3. Find the tricks
4. Ensemble
5. Pseudo-Labels

## Build a strong baseline
1. No fancy NN architecture and loss function
2. Use momentum SGD or Adam optimiser
3. Proper data augmentation
4. Reliable local validation
   
> **A strong baseline** can get you a high ranked **silver medal.**

> **Build your own strong pipeline, and reuse it!**

> Purpose of the baseline is, without special tricks, results can be ranked as top 20 or 15.

### Tips - optimiser and lr strategy
1. Step lr(0.003) schedule with Adam optimiser
2. Try momentum SGD optimiser
3. Try lr warmup
4. Try consine annealing lr / cyclic lr

### Tips - Batch norm
1. Inconsistency between train and test

### The key to win -- find the tricks
1. Task specific trick
    1. Tricks for image classification / object detection. Need to read papers!
2. Data specific trick
    1. Error analysis

### Ensemble
1. Weight average is simple and effective
2. Use NN to do ensemble: stacking

### Final boost -- Pseudo labels
1. Easy to use and works in almost every DL competition
2. Use it in the final stage - overfit the LB then create pseduo labels for testset or external unlabeled data
