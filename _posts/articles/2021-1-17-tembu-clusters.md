---
layout: post
title: "Using Tembusu Clusters"
author: "Larry Law"
categories: articles
tags: [Networking]
image: cs.png
---

## Crash Course on using Tembusu Clusters
*"I google this everytime. Related comic."* -Top comment on stackoverflow
![Comic](/assets/img/2021-1-17-tembusu-clusters/comic.png)

### Logging In
```
# VPN or when using NUS wifi
ssh larrylaw@xgpc2.comp.nus.edu.sg

# Otherwise through sunfire
ssh larrylaw@sunfire.comp.nus.edu.sg
ssh xgpc2
```

> Guide to use [SoC VPN](https://dochub.comp.nus.edu.sg/cf/guides/network/vpn). Compute cluster hardware configuration [here](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware).

### Transfering Data
```
# From tembusu cluster to local 
## VPN or when using NUS wifi
scp -r larrylaw@xgpc2.comp.nus.edu.sg:~/NM2/results/exp-e ./

## Otherwise through sunfire
scp -r larrylaw@sunfire.comp.nus.edu.sg:~/net_75 .
scp -r results/rs-obs/net_75/ larrylaw@sunfire.comp.nus
.edu.sg:~/
```

### Development
1. `pyenv` for python version and `pyvenv` for virtual environment
2. `tmux` to run parallal processes
3. `nvidia-smi` to check GPU usage (before sending jobs)




