# Demo Strategy

---
author: Joseph Chen\
date: Sep 14th, 2022
---

In this demo, I will show how to build a practical strateqy 
(pairs trading) from scratch in the framework of 
[`QTrader`](https://github.com/josephchenhk/qtrader).

## Python environment

Use conda to manage your environment. The following code
creating a virtual environment named `demo_strategy` and
installing the relevant packages:

```shell
> conda create -n demo_strategy python=3.8
> conda activate demo_strategy
> pip install git+https://github.com/josephchenhk/qtrader@master
> pip install dill finta termcolor pyyaml func_timeout scipy statsmodels
```

## Hypothesis 

Cryptocurrencies tend to co-move due to some common driven forces
in this space,