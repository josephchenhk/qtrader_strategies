# Systematic Strategies

---
author: Joseph Chen\
date: Oct 4th, 2022
---

This demo will show how to build practical strateqies 
(e.g., pairs trading) from scratch in the framework of 
[`QTrader`](https://github.com/josephchenhk/qtrader).

## Python environment

It is recommended to use conda to manage your environment. 
The following code creating a virtual environment named 
`demo_strategy` and installing the relevant packages:

```shell
> chmod +x ./preparations.sh 
> ./preparations.sh
```

Or you can manually input the following commands:

```shell
> conda create -n demo_strategy python=3.8
> conda activate demo_strategy
> pip install --force-reinstall git+https://github.com/josephchenhk/qtrader@master
> pip install --force-reinstall git+https://github.com/josephchenhk/qtalib@main
> pip install dill finta termcolor pyyaml func_timeout scipy statsmodels hyperopt jupyter seaborn
```

## EDA and Data Preparations

Refer to notebook `EDA.ipynb` for exploratory data analysis. After
running the notebook, you should have prepared a data folder named
`clean_data` in your current working directory. We will use this 
dataset to backtest our model.