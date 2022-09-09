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

Compared to traditional assets, cryptocurrencies is more inclined 
to co-move due to some common driven forces. Therefore, they could 
be good candidates for a pairs trading strategy, which is based on 
exploiting mean reversion in prices of securities.

Suppose `S1` and `S2` are the prices of two securities. In a training
window, the linear regression of the logarithmic prices gives:

$$\log(S_1) = \gamma\cdot\log(S_2) + \mu$$

# Simulation Results

As discussed in EDA，the trading universe is six cryptocurrency 
pairs:`BTC.USD`, `EOS.USD`, `ETH.USD`, `LTC.USD`, `TRX.USD`, 
and`XRP.USD`.

We use 15-minute OHLCV for each pair, with a look-back 
window of 10 days (`lookback_period=960` bars). In this training period, we 
use a correlation method to determine the possible
trading pairs. Only those pairs with a correlation higher
than the threshold (`correlation_threshold=0.8`) will be shortlisted. The trading
window is 5 days immediately following the previous 10 days
training period. When the trading period completes, the 
dynamic rolling window will be automatically shifted 5 days ahead
for the next training and trading periods (`recalibration_interval=5`).
In the trading period, a rigorous co-integration test is employed
to find out the cointegrated pairs, only the pairs with
a p-value in cointegration test lesser than 0.1 will
be qualified for trading (`cointegration_pvalue_threshold=0.1`). 

The trading rules are as follows: to open a

  "lookback_period": 960,
  "correlation_threshold": 0.8,
  "recalibration_interval": 5,
  "cointegration_pvalue_threshold": 0.1,
  "entry_threshold": 2,
  "exit_threshold": 4,
  "max_number_of_entry": 1,
  "capital_per_entry": 1000000

![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/pnl_01.jpeg "pnl_01")
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/closes_Sep11_Sep14.jpeg "closes_sep")