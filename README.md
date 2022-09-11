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
> pip install dill finta termcolor pyyaml func_timeout scipy statsmodels hyperopt
```

## Hypothesis 

Compared to traditional assets, cryptocurrencies is more inclined 
to co-move due to some common driven forces. Therefore, they could 
be good candidates for a pairs trading strategy, which is based on 
exploiting mean reversion in prices of securities.

Suppose `S1` and `S2` are the prices of two securities. In a training
window, the linear regression of the logarithmic prices gives:

$$\log(S_1) = \gamma\cdot\log(S_2) + \mu + \epsilon$$

The residual $\epsilon$ is expected to exhibit mean-reverting
properties. If so, a pairs trading strategy can be implemented as
follows: when the latest $\epsilon$ exceeds 
$\text{mean}(\epsilon) + \delta\cdot\text{std}(\epsilon)$, 
security 1 is over-valued, and security 2 is under-valued, 
therefore we open short position for security 1, and
long position for security 2; when the latest $\epsilon$ is
less than 
$\text{mean}(\epsilon) - \delta\cdot\text{std}(\epsilon)$, 
security 1 is under-valued, and security 2 is over-valued,
as a result we open long position for security 1, and
short position for security 2. The parameter $\delta$
here is a threshold that should be measured with
simulation data.

To control the risk, we also need to apply the stop
loss rule to the strategy: when we are long security 1, and
short security 2, $\epsilon$ does not mean-revert to 
its historical mean $\text{mean}(\epsilon)$, but
deviates further to be even smaller than
$\text{mean}(\epsilon) - m\delta\cdot\text{std}(\epsilon)$,
where $m$ is a multiple which is usually larger than 1. In
this case, we will close the position even if we
have to realize the loss. Similarly, when we are
short security 1, and long security 2, $\epsilon$ does not 
mean-revert to its historical mean $\text{mean}(\epsilon)$,
but moves further to be even larger than
$\text{mean}(\epsilon) + m\delta\cdot\text{std}(\epsilon)$,
we will also close the position which means we will
realize the loss in the exisitng position. Similar to
$\delta$, the parameter $m$ is also a threshold that needs
to be measured with simulation data.


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

As a naive configuration for the strategy, we set
$\delta=2$ (`entry_threshold=2`), and $m=4$ 
(`exit_threshold=4`). Note that these two parameters
are supposed to be fine tuned with the backtest data.
We also assume for each trading oppotunity, 
the capital allocated to each security is USD 1 million
(`capital_per_entry=1000000`). And we only enter
the trade once for repeating signals 
(`max_number_of_entry=1`). Since there are six pairs
of cryptocurrencies, the number of combination is
$C^2_6 = 15$, the capital required for this 
strategy is $15$ million.

A summary of the strategy parameters is shown as below:

```json
"lookback_period": 960,
"correlation_threshold": 0.8,
"recalibration_interval": 5,
"cointegration_pvalue_threshold": 0.1,
"entry_threshold": 2,
"exit_threshold": 4,
"max_number_of_entry": 1,
"capital_per_entry": 1000000
```
Below is the Backtest result from 2021-01-01 to 2021-12-31: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/pnl_01.jpeg "pnl_01")

As can be seen, there is a significant drawdown and quick rebounce 
on 13/14 Sep. The strategy opened new positions for most of the 
currency pairs with `LTCUSD` almost immediately after hitting 
the stop loss. We can examine this by viewing the normalized
prices of all six cryptocurrency pairs. As expected, there was
indeed observable turbulence in `LTCUSD` during 13 - 14 Sep.
It is this abnormal price movement that makes the drawdown
and rebounce.

![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/closes_Sep11_Sep14.jpeg "closes_sep")