# Demo Strategy

---
author: Joseph Chen\
date: Sep 14th, 2022
---

This demo will show how to build a practical strateqy 
(pairs trading) from scratch in the framework of 
[`QTrader`](https://github.com/josephchenhk/qtrader).

## Python environment

Use conda to manage your environment. The following code
creating a virtual environment named `demo_strategy` and
installing the relevant packages:

```shell
> chmod +x ./preparations.sh 
> ./preparations.sh
```

Or you can manually input the following commands:

```shell
> conda create -n demo_strategy python=3.8
> conda activate demo_strategy
> pip install git+https://github.com/josephchenhk/qtrader@master
> pip install dill finta termcolor pyyaml func_timeout scipy statsmodels hyperopt jupyter seaborn
```

## EDA and Data Preparations

Refer to notebook `EDA.ipynb` for exploratory data analysis. After
running the notebook, you should have prepared a data folder named
`clean_data` in your current working directory. We will use this 
dataset to backtest our model.

## Model Description 

Cryptocurrencies are more inclined to co-move due to some 
common driven forces than traditional assets. Therefore, they 
be good candidates for a pairs trading strategy, which is based on 
exploiting mean reversion in prices of securities.

Suppose `S1` and `S2` are the prices of two securities. In a training
window, the linear regression of the logarithmic prices gives:

$$\log(S_1) = \gamma\cdot\log(S_2) + \mu + \epsilon$$

Given the data window, the slope ($\gamma$) and intercept
($\mu$) could be obtained through linear regression.
The residual term $\epsilon$ is the spread 
$s = \log(S_1) - \gamma\cdot\log(S_2)$ which is expected 
to exhibit mean-reverting
properties. If so, a pairs trading strategy can be implemented as
follows: when the latest spread $s$ exceeds 
$\text{mean}(s) + \delta\cdot\text{std}(s)$, 
security 1 is over-valued, and security 2 is under-valued, 
therefore we open 1 unit of short position for security 1, and
$\gamma$ unit of long position for security 2; when the 
latest $s$ is less than 
$\text{mean}(s) - \delta\cdot\text{std}(s)$, 
security 1 is under-valued, and security 2 is over-valued,
as a result we open 1 unit of long position for security 1, 
and $\gamma$ unit of short position for security 2. The 
parameter $\delta$ here is a threshold that should be measured 
with simulation data.

To control the risk, we also need to apply the stop
loss rule to the strategy: if we are long security 1, and
short security 2, when $s$ does not mean-revert to 
its historical mean $\text{mean}(s)$, but
deviates further to be even smaller than
$\text{mean}(s) - \Delta\cdot\text{std}(s)$,
where $\Delta$ is a multiple which is usually larger than $\delta$,
we will close the position even if we
have to realize the loss. Similarly, when we are
short security 1, and long security 2, and $s$ does not 
mean-revert to its historical mean $\text{mean}(s)$,
but moves further to be even larger than
$\text{mean}(s) + \Delta\cdot\text{std}(s)$,
we will also close the position which means we will
realize the loss in the existing position. Similar to
$\delta$, the parameter $\Delta$ is also a threshold that needs
to be measured with simulation data.

Besides the z-score condition, there are
also other exit conditions: when a pair is on, 
the time series should remain cointegrated and the 
hedge ratio $\gamma$ should be positive. If any 
of these conditions are violated, the foundation of 
the pairs trading strategy is not valid anymore, 
and the existing position of the pair should be 
cleared. 

# Simulation Results

As discussed in EDA，the trading universe is six cryptocurrency 
pairs:`BTC.USD`, `EOS.USD`, `ETH.USD`, `LTC.USD`, `TRX.USD`, 
and`XRP.USD`.

We use 15-minute OHLCV for each pair, with a look-back 
window of 15 days (`lookback_period=1440` bars). 
In this training period, we use a correlation method to 
determine the possible trading pairs. Only those pairs with 
a correlation higher than the threshold 
(`correlation_threshold=0.8`) will be shortlisted. The trading
window is 4 days immediately following the previous 15 days
training period. When the trading period completes, the 
dynamic rolling window will be automatically shifted 4 days ahead
for the next training and trading periods (`recalibration_interval=4`).
In the trading period, a rigorous co-integration test is employed
to find out the cointegrated pairs, only the pairs with
a p-value in cointegration test lesser than 0.05 will
be qualified for trading (`cointegration_pvalue_entry_threshold=0.05`). 
Once any pair of cryptocurrency is on, the cointegration
condition must be valid, otherwise the position will
be closed (`cointegration_pvalue_exit_threshold=0.15`).

As a naive configuration for the strategy, we set
$\delta=1.5$ (`entry_threshold=1.5`), and $\Delta=2.0$ 
(`exit_threshold=2.0`). Note that these two parameters
are supposed to be fine-tuned with the backtest data.
We also assume for each trading opportunity, 
the maximum capital allocated to individual security 
is USD 1 million (`capital_per_entry=1000000`). And we 
only enter the trade once for repeating signals 
(`max_number_of_entry=1`). Since there are six pairs
of cryptocurrencies, the number of combinations is
$C^2_6 = 15$, the capital required for this 
strategy is $15$ million.

A summary of the strategy parameters is shown below:

```json
"lookback_period": 1440,
"correlation_threshold": 0.80,
"recalibration_interval": 4,
"cointegration_pvalue_entry_threshold": 0.05,
"cointegration_pvalue_exit_threshold": 0.15,
"entry_threshold": 1.5,
"exit_threshold": 2.5,
"max_number_of_entry": 1,
"capital_per_entry": 1000000
```

### Training Period
Below is the Backtest result from 2021-01-01 to 2021-12-31: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/pnl_01.jpeg "pnl_01")

```html
____________Performance____________
Start Date: 2021-01-01
End Date: 2021-12-31
Total Return: 5.60%
Sharpe Ratio: 0.68
Rolling Maximum Drawdown: -4.69%
```

As can be seen, there is a significant drawdown period
in mid Jul to mid Aug. The strategy opened new positions 
for most of the 
currency pairs with `XRPUSD` almost immediately after hitting 
the stop loss. We can examine this by viewing the normalized
prices of all six cryptocurrency pairs. As expected, there was
indeed observable turbulence in `XRPUSD` during 19 Jul - 16 Aug.
It is this abnormal price movement that makes the drawdown
and rebounce subsequently.

![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/closes_jul19_aug16.jpeg "closes_sep")

### Testing Period

Once we optimized parameters in training dataset, we
are ready to test our strategy in testing dataset.
Below is the Backtest result from 2022-01-01 to 2022-08-11: 


## Future Work

There is a lot of work to be done to improve the strategy, which is 
included but not limited to:

- (1). Train the model in a dynamic rolling window, i.e., recalibrate
the parameters `entry_threshold` and `exit_threshold` regularly.
  The code for optimization is in `optimization_pair.py`.

- (2). Add an absolute stop loss to each traded pair.

- (3). Consider the actual volume to have a better estimation of 
executed shares.
  
- (4). Consider a vectorization (dataframe/numpy) implementation 
  of the backtest, to increase the optimization speed. 