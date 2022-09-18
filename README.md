# Demo Strategy

---
author: Joseph Chen\
date: Sep 14th, 2022
---

This demo will show how to build a practical strateqy 
(pairs trading) from scratch in the framework of 
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
> pip install dill finta termcolor pyyaml func_timeout scipy statsmodels hyperopt jupyter seaborn
```

## EDA and Data Preparations

Refer to notebook `EDA.ipynb` for exploratory data analysis. After
running the notebook, you should have prepared a data folder named
`clean_data` in your current working directory. We will use this 
dataset to backtest our model.

## Model Description 

Cryptocurrencies are more inclined to co-move due to some 
common market sentiment than traditional assets. Therefore, they 
could be good candidates for a pairs trading strategy, which 
is based on exploiting the mean reversion in prices of securities.

There are two key things concerning a pairs trading strategy:

* Determine the candidate pairs

* Determine the entry and exit rules

The following paragraphs explain these in details: 

Suppose `S1` and `S2` are the prices of two securities. In the
training window, the linear regression of the logarithmic 
prices give:

$$\log(S_1) = \gamma\cdot\log(S_2) + \epsilon$$

where the slope $\gamma$ represents the hedge ratio.
We can define the residual term $\epsilon$ as the spread 
$s = \log(S_1) - \gamma\cdot\log(S_2)$, which is expected 
to exhibit mean-reverting properties. 

A Two-step method will be used to find out the candidate
pairs. In step 1, for the given lookback window, the 
correlation of the logarithm prices will be calculated, 
and only those with a correlation higher than the threshold
will be selected to enter step 2. In step 2, we will
employ an Augmented Dicky Fuller (ADF) test on the shortlisted
pairs from step 1, and only those with a p-value smaller
than the predetermined threshold will be added to the 
final candidate pool. In the ADF test, linear regression
is carried out and regression coefficients $\gamma$ and
$\mu$ are obtained. These values are assumed to be constant
throughout the testing period. Once the testing period
completes, the training window will move forward to
latest timestamp, and repeat this two-step calculation
to determine candidate pairs in next testing period.

The main assumptions of this strategy could be 
summarized as:

1. The mean-reversion behaviors observed in the training 
period will continue to exist in the testing period, and
the spread will mean-revert to its historical mean.
   
2. Once a candidate pair is determined by the two-step
method, it is valid throughout the next testing period, 
   and the hedge ratio will also remain unchanged.
   
Note that in reality, there is no guarantee for any of
the assumptions above. Violation of the assumptions
could lead to failures of the strategy.

Once we have determined the candidate pairs and their
corresponding parameter $\gamma$, the model
is ready to observe the signals by feeding price information.
A pairs trading 
strategy can be implemented as
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
parameter $\delta$ here is a threshold number that should
be measured with simulation data.

To control the risk, we also need to apply the stop
loss rule to the strategy: if we are long security 1, and
short security 2, when $s$ does not mean-revert to 
its historical mean $\text{mean}(s)$, but instead
deviates further to be even smaller than
$\text{mean}(s) - \Delta\cdot\text{std}(s)$,
where $\Delta$ is a multiplier which is usually larger 
than entry threshold $\delta$,
we will close the position and realize the loss. 
Similarly, when we are short security 1, long security 2, 
and $s$ does not mean-revert to its historical mean 
$\text{mean}(s)$, but moves further to be even larger than
$\text{mean}(s) + \Delta\cdot\text{std}(s)$,
the position will be closed and a loss will be realized. 
Similar to $\delta$, the exit threshold $\Delta$ is also 
a number that needs to be found out with training data.

Besides the z-score condition, there are
also other entry conditions: when a pair is on, 
the hedge ratio $\gamma$ should be positive. This
is to ensure that we always have a market-neutral
position, i.e., long position in one security
and short position in another. And we also 
close existing positions and avoid
opening new position at the end of the testing
period.

# Simulation Results

As discussed in EDA，the trading universe is six cryptocurrency 
pairs:`BTC.USD`, `EOS.USD`, `ETH.USD`, `LTC.USD`, `TRX.USD`, 
and`XRP.USD`.

The OHLCV data of different intervals (5-min, 15-min, and 30-min)
are used for simulations. The look-back 
window is fixed to be 960 bars (`lookback_period=960`). 
In the training period, we apply a two-step 
statistical method to the data in lookback window to 
determine the candidate pairs. Only those pairs with 
a correlation higher than the threshold 
(`correlation_threshold=0.8`) and an 
ADF p-value less than the threshold 
(`cointegration_pvalue_entry_threshold=0.1`) will be shortlisted. 
The trading
window is next 480 bars (`recalibration_interval=480`) immediately 
following the previous training period. When the trading period 
completes, the dynamic rolling window will be automatically 
shifted 480 bars ahead
for the next training and trading periods.
In the trading period, the spread 
$s = \log(S_1) - \gamma\cdot\log(S_2)$ is updated
by feeding the new price $S_1$ and $S_2$, and entry and
exit are determined by z-score of the calculated spread.

The entry threshold is defined as anything between 1.5-sigma and 2-sigma
(`1.5 < entry_threshold < 2.0`); and the exit threshold is defined
as anything between 2.5-sigma and 3.5-sigma
(`2.5 < exit_threshold < 3.5`). These parameters will change as per the 
backtesting results and individual security without risking overfitting 
data. We also assume for each trading opportunity, 
the maximum capital allocated to individual security 
is USD 1 million (`capital_per_entry=1000000`). And we 
only enter the trade once for repeating signals 
(`max_number_of_entry=1`). 

A summary of the strategy parameters is shown below:

```json
"lookback_period": 960,
"correlation_threshold": 0.8,
"recalibration_interval": 480,
"cointegration_pvalue_entry_threshold": 0.1,
"entry_threshold": [1.5, 2.0],
"exit_threshold": [2.5, 3.5],
"max_number_of_entry": 1,
"capital_per_entry": 1000000
```

## Optimization Objective Function

The objective of the strategy is to maximize the 
Sharpe ratio and the total return. Therefore, the objective
function is defined as minimizing $f$:

$$
f(threshold_{entry}, threshold\{exit}) 
= -\min(\max(\text{SR}, 0), 0.5) * \text{TOTR}
$$

where $SR$ is the Sharpe ratio, and $TOTR$ is the total
return. 

## 5-min Interval

We firstly test the strategy in a 5-min interval. This
means we have a training window (lookback window) of 80 hours 
($5 * 960 / 60 = 80$), and a testing window (trading window)
of 40 hours.

In the training dataset (in-sample), we trained the model 
and selected the cryptocurrency pairs with negative best loss 
as we are minimizing the objective function. There are 7 
pairs that are selected. And the strategy will be tested on both 
in-sample and out-of-sample datasets.

```html
{
    ('EOS.USD', 'ETH.USD'): {
        'entry_threshold': 1.5000392943615142, 
        'exit_threshold': 3.376602464111785, 
        'best_loss': -0.03475020027151503}, 
    ('TRX.USD', 'XRP.USD'): {
        'entry_threshold': 1.551051140512154, 
        'exit_threshold': 3.017624989640105, 
        'best_loss': -0.08643681591995156}, 
    ('BTC.USD', 'EOS.USD'): {
        'entry_threshold': 1.9633378226348694, 
        'exit_threshold': 3.329967191852652, 
        'best_loss': -0.017663755051963572}, 
    ('EOS.USD', 'LTC.USD'): {
        'entry_threshold': 1.7693316696798091,
        'exit_threshold': 2.5532912245203043, 
        'best_loss': -0.016165476679308892}, 
    ('BTC.USD', 'LTC.USD'): {
        'entry_threshold': 1.8775675045977969, 
        'exit_threshold': 2.670390189458025, 
        'best_loss': -0.007816275748584213}, 
    ('ETH.USD', 'LTC.USD'): {
        'entry_threshold': 1.762856210892485, 
        'exit_threshold': 2.707378771966564, 
        'best_loss': -0.014213080160328405}, 
    ('BTC.USD', 'ETH.USD'): {
        'entry_threshold': 1.9332887682355882, 
        'exit_threshold': 3.407991025881241, 
        'best_loss': -0.005658161919799375}
}
```

### In-sample
Below is the Backtest result from 2021-01-01 to 2021-12-31: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/5min_in_sample.jpeg "5min_in_sample")

```html
____________Performance____________
Start Date: 2021-01-01
End Date: 2022-01-01
Number of Trading Days: 365
Number of Instruments: 7
Number of Trades: 168
Total Return: 6.38%
Annualized Return: 6.38%
Sharpe Ratio: 0.96
Rolling Maximum Drawdown: -5.39%
```

### Out-of-sample

Below is the Backtest result from 2021-01-01 to 2022-01-01: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/5min_out_of_sample.jpeg "5min_out_of_sample")

```html
____________Performance____________
Start Date: 2022-01-01
End Date: 2022-08-01
Number of Trading Days: 212
Number of Instruments: 7
Number of Trades: 89
Total Return: 1.82%
Annualized Return: 3.14%
Sharpe Ratio: 0.82
Rolling Maximum Drawdown: -3.00%
```

## 15-min Interval

We then test the strategy in a 15-min interval. This
means we have a training window (lookback window) of 10 days 
($15 * 960 / (60 * 24) = 10$), and a testing window (trading window)
of 5 days.

As discussed, there are 5 pairs that 
are selected. And the strategy will be tested on both 
in-sample and out-of-sample datasets.

```html
{
    ('EOS.USD', 'LTC.USD'): {
        'entry_threshold': 1.85083364536054, 
        'exit_threshold': 3.224360323840364, 
        'best_loss': -0.047703687981827114}, 
    ('EOS.USD', 'XRP.USD'): {
        'entry_threshold': 1.8869138038036657, 
        'exit_threshold': 2.9094095009860723, 
        'best_loss': -0.046187517027634906}, 
    ('BTC.USD', 'EOS.USD'): {
        'entry_threshold': 1.8767472177155844, 
        'exit_threshold': 2.6226223785191993, 
        'best_loss': -6.446680549378299e-05}, 
    ('EOS.USD', 'ETH.USD'): {
        'entry_threshold': 1.603040067942517, 
        'exit_threshold': 3.4874437489605867, 
        'best_loss': -0.10111409247066716}, 
    ('TRX.USD', 'XRP.USD'): {
        'entry_threshold': 1.5092092690764671,
        'exit_threshold': 2.912104597010566, 
        'best_loss': -0.0025735030462288046}
}
```

### In-sample
Below is the Backtest result from 2021-01-01 to 2021-12-31: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/15min_in_sample.jpeg "15min_in_sample")

```html
____________Performance____________
Start Date: 2021-01-01
End Date: 2022-01-01
Number of Trading Days: 365
Number of Instruments: 5
Number of Trades: 33
Total Return: 8.14%
Annualized Return: 8.14%
Sharpe Ratio: 1.20
Rolling Maximum Drawdown: -4.92%
```

### Out-of_sample

Below is the Backtest result from 2021-01-01 to 2022-01-01: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/15min_out_of_sample.jpeg "15min_out_of_sample")

```html
____________Performance____________
Start Date: 2022-01-01
End Date: 2022-08-01
Number of Trading Days: 212
Number of Instruments: 5
Number of Trades: 13
Total Return: -8.47%
Annualized Return: -14.59%
Sharpe Ratio: -1.38
Rolling Maximum Drawdown: -11.29%
```

## 60-min Interval

We then test the strategy in a 60-min interval. This
means we have a training window (lookback window) of 40 days 
($60 * 960 / (60 * 24) = 40$), and a testing window (trading window)
of 10 days.

As discussed, there is one pair that 
are selected. And the strategy will be tested on both 
in-sample and out-of-sample datasets.

```html
{
    ('BTC.USD', 'LTC.USD'): {
        'entry_threshold': 1.8959144645762966, 
        'exit_threshold': 2.9436715640836755, 
        'best_loss': -0.08387009026604986}
}
```

### In-sample
Below is the Backtest result from 2021-01-01 to 2021-12-31: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/60min_in_sample.jpeg "60min_in_sample")

```html
____________Performance____________
Start Date: 2021-01-01
End Date: 2022-01-01
Number of Trading Days: 365
Number of Instruments: 1
Number of Trades: 1
Total Return: 16.77%
Annualized Return: 16.77%
Sharpe Ratio: 1.30
Rolling Maximum Drawdown: -4.40%
```

### Out-of-sample

Below is the Backtest result from 2021-01-01 to 2022-01-01: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/60min_out_of_sample.jpeg "60min_out_of_sample")

```html
____________Performance____________
Start Date: 2022-01-01
End Date: 2022-08-01
Number of Trading Days: 212
Number of Instruments: 1
Number of Trades: 1
Total Return: 7.72%
Annualized Return: 13.28%
Sharpe Ratio: 1.15
Rolling Maximum Drawdown: -4.58%
```

## Summary & Future Work

As can be seen, both the 5-min and 60-min intervals deliver
positive returns in both in-sample and out-of-sample datasets.
However, as the interval increases, the trading opportunities
decrease. 

<table>
    <thead>
        <tr>
            <th rowspan=2>Interval</th>
            <th colspan=2>Annualized Return</th>
            <th colspan=2>Sharpe Ratio</th>
            <th colspan=2>Maximum Drawdown</th>
            <th colspan=2>Number of Trades</th>
        </tr>
        <tr>
            <th>In-sample</th>
            <th>Out-of-sample</th>
            <th>In-sample</th>
            <th>Out-of-sample</th>
            <th>In-sample</th>
            <th>Out-of-sample</th>
            <th>In-sample</th>
            <th>Out-of-sample</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td >5-min</td>
            <td>6.38%</td>
            <td>3.14%</td>
            <td>0.96</td>
            <td>0.82</td>
            <td>-5.39%</td>
            <td>-3.00%</td>
            <td>168</td>
            <td>89</td>
        </tr>
        <tr>
            <td >15-min</td>
            <td>8.14%</td>
            <td>-14.59%</td>
            <td>1.20</td>
            <td>-1.38</td>
            <td>-4.92%</td>
            <td>-11.29%</td>
            <td>33</td>
            <td>13</td>
        </tr>
        <tr>
            <td >60-min</td>
            <td>16.77%</td>
            <td>13.28%</td>
            <td>1.30</td>
            <td>1.15</td>
            <td>-4.4%</td>
            <td>-4.58%</td>
            <td>1</td>
            <td>1</td>
        </tr>
    </tbody>
</table>

There is a lot of work to be done to improve the strategy, which is 
included but not limited to:

- (1). In practice, the model should be trained
  in a dynamic rolling window, i.e., recalibrating
the parameters `entry_threshold` and `exit_threshold` regularly.
  The code for optimization is in `optimization_pair.py`.
  
- (2). Consider a vectorization (dataframe/numpy) implementation 
  of the backtest, to increase the optimization speed. It
  is relatively difficult to fully replicate the strategy in dataframe
  operations. An illustrative example is given in 
  `pandas_pairs.py`, which covers most of the features 
  in the model, and with much less execution time.
  
- (3). Add an absolute stop loss to each traded pair
  to mitigate drawdowns.

- (4). Consider the actual volume to have a better estimation of 
executed shares.
  
- (5). Consider using total least squares intead of OLS
to obtain the regression coefficients (hedge ratios).
  
- (6). Consider transaction costs in the simulation.

- (7). Consider different lookback window and trading window
for different time intervals.
  
