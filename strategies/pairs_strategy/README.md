# Pairs Strategy

---
author: Joseph Chen\
date: Oct 13rd, 2022
---

This demo will show how to build a practical strateqy 
(pairs trading) from scratch in the framework of 
[`QTrader`](https://github.com/josephchenhk/qtrader).

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

$$\log(S_1) = \gamma\cdot\log(S_2) + \mu + \epsilon$$

where the slope $\gamma$ represents the hedge ratio.
We can define the residual term $\epsilon$ as the spread 
$s = \log(S_1) - \gamma\cdot\log(S_2) - \mu$, which is expected 
to exhibit mean-reverting properties. 

Although this spread is mean-reverting in principle, 
there is too much noise that could be harmful to
the profitability and stability of the strategy. Here
is an example of this calculated spread:

![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/mean_reversion.jpeg "mean_reversion")

The spread could be smoothed with the aid of 
moving average. We can define the smoothed spread
as:

$$
\tilde{s} = (s_{SMA} - \text{avg}(s_{LMA})) / \text{std}(s_{LMA}) 
$$

where $s_{SMA}$ is the short-term moving average of $s$,
and $s_{LMS}$ is the long-term moving average of $s$.
The moving average windows are defined in the parameters:
`ma_short_length` and `ma_long_length`. Below is 
the smoothed spread from previous data:

![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/mean_reversion_ma.jpeg "mean_reversion_ma")

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
   and the hedge ratio will also remain constant.
   
3. The distribution of the spread follows a normal
distribution, and the distances can be measured by
   standard deviations.
   
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
parameter $\delta$ here is a threshold number to open trades.

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
a number to be used for closing trades.

Besides the z-score condition, there are
also other entry conditions: when a pair is on, 
the hedge ratio $\gamma$ should be positive. This
is to ensure that we always have a market-neutral
position, i.e., long position in one security
and short position in another. 
~~And we also close existing positions and avoid
opening new position at the end of the testing
period.~~
When the testing period finishes, a recalibration
will be carried out. The cointegration properties
are re-evaluated by measuring the 
correlation (`corr_maintain_threshold`) and 
cointegration p-value (`coint_pvalue_maintain_threshold`).
If this property is violated, existing positions
will be closed immediately; otherwise, the hedge
ratios will be updated, and existing positions
will be adjusted accordingly.

# Simulation Results

As discussed in EDA，the trading universe is six cryptocurrency 
pairs:`BTC.USD`, `EOS.USD`, `ETH.USD`, `LTC.USD`, `TRX.USD`, 
and`XRP.USD`. Hence there are 15 ( $C^2_6=15$ ) candidate pairs:

```html
('BTC.USD', 'EOS.USD'),
('BTC.USD', 'ETH.USD'),
('BTC.USD', 'LTC.USD'),
('BTC.USD', 'TRX.USD'),
('BTC.USD', 'XRP.USD'),
('EOS.USD', 'ETH.USD'),
('EOS.USD', 'LTC.USD'),
('EOS.USD', 'TRX.USD'),
('EOS.USD', 'XRP.USD'),
('ETH.USD', 'LTC.USD'),
('ETH.USD', 'TRX.USD'),
('ETH.USD', 'XRP.USD'),
('LTC.USD', 'TRX.USD'),
('LTC.USD', 'XRP.USD'),
('TRX.USD', 'XRP.USD')
```

We use the training data (from 2021-01-01 to 2021-12-31)
to calculate the correlation of different pairs in sample,
and get seven pairs that have a correlation over 0.7:

```html
corr("BTC.USD", "LTC.USD") = 0.7410557870708834
corr("EOS.USD", "LTC.USD") = 0.7968610662268301
corr("EOS.USD", "TRX.USD") = 0.7397461563710355
corr("EOS.USD", "XRP.USD") = 0.7456244422109919
corr("ETH.USD", "TRX.USD") = 0.819769004085257
corr("ETH.USD", "XRP.USD") = 0.8416744722298698
corr("TRX.USD", "XRP.USD") = 0.9535877879201461
```

The OHLCV data of interval 15-min/30-min/60-min
are used for simulations. The look-back 
window is fixed to be 2000/1000/750 bars (`lookback_period`). 
In the training period, we apply a two-step 
statistical method to the data in lookback window to 
determine the candidate pairs. Only those pairs with 
a correlation higher than the threshold 
(`corr_init_threshold=0.85`) and an 
ADF p-value less than the threshold 
(`coint_pvalue_init_threshold=0.01`) will be shortlisted. 
The trading
window is next few bars (use `recalibration_lookback_ratio`
to calculate it as a portion of the lookback period) immediately 
following the previous training period. When the trading period 
completes, the dynamic rolling window will be automatically 
shifted ahead for the next training and trading periods.
In the trading period, the spread 
$s = \log(S_1) - \gamma\cdot\log(S_2) - \mu$ 
(and thus $\tilde{s}$) is updated
by feeding the new price $S_1$ and $S_2$, and entry and
exit are determined by z-score of the calculated spread.

Assuming the spread follows normal distribution, 
the entry threshold is defined as percentile
level of 75% (`entry_threshold_pct = 0.75`); and the 
exit threshold is defined
as percentile level of 99%
(`exit_threshold_pct = 0.99`). To avoid overfit, 
these parameters will stay invariant throughout 
the training and testing periods. We also assume for 
each trading opportunity, 
the maximum capital allocated to individual security 
is USD 1 million (`capital_per_entry=1000000`). And we 
only enter the trade once for repeating signals 
(`max_number_of_entry=1`). 

A summary of the strategy parameters is shown below:

```json
"lookback_period": 2000/1000/750,
"recalibration_lookback_ratio": (to be determined),
"corr_init_threshold": 0.85,
"corr_maintain_threshold": 0.65,
"coint_pvalue_init_threshold": 0.01,
"coint_pvalue_maintain_threshold": 0.10,
"entry_threshold_pct": 0.75,
"exit_threshold_pct": 0.99,
"max_number_of_entry": 1,
"capital_per_entry": 1000000,
"ma_short_length": (to be determined),
"ma_long_length": 100/50/30
```

## Optimization Objective Function

The objective of the strategy is to maximize the 
the total return and minimize the drawdown. 
Therefore, the objective
function is defined as minimizing $f$:

$$
f(entry\textunderscore threshold, exit\textunderscore threshold) 
= -\min(\max(\text{SR}, 0), 1.0) * \text{TOTR}
$$

where $\text{SR}$ is the Sharpe ratio, and $\text{TOTR}$ is the total
return. 

## Stocks

The strategy is tested on both 
in-sample (from 2021-03-01 to 2022-03-01) 
and out-of-sample (from 2022-03-01 to 2022-10-01) 
datasets.

Below is the Backtest results: 
![alt text](https://github.com/josephchenhk/demo_strategy/blob/main/contents/v1_5min_in_sample.jpeg "5min_in_sample")

<table>
    <thead>
        <tr>
            <th rowspan=2>Interval</th>
            <th colspan=2>Annualized Return</th>
            <th colspan=2>Sharpe Ratio</th>
            <th colspan=2>Maximum Drawdown</th>
            <th colspan=2>Number of Days/Number of Trades</th>
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
            <td style="color:blue;"><b>15-min</b></td>
            <td style="color:blue;"><b>15.50%</b></td>
            <td style="color:blue;"><b>19.14%</b></td>
            <td style="color:blue;"><b>5.28</b></td>
            <td style="color:blue;"><b>8.45</b></td>
            <td style="color:blue;"><b>-2.17%</b></td>
            <td style="color:blue;"><b>-1.13%</b></td>
            <td style="color:blue;"><b>364/584</b></td>
            <td style="color:blue;"><b>213/235</b></td>
        </tr>
        <tr>
            <td style="color:blue;"><b>30-min</b></td>
            <td style="color:blue;"><b>11.94%</b></td>
            <td style="color:blue;"><b>12.55%</b></td>
            <td style="color:blue;"><b>4.96</b></td>
            <td style="color:blue;"><b>7.19</b></td>
            <td style="color:blue;"><b>-1.79%</b></td>
            <td style="color:blue;"><b>-0.93%</b></td>
            <td style="color:blue;"><b>364/449</b></td>
            <td style="color:blue;"><b>213/176</b></td>
        </tr>
        <tr>
            <td style="color:blue;"><b>60-min</b></td>
            <td style="color:blue;"><b>5.53%</b></td>
            <td style="color:blue;"><b>9.75%</b></td>
            <td style="color:blue;"><b>2.13</b></td>
            <td style="color:blue;"><b>4.16</b></td>
            <td style="color:blue;"><b>-1.74%</b></td>
            <td style="color:blue;"><b>-0.96%</b></td>
            <td style="color:blue;"><b>364/291</b></td>
            <td style="color:blue;"><b>213/106</b></td>
        </tr>
    </tbody>
</table>

## Summary & Future Work

A comparison with the previous version:

There is a lot of work to be done to improve the strategy, which is 
included but not limited to:

~~- (1). In practice, the model should be trained 
in a dynamic rolling window, i.e., recalibrating
the parameters `entry_threshold` and `exit_threshold` regularly. 
The code for optimization is in `optimization_pair.py`.~~
  
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

~~- (7). Consider different lookback window and 
trading window for different time intervals.~~
  
- (8). Utilize a one-period execution lag for all trade
orders to approximate the bid-ask spread since 
  contrarian trading strategies might be unknowingly
  buying for bid prices and vice versa.
  
## References

[1]. [High Frequency and Dynamic Pairs Trading Based on 
Statistical Arbitrage Using a Two-Stage Correlation and
Cointegration Approach](https://www.ccsenet.org/journal/index.php/ijef/article/view/33007)

[2]. [Pairs Trading: Optimal Threshold Strategies](https://www.uv.es/bfc/TFM%202018/18.%20Alejandro%20Alvarez.pdf)

[3]. [Enhancing a Pairs Trading strategy with the
application of Machine Learning](https://www.sciencedirect.com/science/article/abs/pii/S0957417420303146)

[4]. [Pairs Trading in Cryptocurrency Markets](https://www.researchgate.net/publication/346845365_Pairs_Trading_in_Cryptocurrency_Markets)
