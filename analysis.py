# -*- coding: utf-8 -*-
# @Time    : 9/9/2022 2:38 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: analysis.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
from datetime import datetime
import ast

import pandas as pd
import matplotlib.pyplot as plt

from qtrader.plugins.analysis.metrics import sharp_ratio
from qtrader.plugins.analysis.metrics import rolling_maximum_drawdown

instruments = {
    "security": {
        "Backtest": ["BTC.USD", "EOS.USD", "ETH.USD", "LTC.USD", "TRX.USD", "XRP.USD"],
    },
    "lot": {
        "Backtest": [1, 1, 1, 1, 1, 1],
    },
    "commission": {
        "Backtest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "slippage": {
        "Backtest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

result_df = pd.read_csv("results/2022-09-13 16-30-58.390740/result_pairs.csv")

# # Plot normalized prices
# closes_df = pd.DataFrame(
#     index=result_df.datetime.apply(
#         lambda x: datetime.strptime(
#             ast.literal_eval(x)[0], "%Y-%m-%d %H:%M:%S")),
#     columns=instruments["security"]["Backtest"]
# )
# for i, sec_code in enumerate(closes_df.columns):
#     closes_df[sec_code] = result_df.close.apply(
#         lambda x: ast.literal_eval(x)[0][i]
#     ).to_list()
#
# range_start = datetime(2021, 7, 19)
# range_end = datetime(2021, 8, 16)
# closes_df = closes_df[
#     (closes_df.index >= range_start)
#     & (closes_df.index <= range_end)
# ]
# norm_closes_df = closes_df/closes_df.iloc[0]
# norm_closes_df.plot()
# plt.show()
# print("-"*40)

# Calculate performance
perf_df = pd.DataFrame(
    index=result_df.datetime.apply(
        lambda x: datetime.strptime(
            ast.literal_eval(x)[0], "%Y-%m-%d %H:%M:%S")),
    columns=["portfolio_value"]
)
perf_df["portfolio_value"] = result_df.strategy_portfolio_value.apply(
        lambda x: sum(ast.literal_eval(x))).to_list()
perf_daily_df = perf_df.resample('D').agg({"portfolio_value": "last"})
sr = sharp_ratio(perf_daily_df["portfolio_value"].pct_change(), 252)
roll_mdd = rolling_maximum_drawdown(perf_daily_df['portfolio_value'])
number_instruments = 6
num_trades = result_df.action.apply(lambda x: ast.literal_eval(x)[0].count('OPEN')).sum()
print(
    "____________Performance____________\n"
    + "Start Date: {}\n".format(perf_daily_df.index[0].strftime("%Y-%m-%d"))
    + "End Date: {}\n".format(perf_daily_df.index[-1].strftime("%Y-%m-%d"))
    + "Number of Instruments: {}\n".format(number_instruments)
    + "Number of Trades: {}\n".format(num_trades)
    + "Total Return: {:.2f}%\n".format(
        (perf_daily_df["portfolio_value"].iloc[-1]
         / perf_daily_df["portfolio_value"].iloc[0] - 1) * 100)
    + "Sharpe Ratio: {:.2f}\n".format(sr)
    + "Rolling Maximum Drawdown: {:.2f}%\n".format(roll_mdd.min() * 100)
)
