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
import pickle

import pandas as pd

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

with open("saved_results/opt_params_5min.pkl", "rb") as f:
    opt_params = pickle.load(f)
    # select securities to trade
    opt_params = {k: v for k, v in opt_params.items() if v["best_loss"] < 0}
    print(opt_params)
    number_instruments = len(opt_params)


result_df = pd.read_csv("saved_results/5min_in_sample/result_pairs.csv")

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
sr = sharp_ratio(perf_daily_df["portfolio_value"].pct_change(), 365)
roll_mdd = rolling_maximum_drawdown(perf_daily_df['portfolio_value'])
number_instruments = number_instruments
number_of_trading_days = (perf_daily_df.index[-1] - perf_daily_df.index[0]).days
num_trades = result_df.action.apply(lambda x: ast.literal_eval(x)[0].count('OPEN')).sum() // 2
tot_return = (perf_daily_df["portfolio_value"].iloc[-1]
         / perf_daily_df["portfolio_value"].iloc[0]) - 1
annualizd_return = tot_return * 365 / number_of_trading_days
print(
    "____________Performance____________\n"
    + "Start Date: {}\n".format(perf_daily_df.index[0].strftime("%Y-%m-%d"))
    + "End Date: {}\n".format(perf_daily_df.index[-1].strftime("%Y-%m-%d"))
    + "Number of Trading Days: {}\n".format(number_of_trading_days)
    + "Number of Instruments: {}\n".format(number_instruments)
    + "Number of Trades: {}\n".format(num_trades)
    + "Total Return: {:.2f}%\n".format(tot_return * 100)
    + "Annualized Return: {:.2f}%\n".format(annualizd_return * 100)
    + "Sharpe Ratio: {:.2f}\n".format(sr)
    + "Rolling Maximum Drawdown: {:.2f}%\n".format(roll_mdd.min() * 100)
)
print("Performance is done.")
