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

result_df = pd.read_csv("results/2022-09-09 12-47-25.908496/result_pairs.csv")

closes_df = pd.DataFrame(
    index=result_df.datetime.apply(
        lambda x: datetime.strptime(
            ast.literal_eval(x)[0], "%Y-%m-%d %H:%M:%S")),
    columns=instruments["security"]["Backtest"]
)
for i, sec_code in enumerate(closes_df.columns):
    closes_df[sec_code] = result_df.close.apply(
        lambda x: ast.literal_eval(x)[0][i]
    ).to_list()

range_start = datetime(2021, 9, 11)
range_end = datetime(2021, 9, 15)
closes_df = closes_df[
    (closes_df.index >= range_start)
    & (closes_df.index <= range_end)
]
norm_closes_df = closes_df/closes_df.iloc[0]
norm_closes_df.plot()
plt.show()
print()
