# -*- coding: utf-8 -*-
# @Time    : 15/12/2022 5:44 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: optimize_tsv.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
import os
import json
from datetime import datetime
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
from numpy_ext import rolling_apply
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK

from qtrader.core.security import Security, Currency, Futures, Stock
from qtrader.core.constants import Exchange
from qtrader.core.data import _get_data
from qtrader.core.utility import timeit
from qtalib.indicators import CYC, TSV
from qtrader_config import *

SEED = 2023

def load_data(
        security: Security,
        data_start: datetime,
        start: datetime,
        end: datetime,
        lookback_period: int = None
) -> pd.DataFrame:
    """Load OHLCV"""
    data = _get_data(
        security=security,
        start=data_start,
        end=end,
        dfield="kline",
        dtype=['time_key', 'open', 'high', 'low', 'close', 'volume']
    ).set_index("time_key")
    data = data.ffill().bfill()
    if data[data.index <= start].shape[0] < lookback_period:
        raise ValueError("There is not enough lookback data, change data_start")
    ret_data = pd.concat(
        [data[data.index <= start].iloc[-lookback_period:],
         data[data.index > start]]
    )
    return ret_data

security = Futures(
    code="HK.HHImain",
    lot_size=50,
    security_name="HK.HHImain",
    exchange=Exchange.HKFE,
    expiry_date="20221231"
)

# security = Stock(
#     code="US.SPY",
#     lot_size=1,
#     security_name="US.SPY",
#     exchange=Exchange.SMART
# )

data_start = datetime(2021, 12, 1, 0, 0, 0)
start = datetime(2022, 1, 1, 0, 0, 0)
end = datetime(2022, 12, 31, 23, 59, 59)
data_lookback_window = 300

# Load data
data = load_data(security, data_start, start, end, data_lookback_window)

def rolling_corr(args, **kwargs):
    """Rolling Pearson correlation"""
    case, tsv_length, tsv_ma_length = args
    data = kwargs.get("data")
    data_lookback_window = kwargs.get("data_lookback_window")

    TSV_lst = []
    for idx in range(data_lookback_window, data.shape[0]):
        data_lb = data.iloc[idx-data_lookback_window+1:idx+1]
        closes = data_lb["close"].to_numpy()
        volumes = data_lb["volume"].to_numpy()
        tsv = TSV(
            closes,
            volumes,
            tsv_length=tsv_length,
            tsv_ma_length=tsv_ma_length,
            tsv_lookback_length=44,
        )
        TSV_lst.append(tsv['tsv_ma'.encode('utf-8')])
    data_bt = data.iloc[data_lookback_window:].copy()
    data_bt["TSV"] = TSV_lst

    def pcy_turning_points(x):
        if x[0] > x[1] and x[1] < x[2]:  # and x[1] < 10:
            return 1
        elif x[0] < x[1] and x[1] > x[2]:  # and x[1] > 90:
            return -1
        return 0

    data_bt["TSV_interval"] = rolling_apply(
        pcy_turning_points,
        3,
        data_bt.TSV.values
    )
    data_bt = data_bt[
        (data_bt.TSV_interval == 1) | (data_bt.TSV_interval == -1)
    ]

    if len(data_bt) < 50:
        return {
            'loss': np.inf,
            'status': STATUS_OK,
            'rolling_corr': np.nan
        }
    def corr(x, y):
        return np.corrcoef(x, y)[0][1]

    n = len(data_bt) // 2
    rolling_corr = rolling_apply(
        corr,
        n,
        data_bt.TSV.diff().apply(lambda x: np.sign(x)).dropna().values,
        data_bt.close.diff().apply(lambda x: np.sign(x)).dropna().values
    )
    rolling_corr = rolling_corr[~np.isnan(rolling_corr)]
    rolling_corr_mean = rolling_corr.mean()
    loss = -rolling_corr_mean
    return {
        'loss': loss,
        'status': STATUS_OK,
        'rolling_corr': rolling_corr_mean,
    }

def worker(
        data, data_lookback_window, space
) -> Dict[str, float]:
    """Process that run the optimization"""
    trials = Trials()
    best = timeit(fmin)(
        partial(rolling_corr,
                data=data,
                data_lookback_window=data_lookback_window
                ),
        space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials,
        rstate=np.random.default_rng(SEED)
    )
    tsv_length = tsv_length_choice[best['tsv_length']]
    tsv_ma_length = tsv_ma_length_choice[best['tsv_ma_length']]
    opt_params = {
        'tsv_length': tsv_length,
        'tsv_ma_length': tsv_ma_length,
        'rolling_corr': trials.best_trial['result']['rolling_corr']
    }
    print(opt_params)
    target = os.path.basename(__file__).replace(".py", "").split("_")[-1]
    with open(f'optimized_{target}_params.json', 'w') as f:
        json.dump(opt_params, f)
    return opt_params

# define a search space
tsv_length_choice = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
tsv_ma_length_choice = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
space = hp.choice('a', [
    ('case 1',
     hp.choice('tsv_length', tsv_length_choice),
     hp.choice('tsv_ma_length', tsv_ma_length_choice),
     )]
)

worker(data, data_lookback_window, space)
