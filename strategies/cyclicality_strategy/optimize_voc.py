# -*- coding: utf-8 -*-
# @Time    : 15/12/2022 5:44 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: optimize_cyclicality.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
import pickle
from datetime import datetime
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy_ext import rolling_apply
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK

from qtrader.core.security import Security, Currency, Futures
from qtrader.core.constants import Exchange
from qtrader.core.data import _get_data
from qtrader.core.utility import timeit
from qtalib.indicators import CYC
from qtrader_config import *

SEED = 2022

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

# security = Currency(
#     code="BTC.USD",
#     lot_size=1,
#     security_name="BTC.USD",
#     exchange=Exchange.SMART
# )
security = Futures(
    code="HK.HSImain",
    lot_size=50,
    security_name="HK.HSImain",
    exchange=Exchange.HKFE,
    expiry_date="20221231"
)

data_start = datetime(2020, 11, 15, 0, 0, 0)
start = datetime(2021, 1, 1, 0, 0, 0)
end = datetime(2021, 12, 31, 23, 59, 59)
lookback_window = 150

# Load data
data = load_data(security, data_start, start, end, lookback_window)

def rolling_corr(args, **kwargs):
    """Rolling Pearson correlation"""
    case, short_ma_length, long_ma_length = args
    data = kwargs.get("data")
    lookback_window = kwargs.get("lookback_window")

    if short_ma_length >= long_ma_length:
        return {
            'loss': float("inf"),
            'status': STATUS_OK,
            'rolling_corr': np.nan
        }

    PCY = []
    pcy = 0
    for idx in range(lookback_window, data.shape[0]):
        data_lb = data.iloc[idx-lookback_window+1:idx+1]
        closes = data_lb["close"].to_numpy()
        volumes = data_lb["volume"].to_numpy()
        pcy = CYC(
            data=closes,
            cyc=pcy,
            short_ma_length=short_ma_length,
            long_ma_length=long_ma_length,
            alpha=0.33,
            lookback_window=10,
        )
        PCY.append(pcy)
    data_bt = data.iloc[lookback_window:].copy()
    data_bt["PCY"] = PCY

    def turning_points(x):
        if x[0] > x[1] and x[2] > x[1]:
            return 1
        elif x[0] < x[1] and x[2] < x[1]:
            return -1
        return 0

    data_bt["PCY_interval"] = rolling_apply(
        turning_points,
        3,
        data_bt.PCY.values
    )
    data_bt = data_bt[
        (data_bt.PCY_interval == 1) | (data_bt.PCY_interval == -1)
    ]

    s1 = data_bt.close.diff().apply(lambda x: int(x > 0))
    s2 = data_bt.PCY.diff().apply(lambda x: int(x > 0))
    rolling_corr = s1.rolling(200).corr(s2).dropna()
    return {
        'loss': -rolling_corr.mean(),
        'status': STATUS_OK,
        'rolling_corr': rolling_corr.mean()
    }

def worker(
        data, lookback_window, space
) -> Dict[str, float]:
    """Process that run the optimization"""
    trials = Trials()
    best = timeit(fmin)(
        partial(rolling_corr,
                data=data,
                lookback_window=lookback_window),
        space,
        algo=tpe.suggest,
        max_evals=60,
        trials=trials,
        rstate=np.random.default_rng(SEED)
    )
    opt_params = {
        'short_ma_length': short_ma_length_choice[best['short_ma_length']],
        'long_ma_length': long_ma_length_choice[best['long_ma_length']],
        'rolling_corr': trials.best_trial['result']['rolling_corr']
    }
    print(opt_params)
    return opt_params

# define a search space
short_ma_length_choice = [10, 15, 20, 25, 30, 35, 40, 45, 50]
long_ma_length_choice = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
space = hp.choice('a', [
    ('case 1',
     hp.choice('short_ma_length', short_ma_length_choice),
     hp.choice('long_ma_length', long_ma_length_choice),
     )]
)

worker(data, lookback_window, space)
