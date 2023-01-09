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
import os
import json
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

from qtrader.core.security import Security, Currency, Futures, Stock
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

# data_start = datetime(2014, 1, 1, 0, 0, 0)
# start = datetime(2015, 1, 1, 0, 0, 0)
# end = datetime(2021, 12, 31, 23, 59, 59)
# lookback_window = 150

security = Futures(
    code="HK.HSImain",
    lot_size=50,
    security_name="HK.HSImain",
    exchange=Exchange.HKFE,
    expiry_date="20221231"
)

# security = Stock(
#     code="US.SPY",
#     lot_size=1,
#     security_name="US.SPY",
#     exchange=Exchange.SMART
# )

data_start = datetime(2016, 1, 1, 0, 0, 0)
start = datetime(2018, 1, 1, 0, 0, 0)
end = datetime(2023, 1, 1, 23, 59, 59)
data_lookback_window = 110

# Load data
data = load_data(security, data_start, start, end, data_lookback_window)

def rolling_corr(args, **kwargs):
    """Rolling Pearson correlation"""
    case, alpha, short_ma_length, long_ma_length, lookback_window = args
    data = kwargs.get("data")
    data_lookback_window = kwargs.get("data_lookback_window")

    if short_ma_length >= long_ma_length:
        return {
            'loss': np.inf,
            'status': STATUS_OK,
            'rolling_corr': np.nan
        }

    VOC = []
    voc = 0
    for idx in range(data_lookback_window, data.shape[0]):
        data_lb = data.iloc[idx-data_lookback_window+1:idx+1]
        volumes = data_lb["volume"].to_numpy(dtype=float)
        voc = CYC(
            data=volumes,
            cyc=voc,
            short_ma_length=short_ma_length,
            long_ma_length=long_ma_length,
            alpha=alpha,
            lookback_window=lookback_window,
        )
        VOC.append(voc)
    data_bt = data.iloc[data_lookback_window:].copy()
    data_bt["VOC"] = VOC

    def voc_turning_points(x):
        # VOC trough represents a strong trend, return 1
        if x[0] > x[1] and x[1] < x[2]:  # and x[1] < 10:
            return 1
        # VOC peak represents trend doesn't continue, return -1
        elif x[0] < x[1] and x[1] > x[2]:  # and x[1] > 90:
            return -1
        return 0

    data_bt["VOC_interval"] = rolling_apply(
        voc_turning_points,
        3,
        data_bt.VOC.values
    )
    data_bt = data_bt[
        (data_bt.VOC_interval == 1) | (data_bt.VOC_interval == -1)
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
        data_bt.VOC.diff().apply(lambda x: np.sign(x)).dropna().values,
        data_bt.volume.diff().apply(lambda x: np.sign(x)).dropna().values
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
        max_evals=300,
        trials=trials,
        rstate=np.random.default_rng(SEED)
    )
    short_ma_length = short_ma_length_choice[best['short_ma_length']]
    long_ma_length = long_ma_length_choice[best['long_ma_length']]
    lookback_window = lookback_window_choice[best['lookback_window']]
    opt_params = {
        'alpha': best['alpha'],
        'short_ma_length': short_ma_length,
        'long_ma_length': long_ma_length,
        'lookback_window': lookback_window,
        'rolling_corr': trials.best_trial['result']['rolling_corr']
    }
    print(opt_params)
    target = os.path.basename(__file__).replace(".py", "").split("_")[-1]
    with open(f'optimized_{target}_params.json', 'w') as f:
        json.dump(opt_params, f)
    return opt_params

# define a search space
short_ma_length_choice = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
long_ma_length_choice = [20, 25, 30, 35, 40, 45, 50]
lookback_window_choice = [10, 15, 20, 25, 30, 35, 40, 45, 50]
space = hp.choice('a', [
    ('case 1',
     hp.uniform('alpha', 0.1, 0.33),
     hp.choice('short_ma_length', short_ma_length_choice),
     hp.choice('long_ma_length', long_ma_length_choice),
     hp.choice('lookback_window', lookback_window_choice),
     )]
)

worker(data, data_lookback_window, space)
