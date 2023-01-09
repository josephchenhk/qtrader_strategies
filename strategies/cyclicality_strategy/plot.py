# -*- coding: utf-8 -*-
# @Time    : 8/1/2023 9:06 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: plot.py

"""
Copyright (C) 2022 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the terms of the JXW license, 
which unfortunately won't be written for another century.

You should have received a copy of the JXW license with this file. If not, 
please write to: josephchenhk@gmail.com
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy_ext import rolling_apply

from qtrader.core.security import Security, Futures
from qtrader.core.constants import Exchange
from qtrader.core.data import _get_data
from qtrader.plugins.analysis.metrics import sharpe_ratio
from qtalib.indicators import CYC, SAR
from qtrader_config import *


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

if __name__ == "__main__":
    security = Futures(
        code="HK.HSImain",
        lot_size=50,
        security_name="HK.HSImain",
        exchange=Exchange.HKFE,
        expiry_date="20221231"
    )

    data_start = datetime(2016, 1, 1, 0, 0, 0)
    start = datetime(2022, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 1, 23, 59, 59)
    data_lookback_window = 110

    # Load data
    data = load_data(security, data_start, start, end, data_lookback_window)

    cyc = {
        "close": 50,
        "volume": 50
    }
    PCY = []
    VOC = []
    for idx in range(data_lookback_window, data.shape[0]):
        data_lb = data.iloc[idx - data_lookback_window + 1:idx + 1]
        highs = data_lb["high"].to_numpy(dtype=float)
        lows = data_lb["low"].to_numpy(dtype=float)
        closes = data_lb["close"].to_numpy(dtype=float)
        volumes = data_lb["volume"].to_numpy(dtype=float)
        with open("optimized_pcy_params.json", "r") as f:
            pcy_params = json.load(f)
        cyc["close"] = CYC(
            data=closes,
            cyc=cyc["close"],
            short_ma_length=pcy_params['short_ma_length'],
            long_ma_length=pcy_params['long_ma_length'],
            alpha=pcy_params['alpha'],
            lookback_window=pcy_params['lookback_window'],
        )
        PCY.append(cyc["close"])

        with open("optimized_voc_params.json", "r") as f:
            voc_params = json.load(f)
        cyc["volume"] = CYC(
            data=volumes,
            cyc=cyc["volume"],
            short_ma_length=voc_params['short_ma_length'],
            long_ma_length=voc_params['long_ma_length'],
            alpha=voc_params['alpha'],
            lookback_window=voc_params['lookback_window'],
        )
        VOC.append(cyc["volume"])
    data_bt = data.iloc[data_lookback_window:].copy()
    data_bt["VOC"] = VOC
    data_bt["PCY"] = PCY

    data_bt_plot = data_bt[(data_bt.index >= start) & (data_bt.index <= end)]
    fig, (ax1, ax3) = plt.subplots(
        2, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})
    ax2 = ax1.twinx()
    ax1.plot(data_bt_plot.index, data_bt_plot.close, 'g-', label='price')
    ax2.plot(data_bt_plot.index, data_bt_plot.VOC, 'c-', label='VOC')
    ax2.plot(data_bt_plot.index, data_bt_plot.PCY, 'y-', label='PCY')
    ax1.legend(loc=0)
    ax2.legend(loc=1)
    ax1.set_xlabel('datetime')
    ax2.set_ylabel('Price', color='b')
    ax2.set_ylabel('VOC/PCY', color='g')
    ax3.bar(data_bt_plot.index, data_bt_plot.volume, label='volume')

    plt.show()
