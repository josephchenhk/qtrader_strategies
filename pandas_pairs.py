# -*- coding: utf-8 -*-
# @Time    : 13/9/2022 4:29 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: pandas_pairs.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
from typing import List, Dict
from datetime import datetime
import itertools
import json

import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller

from qtrader.core.security import Security, Currency
from qtrader.core.constants import Exchange
from qtrader.core.data import _get_data


def load_data(
        stock_list: List[Security],
        data_start: datetime,
        start: datetime,
        end: datetime,
        lookback_period: int = None
) -> pd.DataFrame:
    """Load close prices"""
    all_data = pd.DataFrame(columns=[s.code for s in stock_list])
    for security in stock_list:
        data = _get_data(
            security=security,
            start=data_start,
            end=end,
            dfield="kline",
            dtype=['time_key', 'open', 'high', 'low', 'close', 'volume'])
        if all_data.empty:
            all_data = data[["time_key", "close"]].set_index("time_key")
            all_data.columns = [security.code]
        else:
            new_data = data[["time_key", "close"]].set_index("time_key")
            new_data.columns = [security.code]
            all_data = all_data.join(new_data, how="outer")
    all_data = all_data.ffill().bfill()
    if all_data[all_data.index <= start].shape[0] < lookback_period:
        raise ValueError("There is not enough lookback data, change data_start")
    ret_data = pd.concat(
        [all_data[all_data.index <= start].iloc[-lookback_period:],
         all_data[all_data.index > start]]
    )
    return ret_data

def run_pairs(
        data: pd.DataFrame,
        asset1: str,
        asset2: str,
        lookback_period: int = 1440,
        correlation_threshold: float = -1.1,
        recalibration_interval: int = 384,
        cointegration_pvalue_entry_threshold: float = 0.05,
        entry_threshold: float = 1.5,
        exit_threshold: float = 2.5,
        max_number_of_entry: int = 1,
        capital_per_entry: float = 1000000.0
):
    """run backtest for a pair"""
    # (asset1, asset2) is the pair to test strategy
    signals = pd.DataFrame()
    signals['asset1'] = data[asset1]
    signals['asset2'] = data[asset2]

    # correlation
    signals['corr'] = np.log(signals['asset1']).rolling(
        lookback_period).corr(np.log(signals['asset2']))
    recalibration_date = [0] * signals.shape[0]
    recalibration_corr = [np.nan] * signals.shape[0]
    for i in range(lookback_period-1, signals.shape[0], recalibration_interval):
        recalibration_date[i] = 1
        recalibration_corr[i] = signals['corr'].iloc[i]
    signals['recalibration_date'] = recalibration_date
    signals['recalibration_corr'] = recalibration_corr
    signals['recalibration_corr'] = signals['recalibration_corr'].ffill()

    # cointegration
    model = RollingOLS(
        endog=np.log(signals['asset1']),
        exog=np.log(signals['asset2']),
        window=lookback_period,
    )
    model_fit = model.fit()
    gamma = [np.nan] * signals.shape[0]
    gamma_lst = model_fit.params.to_numpy().reshape(-1).tolist()
    for i in range(lookback_period-1, signals.shape[0], recalibration_interval):
        gamma[i] = gamma_lst[i]
    signals['gamma'] = gamma
    signals['gamma_ffill'] = signals['gamma'].ffill()
    signals['gamma_bfill'] = signals['gamma'].bfill()
    signals['spread'] = np.log(signals['asset1']) - signals["gamma_ffill"] * np.log(signals['asset2'])
    signals['spread_bfill'] = np.log(signals['asset1']) - signals["gamma_bfill"] * np.log(signals['asset2'])

    # This code will crash for unknown reasons
    # signals['adf_pvalue'] = signals['spread'].rolling(lookback_period).apply(
    #     lambda x: adfuller(x, autolag="AIC")[1])

    adf_pvalue = [np.nan] * signals.shape[0]
    for i in range(lookback_period-1, signals.shape[0], recalibration_interval):
        p = adfuller(
            signals['spread_bfill'].iloc[i+1-lookback_period:i+1], autolag="AIC")[1]
        if isinstance(p, float):
            adf_pvalue[i] = p
    signals['adf_pvalue'] = adf_pvalue
    signals['adf_pvalue'] = signals['adf_pvalue'].ffill()

    # calculate z-score
    signals['spread_zscore'] = (
        (signals["spread"] - signals["spread"].rolling(lookback_period).mean())
        / signals["spread"].rolling(lookback_period).std()
    )

    # create entry signal - short if z-score is greater than upper limit else long
    signals['entry_signals'] = 0
    signals['entry_signals'] = np.select(
        [(entry_threshold < signals['spread_zscore'])
         & (signals['spread_zscore'] < (entry_threshold + exit_threshold)/2)
         & (signals['recalibration_date'] == 0)
         & (signals['gamma_ffill'] > 0.1)
         & (signals['adf_pvalue'] < cointegration_pvalue_entry_threshold)
         & (signals['recalibration_corr'] > correlation_threshold),
         (-(entry_threshold + exit_threshold)/2 < signals['spread_zscore'])
         & (signals['spread_zscore'] < -entry_threshold)
         & (signals['recalibration_date'] == 0)
         & (signals['gamma_ffill'] > 0.1)
         & (signals['adf_pvalue'] < cointegration_pvalue_entry_threshold)
         & (signals['recalibration_corr'] > correlation_threshold)],
        [-1, 1],
        default=0)

    # Create exit signal
    signals['exit_long1_short2_signals'] = 0
    signals['exit_long1_short2_signals'] = np.where(
         (signals['recalibration_date'] == 1)
         | (signals['spread_zscore'] >= 0)
         | (signals['spread_zscore'] < -exit_threshold),
         1, 0)

    signals['exit_short1_long2_signals'] = 0
    signals['exit_short1_long2_signals'] = np.where(
         (signals['recalibration_date'] == 1)
         | (signals['spread_zscore'] <= 0)
         | (signals['spread_zscore'] > exit_threshold),
         1, 0)

    # shares to buy for each position
    signals['qty1'] = np.select(
        [signals['gamma_ffill'] <= 1, signals['gamma_ffill'] > 1],
        [capital_per_entry // signals['asset1'],
         capital_per_entry / signals['gamma_ffill'] // signals['asset1']],
        default=0)
    signals['qty2'] = np.select(
        [signals['gamma_ffill'] <= 1, signals['gamma_ffill'] > 1],
        [capital_per_entry * signals['gamma_ffill'] // signals['asset2'],
         capital_per_entry // signals['asset2']],
        default=0)

    # calculate position and pnl
    position = [0] * signals.shape[0]
    qty1 = [0] * signals.shape[0]
    qty2 = [0] * signals.shape[0]
    for i, (timestamp, row) in enumerate(signals.iterrows()):
        if i < lookback_period-1:
            continue

        entry_signals = row['entry_signals']
        exit_long1_short2_signals = row['exit_long1_short2_signals']
        exit_short1_long2_signals = row['exit_short1_long2_signals']

        if entry_signals == 1 and exit_long1_short2_signals:
            raise ValueError("entry and exit long1|short2 signals coexist!")
        if entry_signals == -1 and exit_short1_long2_signals:
            raise ValueError("entry and exit short1|long2 signals coexist!")

        if i == lookback_period-1:
            prev_position = 0
            prev_qty1 = 0
            prev_qty2 = 0
        else:
            prev_position = position[i-1]
            prev_qty1 = qty1[i - 1]
            prev_qty2 = qty2[i - 1]

        if prev_position == 0 and entry_signals:
            position[i] = entry_signals
            qty1[i] = row['qty1']
            qty2[i] = row['qty2']
        elif prev_position == 1 and exit_long1_short2_signals:
            position[i] = 0
            qty1[i] = 0
            qty2[i] = 0
        elif prev_position == -1 and exit_short1_long2_signals:
            position[i] = 0
            qty1[i] = 0
            qty2[i] = 0
        else:
            position[i] = prev_position
            qty1[i] = prev_qty1
            qty2[i] = prev_qty2

    signals['position'] = position
    signals['qty1'] = qty1
    signals['qty2'] = qty2

    pnl = [0] * signals.shape[0]
    for i, (timestamp, row) in enumerate(signals.iterrows()):
        if i < lookback_period - 1:
            continue
        qty1 = row['qty1']
        qty2 = row['qty2']
        price1 = row['asset1']
        price2 = row['asset2']
        prev_price1 = signals.iloc[i - 1]['asset1']
        prev_price2 = signals.iloc[i - 1]['asset2']

        if position[i-1] == position[i] and position[i] != 0:
            pnl[i] = pnl[i-1] + (
                price1 * qty1 - price2 * qty2
                - prev_price1 * qty1 + prev_price2 * qty2
            )*position[i]
        elif position[i-1] != 0 and position[i] == 0:
            pnl[i] = pnl[i-1] + (
                price1 * qty1 - price2 * qty2
                - prev_price1 * qty1 + prev_price2 * qty2
            )*position[i-1]
        else:
            pnl[i] = pnl[i-1]
    signals['pnl'] = pnl
    signals['pnl'] += capital_per_entry
    return signals['pnl']


def run_strategy(**kwargs):
    """Run strategy for portfolio"""
    # Load default parameters
    with open("strategies/pairs_strategy/params.json", "r") as f:
        params = json.load(f)
    # Override parameters
    if kwargs.get("override_indicator_cfg"):
        for k, v in kwargs["override_indicator_cfg"]["params"].items():
            params[k] = v

    # Instruments
    stock_list = [
        Currency(
            code="BTC.USD",
            lot_size=1,
            security_name="BTC.USD",
            exchange=Exchange.SMART),
        Currency(
            code="EOS.USD",
            lot_size=1,
            security_name="EOS.USD",
            exchange=Exchange.SMART),
        Currency(
            code="ETH.USD",
            lot_size=1,
            security_name="ETH.USD",
            exchange=Exchange.SMART),
        Currency(
            code="LTC.USD",
            lot_size=1,
            security_name="LTC.USD",
            exchange=Exchange.SMART),
        Currency(
            code="TRX.USD",
            lot_size=1,
            security_name="TRX.USD",
            exchange=Exchange.SMART),
        Currency(
            code="XRP.USD",
            lot_size=1,
            security_name="XRP.USD",
            exchange=Exchange.SMART),
    ]
    data_start = datetime(2020, 11, 15, 0, 0, 0)
    start = datetime(2021, 1, 1, 0, 0, 0)
    end = datetime(2021, 12, 31, 23, 59, 59)

    # Load data
    data = load_data(stock_list, data_start, start, end, params["lookback_period"])
    security_codes = [s.code for s in stock_list]
    security_pairs = list(itertools.combinations(security_codes, 2))

    # override security pairs
    if kwargs.get("security_pairs"):
        security_pairs = kwargs["security_pairs"]

    portfolio_pnl = None
    for i, security_pair in enumerate(security_pairs):
        # print(i, security_pair)
        pair_pnl = run_pairs(
            data=data,
            asset1=security_pair[0],
            asset2=security_pair[1],
            **params
        )
        pair_pnl.name = "|".join(security_pair)
        if portfolio_pnl is None:
            portfolio_pnl = pair_pnl.to_frame()
        else:
            portfolio_pnl = portfolio_pnl.join(pair_pnl, how="outer")
    portfolio_pnl = portfolio_pnl.ffill().bfill()
    portfolio_pnl["portfolio_value"] = portfolio_pnl.sum(axis=1)
    portfolio_pnl["datetime"] = portfolio_pnl.index
    return portfolio_pnl[["datetime", "portfolio_value"]]

if __name__ == "__main__":
    df = run_strategy(
        override_indicator_cfg=
        {'params':
             {'entry_threshold': 1.4055718649894686,
              'exit_threshold': 3.1614296858576507}
         },
    )
    print()
