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

import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS

from qtrader.core.security import Security, Currency
from qtrader.core.constants import Exchange
from qtrader.core.data import _get_data


def zscore(series: pd.Series) -> pd.Series:
    """calculate z-score"""
    return (series - series.mean()) / np.std(series)

def load_data(
        stock_list: List[Security],
        start: datetime,
        end: datetime
) -> pd.DataFrame:
    """Load close prices"""
    all_data = pd.DataFrame(columns=[s.code for s in stock_list])
    for security in stock_list:
        data = _get_data(
            security=security,
            start=start,
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
    return all_data

def run_pairs(
        data: pd.DataFrame,
        asset1: str,
        asset2: str,
        entry_threshold: float = 1.5,
        exit_threshold: float = 2.0,
        lookback_period: int = 1440,
        capital_per_entry: float = 1000000
):
    """run backtest for a pair"""
    # (asset1, asset2) is the pair to test strategy
    # create a train dataframe of 2 assets
    train = pd.DataFrame()
    train['asset1'] = data[asset1]
    train['asset2'] = data[asset2]

    # create a dataframe for trading signals
    signals = pd.DataFrame()
    signals['asset1'] = train['asset1']
    signals['asset2'] = train['asset2']
    model = RollingOLS(
        endog=np.log(signals['asset1']),
        exog=np.log(signals['asset2']),
        window=lookback_period,
    )
    model_fit = model.fit()
    signals["gamma"] = model_fit.params
    signals["spread"] = np.log(signals['asset1']) - signals["gamma"] * np.log(signals['asset2'])

    # calculate z-score and define upper and lower thresholds
    signals['z'] = zscore(signals["spread"])

    # create entry signal - short if z-score is greater than upper limit else long
    signals['entry_signals'] = 0
    signals['entry_signals'] = np.select(
        [(entry_threshold < signals['z'])
         & (signals['z'] < (entry_threshold + exit_threshold)/2)
         & (signals['gamma'] < 10)
         & (signals['gamma'] > 0.1),
         (-(entry_threshold + exit_threshold)/2 < signals['z'])
         & (signals['z'] < -entry_threshold)
         & (signals['gamma'] < 10)
         & (signals['gamma'] > 0.1)],
        [-1, 1],
        default=0)

    # Create exit signal
    signals['exit_signals'] = 0
    signals['exit_signals'] = np.where(
         (exit_threshold < signals['z'])
         | (-exit_threshold > signals['z'])
         | (signals['gamma'] < 0),
         1, 0)

    # shares to buy for each position
    q1 = capital_per_entry // signals['asset1']
    q2 = capital_per_entry // (signals['asset2'] * signals['gamma'])
    signals['qty1'] = np.select(
        [q1 <= q2, q1 > q2],
        [q1, q2 // signals['gamma']],
        default=0)
    signals['qty2'] = np.select(
        [q1 <= q2, q1 > q2],
        [(q1 * signals['gamma']).apply(
            lambda x: int(x) if not np.isnan(x) else x), q2],
        default=0)

    # calculate position and pnl
    position = [0] * signals.shape[0]
    qty1 = [0] * signals.shape[0]
    qty2 = [0] * signals.shape[0]
    for i, (timestamp, row) in enumerate(signals.iterrows()):
        entry_signals = row['entry_signals']
        exit_signals = row['exit_signals']

        if entry_signals and exit_signals:
            raise ValueError("entry and exit signals coexist!")

        if i == 0:
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
        elif prev_position != 0 and exit_signals:
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
        if i == 0:
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


def run_strategy(override_indicator_cfg=None):
    """Run strategy for portfolio"""
    if override_indicator_cfg is not None:
        entry_threshold = override_indicator_cfg['params']['entry_threshold']
        exit_threshold = override_indicator_cfg['params']['exit_threshold']
    else:
        entry_threshold = 1.5
        exit_threshold = 2.5
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
    start = datetime(2021, 12, 15, 0, 0, 0)
    end = datetime(2022, 8, 10, 23, 59, 59)

    # Load data
    data = load_data(stock_list, start, end)
    security_codes = [s.code for s in stock_list]
    security_pairs = list(itertools.combinations(security_codes, 2))

    portfolio_pnl = None
    for security_pair in security_pairs:
        # print(security_pair)
        pair_pnl = run_pairs(
            data=data,
            asset1=security_pair[0],
            asset2=security_pair[1],
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
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
             {'entry_threshold': 1.4,
              'exit_threshold': 2.3,
              }
         },
    )
    print()
