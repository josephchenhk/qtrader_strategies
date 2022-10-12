# -*- coding: utf-8 -*-
# @Time    : 10/9/2022 11:36 am
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: optimization_pairs.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
from typing import Tuple, Dict, List
import multiprocessing
import pickle
from functools import partial
from datetime import datetime

import numpy as np

from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK

from qtrader.plugins.analysis.metrics import sharpe_ratio, rolling_maximum_drawdown
from qtrader.core.utility import timeit

from main_pairs_crypto import run_strategy
# from pandas_pairs import run_strategy

SEED = 2022

# define an objective function
def objective(args, **kwargs):
    (case, recalibration_lookback_ratio, ma_short_length) = args
    if case == 'case 1':
        df = run_strategy(
            security_pairs=kwargs.get("security_pairs"),
            start=kwargs.get("start"),
            end=kwargs.get("end"),
            override_indicator_cfg=
            {'params':
                 {'recalibration_lookback_ratio': recalibration_lookback_ratio,
                  'ma_short_length': ma_short_length,
                 }
             },
        )
        df_daily = df.set_index('datetime').resample('D').agg(
            {"portfolio_value": "last"}).dropna()
        sr = sharpe_ratio(
            returns=(
                df_daily["portfolio_value"].diff()
                / df_daily["portfolio_value"].iloc[0]).dropna(),
            days=365
        )
        sr = -np.Inf if np.isnan(sr) else sr
        tot_r = df_daily["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1.0
        mdd = rolling_maximum_drawdown(
            portfolio_value=df_daily["portfolio_value"].to_numpy(),
            window=365
        ).iloc[-1]
        if mdd == 0:
            RoMaD = -np.Inf
        else:
            RoMaD = -tot_r / mdd
        return {
            'loss': -min(max(sr, 0), 1.0) * tot_r,
            'status': STATUS_OK,
            'sharpe_ratio': sr,
            'total_return': tot_r,
            'maximum_drawdown': mdd,
            'return_over_maximum_drawdown': RoMaD
        }


def worker(
        security_pairs: List[Tuple[str]],
        start: datetime,
        end: datetime,
) -> Dict[str, float]:
    """Process that run the optimization for a pair"""
    trials = Trials()
    best = timeit(fmin)(
        partial(objective,
                security_pairs=security_pairs,
                start=start,
                end=end),
        space,
        algo=tpe.suggest,
        max_evals=40,
        trials=trials,
        rstate=np.random.default_rng(SEED)
    )
    opt_params = {}
    for security_pair in security_pairs:
        opt_params[security_pair] = {
            'recalibration_lookback_ratio': best['recalibration_lookback_ratio'],
            'ma_short_length': ma_short_length_choice[best['ma_short_length']],
            'best_loss': trials.best_trial['result']['loss'],
            'sharpe_ratio': trials.best_trial['result']['sharpe_ratio'],
            'total_return': trials.best_trial['result']['total_return'],
            'maximum_drawdown': trials.best_trial['result']['maximum_drawdown'],
            'return_over_maximum_drawdown': trials.best_trial['result'][
                'return_over_maximum_drawdown'],
        }

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    # Save to pkl file
    with open(
            f"opt_params/opt_params_trials_{start_str}_{end_str}.pkl",
            "wb") as f:
        pickle.dump(trials.trials, f)
    print(f"Optimization trials for {start_str}-{end_str} is done.")

    # Save to pkl file
    with open(
            f"opt_params/opt_params_{start_str}_{end_str}.pkl",
            "wb") as f:
        pickle.dump(opt_params, f)
    print(f"Optimization for {start_str}-{end_str} is done.")
    return opt_params

# define a search space
ma_short_length_choice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
space = hp.choice('a', [
      ('case 1',
       hp.uniform('recalibration_lookback_ratio', 0.05, 0.30),
       hp.choice('ma_short_length', ma_short_length_choice),
       )]
)

# minimize the objective over the space
security_pairs_lst = [
    ("BTC.USD", "LTC.USD"),  # 0.7410557870708834
    ("EOS.USD", "LTC.USD"),  # 0.7968610662268301
    ("EOS.USD", "TRX.USD"),  # 0.7397461563710355
    ("EOS.USD", "XRP.USD"),  # 0.7456244422109919
    ("ETH.USD", "TRX.USD"),  # 0.819769004085257
    ("ETH.USD", "XRP.USD"),  # 0.8416744722298698
    ("TRX.USD", "XRP.USD"),  # 0.9535877879201461
]

if __name__ == "__main__":

    dates = [
        (datetime(2021, 1, 1), datetime(2022, 1, 1)),
        (datetime(2021, 3, 1), datetime(2022, 3, 1)),
        (datetime(2021, 5, 1), datetime(2022, 5, 1)),
        (datetime(2021, 7, 1), datetime(2022, 7, 1)),
    ]

    manager = multiprocessing.Manager()
    jobs = []
    for start, end in dates:
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        # Parallel
        p = multiprocessing.Process(
            target=worker,
            args=(security_pairs_lst, start, end))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(f"Optimization is all done.")
