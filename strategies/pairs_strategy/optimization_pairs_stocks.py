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

from main_pairs_stocks import run_strategy
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
            override_indicator_cfg={
                'params': {
                    'recalibration_lookback_ratio': recalibration_lookback_ratio,
                    'ma_short_length': ma_short_length,
                }},
        )
        df_daily = df.set_index('datetime').resample('D').agg(
            {"portfolio_value": "last"}).dropna()
        sr = sharpe_ratio(
            returns=(
                df_daily["portfolio_value"].diff()
                / df_daily["portfolio_value"].iloc[0]).dropna(),
            days=256
        )
        sr = -np.Inf if np.isnan(sr) else sr
        tot_r = df_daily["portfolio_value"].iloc[-1] / \
            df["portfolio_value"].iloc[0] - 1.0
        mdd = rolling_maximum_drawdown(
            portfolio_value=df_daily["portfolio_value"].to_numpy(),
            window=256
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
        max_evals=25,
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
    # Save trials to pkl
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
    ("HK.00175", "HK.02333"),  # 0.8130455424786539
    # ("HK.02015", "HK.09868"),  # 0.8104253037907465  HK.02015. We want 2000 data points, but only got 359.

    ("HK.00005", "HK.02388"),  # 0.9102538048030802
    ("HK.00011", "HK.02388"),  # 0.8712931344354583
    ("HK.00023", "HK.02356"),  # 0.8817014110514532
    ("HK.02388", "HK.02888"),  # 0.8204095658616172

    ("HK.00002", "HK.00006"),  # 0.8943476985909485
    ("HK.00003", "HK.00006"),  # 0.845261925751612
    ("HK.00006", "HK.02638"),  # 0.8784485296661352

    ("HK.00151", "HK.00322"),  # 0.8010002549765622
    ("HK.00345", "HK.00359"),  # 0.8237822886291755
    ("HK.00345", "HK.01458"),  # 0.8572304867616597
]

if __name__ == "__main__":

    dates = [
        (datetime(2021, 3, 1), datetime(2022, 3, 1)),
        (datetime(2021, 5, 1), datetime(2022, 5, 1)),
        (datetime(2021, 7, 1), datetime(2022, 7, 1)),
        (datetime(2021, 9, 1), datetime(2022, 9, 1)),
    ]

    manager = multiprocessing.Manager()
    jobs = []
    for start, end in dates:

        # # Sync
        # worker(security_pairs_lst, start, end)

        # Parallel
        p = multiprocessing.Process(
            target=worker,
            args=(security_pairs_lst, start, end))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print("Optimization is all done.")
