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

from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK

from main_pairs import run_strategy
# from pandas_pairs import run_strategy
from qtrader.plugins.analysis.metrics import sharp_ratio
from qtrader.core.utility import timeit


# define an objective function
def objective(args, **kwargs):
    (case, entry_threshold, exit_threshold) = args
    if case == 'case 1':
        df = run_strategy(
            start=kwargs["start"],
            end=kwargs["end"],
            security_pairs=kwargs.get("security_pairs"),
            override_indicator_cfg=
            {'params':
                 {'entry_threshold': entry_threshold,
                  'exit_threshold': exit_threshold,
                 }
             },
        )
        df_daily = df.set_index('datetime').resample('D').agg(
            {"portfolio_value": "last"}).dropna()
        sr = sharp_ratio(
            returns=df_daily["portfolio_value"].pct_change().dropna().to_numpy()
        )
        tot_r = df_daily["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1.0
        return {
            'loss': -min(max(sr, 0), 0.5) * tot_r,
            'status': STATUS_OK,
            'sharpe_ratio': sr,
            'total_return': tot_r
        }

def worker(
        security_pairs: List[Tuple[str]],
        start: datetime,
        end: datetime,
        return_dict:  Dict[Tuple, Dict]
) -> Dict[str, float]:
    """Process optimization"""
    trials = Trials()
    best = timeit(fmin)(
        partial(objective,
                security_pairs=security_pairs,
                start=start,
                end=end),
        space,
        algo=tpe.suggest,
        max_evals=12,
        trials=trials
    )
    for security_pair in security_pairs_lst:
        return_dict[security_pair] = {
            'entry_threshold': best['entry_threshold'],
            'exit_threshold': best['exit_threshold'],
            'best_loss': trials.best_trial['result']['loss'],
            'sharpe_ratio': trials.best_trial['result']['sharpe_ratio'],
            'total_return': trials.best_trial['result']['total_return'],
        }
    opt_params = {k: v for k, v in return_dict.items()}
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    with open(f"opt_params_{start_str}_{end_str}.pkl", "wb") as f:
        pickle.dump(opt_params, f)
    print(f"Optimization for {start_str}-{end_str} is done.")

    return best

# define a search space
space = hp.choice('a',
    [
        ('case 1',
            hp.uniform('entry_threshold', 1.0, 2.5),
            hp.uniform('exit_threshold', 2.6, 3.5),
         )
    ])

# minimize the objective over the space
security_pairs_lst = [
    ('BTC.USD', 'EOS.USD'),
    ('BTC.USD', 'ETH.USD'),
    ('BTC.USD', 'LTC.USD'),
    ('BTC.USD', 'TRX.USD'),
    ('BTC.USD', 'XRP.USD'),
    ('EOS.USD', 'ETH.USD'),
    ('EOS.USD', 'LTC.USD'),
    ('EOS.USD', 'TRX.USD'),
    ('EOS.USD', 'XRP.USD'),
    ('ETH.USD', 'LTC.USD'),
    ('ETH.USD', 'TRX.USD'),
    ('ETH.USD', 'XRP.USD'),
    ('LTC.USD', 'TRX.USD'),
    ('LTC.USD', 'XRP.USD'),
    ('TRX.USD', 'XRP.USD')
]

if __name__ == "__main__":
    # Parallel
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    dates = [
        (datetime(2021, 7, 1), datetime(2022, 1, 1)),
        (datetime(2021, 8, 1), datetime(2022, 2, 1)),
        (datetime(2021, 9, 1), datetime(2022, 3, 1)),
        (datetime(2021, 10, 1), datetime(2022, 4, 1)),
        (datetime(2021, 11, 1), datetime(2022, 5, 1)),
        (datetime(2021, 12, 1), datetime(2022, 6, 1)),
        (datetime(2022, 1, 1), datetime(2022, 7, 1)),
    ]
    for start, end in dates:
        p = multiprocessing.Process(
            target=worker,
            args=(security_pairs_lst, start, end, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print("Optimization is done.")
