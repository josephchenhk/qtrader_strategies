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
# from main_pairs import run_strategy
from pandas_pairs import run_strategy
from qtrader.plugins.analysis.metrics import sharp_ratio
from qtrader.core.utility import timeit

# define an objective function
def objective(args):
    (case, entry_threshold, exit_threshold) = args
    if case == 'case 1':
        df = run_strategy(
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
        return -min(max(sr, 0), 1.8) * tot_r

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1',
            hp.uniform('entry_threshold', 0.5, 2.0),
            hp.uniform('exit_threshold', 2.01, 4.0),
         )
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe
best = timeit(fmin)(objective, space, algo=tpe.suggest, max_evals=25)
print(best)
print()