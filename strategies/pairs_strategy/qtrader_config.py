# -*- coding: utf-8 -*-
# @Time    : 6/9/2022 7:53 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: qtrader_config.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""

BACKTEST_GATEWAY = {
    "broker_name": "BACKTEST",
    "broker_account": "",
    "host": "",
    "port": -1,
    "pwd_unlock": -1,
}

GATEWAYS = {
    "Backtest": BACKTEST_GATEWAY,
}

TIME_STEP = 60 * 60 * 1000  # time step in milliseconds

DATA_PATH = {
    # "kline": "/Users/joseph/Dropbox/code/demo_strategy/clean_data/k_line",
    # "kline": "/Users/joseph/Dropbox/code/data/data/k_line",
    "kline": "C:/Users/josephchenj/Quant/demo_strategy/clean_data/k_line",
}

DATA_MODEL = {
    "kline": "Bar",
}

ACTIVATED_PLUGINS = ["analysis"]

AUTO_OPEN_PLOT = False
IGNORE_TIMESTEP_OVERFLOW = False
DATA_FFILL = True