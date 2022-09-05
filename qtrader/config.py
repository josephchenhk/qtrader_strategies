# -*- coding: utf-8 -*-
# @Time    : 5/9/2022 9:04 AM
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: config.py


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

TIME_STEP = 15 * 60 * 1000  # time step in milliseconds

DATA_PATH = {
    "kline": "clean_data/k_line",  # "k1m" is must have
}

DATA_MODEL = {
    "kline": "Bar",
}

ACTIVATED_PLUGINS = ["analysis"]

AUTO_OPEN_PLOT = False