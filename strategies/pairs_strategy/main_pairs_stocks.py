# -*- coding: utf-8 -*-
# @Time    : 9/6/2022 4:45 PM
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: main_pairs_crypto.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""

##########################################################################
#                                                                        #
#                    Pairs Trading strategy                              #
##########################################################################

"""
{
  "lookback_period": 2000,
  "recalibration_lookback_ratio": 0.181,
  "corr_init_threshold": 0.85,
  "corr_maintain_threshold": 0.65,
  "coint_pvalue_init_threshold": 0.01,
  "coint_pvalue_maintain_threshold": 0.05,
  "entry_threshold_pct": 0.75,
  "exit_threshold_pct": 0.99,
  "max_number_of_entry": 1,
  "capital_per_entry": 5000000,
  "ma_short_length": 140,
  "ma_long_length": 200
}
"""
import json
import itertools
from datetime import datetime

import pandas as pd

from qtrader.core.position import Position
from qtrader.core.engine import Engine
from qtrader.core.security import Stock
from qtrader.core.event_engine import BarEventEngineRecorder, BarEventEngine
from qtrader.core.constants import TradeMode, Exchange
from qtrader.gateways import BacktestGateway
from qtrader.gateways.backtest import BacktestFees
from pairs_strategy_v2 import PairsStrategy

with open("params.json", "r") as f:
    params = json.load(f)

def run_strategy(**kwargs):

    trading_sessions = [
        [datetime(1970, 1, 1, 9, 30, 0),
         datetime(1970, 1, 1, 12, 0, 0)],
        [datetime(1970, 1, 1, 13, 0, 0),
         datetime(1970, 1, 1, 16, 0, 0)],
    ]
    gateway_name = "Backtest"

    if "start" in kwargs:
        start = kwargs["start"]
    else:
        start = datetime(2022, 1, 1, 0, 0, 0)

    if "end" in kwargs:
        end = kwargs["end"]
    else:
        end = datetime(2022, 2, 1, 0, 0, 0)

    stock_list = [
        Stock(code='HK.00175', lot_size=1000, security_name='吉利汽车'),
        Stock(code='HK.00305', lot_size=10000, security_name='五菱汽车'),
        Stock(code='HK.00489', lot_size=2000, security_name='东风集团股份'),
        # Stock(code='HK.00708', lot_size=500, security_name='恒大汽车'),
        # Stock(code='HK.01114', lot_size=2000, security_name='华晨中国'),
        Stock(code='HK.01122', lot_size=2000, security_name='庆铃汽车股份'),
        Stock(code='HK.01211', lot_size=500, security_name='比亚迪股份'),
        Stock(code='HK.01958', lot_size=500, security_name='北京汽车'),
        # Stock(code='HK.02015', lot_size=100, security_name='理想汽车-W'),
        Stock(code='HK.02238', lot_size=2000, security_name='广汽集团'),
        Stock(code='HK.02333', lot_size=500, security_name='长城汽车'),
        # Stock(code='HK.09863', lot_size=100, security_name='零跑汽车'),
        # Stock(code='HK.09866', lot_size=10, security_name='蔚来-SW'),
        # Stock(code='HK.09868', lot_size=100, security_name='小鹏汽车-W'),

        # Stock(code='HK.00005', lot_size=400, security_name='汇丰控股'),
        # Stock(code='HK.00011', lot_size=100, security_name='恒生银行'),
        # Stock(code='HK.00023', lot_size=200, security_name='东亚银行'),
        # Stock(code='HK.02356', lot_size=400, security_name='大新银行集团'),
        # Stock(code='HK.02388', lot_size=500, security_name='中银香港'),
        # Stock(code='HK.02888', lot_size=50, security_name='渣打集团')
    ]

    security_pairs = kwargs.get("security_pairs")
    if security_pairs:
        security_codes = []
        [security_codes.extend(p) for p in security_pairs]
        security_codes = list(set(security_codes))
        stock_list = [s for s in stock_list if s.code in security_codes]
    else:
        security_pairs = list(
            itertools.combinations([s.code for s in stock_list], 2))

    gateway = BacktestGateway(
        securities=stock_list,
        start=start,
        end=end,
        gateway_name=gateway_name,
        fees=BacktestFees,
        trading_sessions={
            security.code: trading_sessions for security in stock_list},
    )

    gateway.SHORT_INTEREST_RATE = 0.0
    gateway.trade_mode = TradeMode.BACKTEST

    # Execution engine
    engine = Engine(gateways={gateway_name: gateway})

    # Initialize strategies
    strategy_account = "PairStrategy"
    strategy_version = "1.0"
    init_position = Position()
    init_capital = params["capital_per_entry"] * len(security_pairs)

    strategy = PairsStrategy(
        securities={gateway_name: stock_list},
        strategy_account=strategy_account,
        strategy_version=strategy_version,
        engine=engine,
        strategy_trading_sessions={
            security.code: trading_sessions for security in stock_list},
        init_strategy_cash={gateway_name: init_capital},
        init_strategy_position={gateway_name: init_position},
        **kwargs
    )

    strategy.init_strategy()

    # Recorder
    recorder = BarEventEngineRecorder(datetime=[],
                                      bar_datetime=[],
                                      open=[],
                                      high=[],
                                      low=[],
                                      close=[],
                                      volume=[])

    # Event engine
    event_engine = BarEventEngine(
        {"pairs": strategy},
        {"pairs": recorder},
        engine
    )

    # Event loop
    event_engine.run()

    # Program terminates normally
    engine.log.info("Program shutdown normally.")

    return pd.DataFrame({
        "datetime": [datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") for d in recorder.datetime],
        "portfolio_value": [p[0] for p in recorder.strategy_portfolio_value]}
    )

if __name__ == "__main__":

    if params["load_params"] == 1:
        # Testing period
        start = datetime(2022, 1, 1, 0, 0, 0)
        end = datetime(2022, 8, 1, 0, 0, 0)
    else:
        # Training period
        start = datetime(2021, 4, 1, 0, 0, 0)
        end = datetime(2022, 1, 1, 0, 0, 0)
    df = run_strategy(
        start=start,
        end=end
    )
    print("Backtest is done.")