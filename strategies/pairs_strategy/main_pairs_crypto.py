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
  "lookback_period": 4000,
  "recalibration_lookback_ratio": 0.161,
  "corr_init_threshold": 0.85,
  "corr_maintain_threshold": 0.65,
  "coint_pvalue_init_threshold": 0.01,
  "coint_pvalue_maintain_threshold": 0.10,
  "entry_threshold_pct": 0.75,
  "exit_threshold_pct": 0.99,
  "max_number_of_entry": 1,
  "capital_per_entry": 1000000,
  "ma_short_length": 10,
  "ma_long_length": 200,
  "load_params": 1
}
"""
import json
import itertools
from datetime import datetime

import pandas as pd

from qtrader.core.position import Position
from qtrader.core.engine import Engine
from qtrader.core.security import Currency, Futures
from qtrader.core.event_engine import BarEventEngineRecorder, BarEventEngine
from qtrader.core.constants import TradeMode, Exchange
from qtrader.gateways import BacktestGateway
from qtrader.gateways.backtest import BacktestFees
from pairs_strategy_v2 import PairsStrategy

with open("params.json", "r") as f:
    params = json.load(f)

def run_strategy(**kwargs):

    trading_sessions = [
        [datetime(1970, 1, 1, 0, 0, 0),
         datetime(1970, 1, 1, 23, 59, 59)],
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
            "BTC.USD": trading_sessions,
            "EOS.USD": trading_sessions,
            "ETH.USD": trading_sessions,
            "LTC.USD": trading_sessions,
            "TRX.USD": trading_sessions,
            "XRP.USD": trading_sessions,
        },
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
            "BTC.USD": trading_sessions,
            "EOS.USD": trading_sessions,
            "ETH.USD": trading_sessions,
            "LTC.USD": trading_sessions,
            "TRX.USD": trading_sessions,
            "XRP.USD": trading_sessions,
        },
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
    security_pairs_lst = [
        ("BTC.USD", "LTC.USD"),  # 0.7410557870708834
        ("EOS.USD", "LTC.USD"),  # 0.7968610662268301
        ("EOS.USD", "TRX.USD"),  # 0.7397461563710355
        ("EOS.USD", "XRP.USD"),  # 0.7456244422109919
        ("ETH.USD", "TRX.USD"),  # 0.819769004085257
        ("ETH.USD", "XRP.USD"),  # 0.8416744722298698
        ("TRX.USD", "XRP.USD"),  # 0.9535877879201461
    ]

    if params["load_params"] == 1:
        # Testing period
        start = datetime(2022, 1, 1, 0, 0, 0)
        end = datetime(2022, 8, 1, 0, 0, 0)
    else:
        # Training period
        start = datetime(2021, 1, 1, 0, 0, 0)
        end = datetime(2022, 1, 1, 0, 0, 0)

    df = run_strategy(
        security_pairs=security_pairs_lst,
        start=start,
        end=end
    )
    print("Backtest is done.")