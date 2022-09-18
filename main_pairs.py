# -*- coding: utf-8 -*-
# @Time    : 9/6/2022 4:45 PM
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: main_pairs.py

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
import pickle
import itertools
from datetime import datetime

import pandas as pd

from qtrader.core.position import Position
from qtrader.core.engine import Engine
from qtrader.core.security import Currency
from qtrader.core.event_engine import BarEventEngineRecorder, BarEventEngine
from qtrader.core.constants import TradeMode, Exchange
from qtrader.gateways import BacktestGateway
from qtrader.gateways.backtest import BacktestFees
from strategies.pairs_strategy import PairsStrategy


def run_strategy(**kwargs):
    trading_sessions = [
        [datetime(1970, 1, 1, 0, 0, 0),
         datetime(1970, 1, 1, 23, 59, 59)],
    ]
    gateway_name = "Backtest"
    start = datetime(2021, 1, 1, 0, 0, 0)
    end = datetime(2022, 1, 1, 0, 0, 0)
    # start = datetime(2022, 1, 1, 0, 0, 0)
    # end = datetime(2022, 8, 1, 0, 0, 0)

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
            "XRP.USD": trading_sessions
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
    init_capital = 1000000 * len(security_pairs)

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
            "XRP.USD": trading_sessions
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
    with open("opt_params.pkl", "rb") as f:
        opt_params = pickle.load(f)
        # select securities to trade
        opt_params = {k: v for k, v in opt_params.items() if v["best_loss"] < 0}
        security_pairs = [k for k, v in opt_params.items() if v["best_loss"] < 0]
        df = run_strategy(
            # override_indicator_cfg=
            # {'params':
            #      {'entry_threshold': 1.9512880891669317,
            #       'exit_threshold': 2.1816070388851037}
            #  },
            opt_params=opt_params,
            security_pairs=security_pairs
        )
    print("Backtest is done.")