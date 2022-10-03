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
from qtrader.core.security import Currency, Futures
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

    # stock_list = [
    #     Futures(
    #         code="FUT.GC",
    #         lot_size=100,
    #         security_name="GCQ2",
    #         exchange=Exchange.NYMEX,
    #         expiry_date="20220828"),
    #     Futures(
    #         code="FUT.SI",
    #         lot_size=5000,
    #         security_name="SIU2",
    #         exchange=Exchange.NYMEX,
    #         expiry_date="20220927"),
    # ]
    #
    # stock_list = [
    #     Futures(
    #         code="HK.MHImain",
    #         lot_size=10,
    #         security_name="HK.MHImain",
    #         exchange=Exchange.HKFE,
    #         expiry_date="20220930"),
    #     Futures(
    #         code="HK.HHImain",
    #         lot_size=50,
    #         security_name="HK.HHImain",
    #         exchange=Exchange.HKFE,
    #         expiry_date="20220930"),
    # ]

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
            "FUT.GC": [[datetime(1970, 1, 1, 10, 0, 0), datetime(1970, 1, 1, 5, 0, 0)],
                       [datetime(1970, 1, 1, 7, 0, 0), datetime(1970, 1, 1, 9, 45, 0)]],
            "FUT.SI": [[datetime(1970, 1, 1, 10, 0, 0), datetime(1970, 1, 1, 5, 0, 0)],
                       [datetime(1970, 1, 1, 7, 0, 0), datetime(1970, 1, 1, 9, 45, 0)]],
            "HK.MHImain": [
                [datetime(1970, 1, 1, 10, 0, 0),
                 datetime(1970, 1, 1, 12, 0, 0)],
                [datetime(1970, 1, 1, 13, 0, 0),
                 datetime(1970, 1, 1, 16, 30, 0)],
                [datetime(1970, 1, 1, 17, 15, 0),
                 datetime(1970, 1, 1, 3, 0, 0)],
                [datetime(1970, 1, 1, 9, 15, 0),
                 datetime(1970, 1, 1, 9, 45, 0)],
            ],
            "HK.HHImain": [
                [datetime(1970, 1, 1, 10, 0, 0),
                 datetime(1970, 1, 1, 12, 0, 0)],
                [datetime(1970, 1, 1, 13, 0, 0),
                 datetime(1970, 1, 1, 16, 30, 0)],
                [datetime(1970, 1, 1, 17, 15, 0),
                 datetime(1970, 1, 1, 3, 0, 0)],
                [datetime(1970, 1, 1, 9, 15, 0),
                 datetime(1970, 1, 1, 9, 45, 0)],
            ],
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
            "XRP.USD": trading_sessions,
            "FUT.GC": [
                [datetime(1970, 1, 1, 10, 0, 0), datetime(1970, 1, 1, 5, 0, 0)],
                [datetime(1970, 1, 1, 7, 0, 0),
                 datetime(1970, 1, 1, 9, 45, 0)]],
            "FUT.SI": [
                [datetime(1970, 1, 1, 10, 0, 0), datetime(1970, 1, 1, 5, 0, 0)],
                [datetime(1970, 1, 1, 7, 0, 0),
                 datetime(1970, 1, 1, 9, 45, 0)]],
            "HK.MHImain": [
                [datetime(1970, 1, 1, 10, 0, 0),
                 datetime(1970, 1, 1, 12, 0, 0)],
                [datetime(1970, 1, 1, 13, 0, 0),
                 datetime(1970, 1, 1, 16, 30, 0)],
                [datetime(1970, 1, 1, 17, 15, 0),
                 datetime(1970, 1, 1, 3, 0, 0)],
                [datetime(1970, 1, 1, 9, 15, 0),
                 datetime(1970, 1, 1, 9, 45, 0)],
            ],
            "HK.HHImain": [
                [datetime(1970, 1, 1, 10, 0, 0),
                 datetime(1970, 1, 1, 12, 0, 0)],
                [datetime(1970, 1, 1, 13, 0, 0),
                 datetime(1970, 1, 1, 16, 30, 0)],
                [datetime(1970, 1, 1, 17, 15, 0),
                 datetime(1970, 1, 1, 3, 0, 0)],
                [datetime(1970, 1, 1, 9, 15, 0),
                 datetime(1970, 1, 1, 9, 45, 0)],
            ],
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
    # security_pairs_lst = [
    #     ('FUT.GC', 'FUT.SI'),
    # ]
    # security_pairs_lst = [
    #     ('HK.MHImain', 'HK.HHImain'),
    # ]

    start = datetime(2022, 1, 1, 0, 0, 0)
    end = datetime(2022, 8, 1, 0, 0, 0)
    # start = datetime(2021, 1, 1)
    # end = datetime(2021, 7, 1)
    df = run_strategy(
        security_pairs=security_pairs_lst,
        start=start,
        end=end
    )
    print("Backtest is done.")