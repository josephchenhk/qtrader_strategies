# -*- coding: utf-8 -*-
# @Time    : 17/3/2021 3:56 PM
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: cta_strategy.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
import os
from time import sleep
from typing import Dict, List
from datetime import datetime
from enum import Enum
import itertools
import json

import numpy as np

from qtrader.core.constants import Direction, Offset, OrderType, OrderStatus
from qtrader.core.data import Bar
from qtrader.core.engine import Engine
from qtrader.core.position import Position
from qtrader.core.security import Security
from qtrader.core.strategy import BaseStrategy
from qtrader_config import TIME_STEP

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Signal(Enum):
    ENTRY_LONG = 1
    ENTRY_SHORT = 2
    ENTRY_ADD_LONG = 3
    ENTRY_ADD_SHORT = 4
    EXIT_LONG = 5
    EXIT_SHORT = 6
    STOP_LONG = 7
    STOP_SHORT = 8
    NO_SIGNAL = 9
    TIME_STOP_LONG = 10
    TIME_STOP_SHORT = 11


class PairStrategy(BaseStrategy):
    """Pairs Trading strategy"""

    def __init__(self,
                 securities: Dict[str, List[Security]],
                 strategy_account: str,
                 strategy_version: str,
                 engine: Engine,
                 strategy_trading_sessions: List[List[datetime]] = None,
                 init_strategy_cash: Dict[str, float] = None,
                 init_strategy_position: Dict[str, Position] = None
                 ):
        super().__init__(
            securities=securities,
            strategy_account=strategy_account,
            strategy_version=strategy_version,
            engine=engine,
            strategy_trading_sessions=strategy_trading_sessions,
            init_strategy_cash=init_strategy_cash,
            init_strategy_position=init_strategy_position
        )

    def init_strategy(self):
        # initialize strategy parameters
        self.ohlcv = {}
        self.lookback_period = {}
        for gateway_name in self.securities:
            self.ohlcv[gateway_name] = {}
            self.lookback_period[gateway_name] = {}
            for security in self.securities[gateway_name]:
                self.ohlcv[gateway_name][security] = None
                self.lookback_period[gateway_name][security] = None

        # Load strategy parameters
        with open("strategies/pair_strategy/params.json", "rb") as f:
            self.params = json.load(f)

        # Get security pairs
        security_codes = []
        for gateway_name in self.securities:
            security_codes.extend(
                [s.code for s in self.securities[gateway_name]])
            for security in self.securities[gateway_name]:
                self.lookback_period[gateway_name][security] = self.params["lookback_period"]
                self.request_historical_ohlcv(
                    gateway_name=gateway_name,
                    security=security
                )
        self.security_pairs = list(itertools.combinations(security_codes, 2))

    def request_historical_ohlcv(self, gateway_name: str, security: Security):
        # Request the information every day
        self.ohlcv[gateway_name][security] = self.engine.req_historical_bars(
            security=security,
            gateway_name="Backtest",
            periods=self.lookback_period[gateway_name][security],
            freq=f"{int(TIME_STEP / 60000.)}Min",
            cur_datetime=self.engine.gateways[gateway_name].market_datetime,
            trading_sessions=self.engine.gateways[gateway_name].trading_sessions[
                security.code],
        )

    def on_bar(self, cur_data: Dict[str, Dict[Security, Bar]]):

        self.engine.log.info(
            "-" * 30
            + "Enter <PairStrategy> on_bar"
            + "-" * 30
        )
        self.engine.log.info(cur_data)
        self._cur_data = cur_data

        for gateway_name in self.engine.gateways:

            if gateway_name not in cur_data:
                continue

            self.engine.log.info(
                f"strategy portfolio value: {self.get_strategy_portfolio_value(gateway_name)}")

            # Find possible opportunities in every pair
            for security_pair in self.security_pairs:

                security1, security2 = list(map(
                    self.get_security_from_security_code, security_pair))
                bar1 = cur_data[gateway_name].get(security1)
                bar2 = cur_data[gateway_name].get(security2)

                if (
                    bar1 is not None
                    and bar1.datetime > self.ohlcv[gateway_name][security1][-1].datetime
                ):
                    self.ohlcv[gateway_name][security1].append(bar1)
                    while (
                        len(self.ohlcv[gateway_name][security1])
                        > self.params["lookback_period"]
                    ):
                        self.ohlcv[gateway_name][security1].pop(0)
                if (
                    bar2 is not None
                    and bar2.datetime > self.ohlcv[gateway_name][security2][-1].datetime
                ):
                    self.ohlcv[gateway_name][security2].append(bar1)
                    while (
                        len(self.ohlcv[gateway_name][security2])
                        > self.params["lookback_period"]
                    ):
                        self.ohlcv[gateway_name][security2].pop(0)

                if (
                    len(self.ohlcv[gateway_name][security1]) < self.params["lookback_period"]
                    or len(self.ohlcv[gateway_name][security2]) < self.params["lookback_period"]
                ):
                    continue

                corr = np.corrcoef(
                    [b.close for b in self.ohlcv[gateway_name][security1]],
                    [b.close for b in self.ohlcv[gateway_name][security2]]
                )[0][1]

                if corr > self.params["correlation_threshold"]:
                    print()

    def send_order(self, security: Security,
                   quantity: int,
                   direction: Direction,
                   offset: Offset,
                   order_type: OrderType,
                   gateway_name: str
                   ) -> bool:
        """return True if order is successfully fully filled, else False"""
        order_instruct = dict(
            security=security,
            quantity=quantity,
            direction=direction,
            offset=offset,
            order_type=order_type,
            gateway_name=gateway_name,
        )

        self.engine.log.info(f"提交订单:\n{order_instruct}")
        orderid = self.engine.send_order(**order_instruct)
        # TODO: sometimes timeout here.
        if orderid == "":
            self.engine.log.info("提交订单失败")
            return False
        self.engine.log.info(f"订单{orderid}已发出")
        sleep(self.sleep_time)
        order = self.engine.get_order(
            orderid=orderid, gateway_name=gateway_name)
        self.engine.log.info(f"订单{orderid}详情:{order}")

        deals = self.engine.find_deals_with_orderid(
            orderid, gateway_name=gateway_name)
        for deal in deals:
            self.engine.portfolios[gateway_name].update(deal)
            self.portfolios[gateway_name].update(deal)

        if order.status == OrderStatus.FILLED:
            self.engine.log.info(f"订单已成交{orderid}")
            return True
        else:
            err = self.engine.cancel_order(
                orderid=orderid, gateway_name=gateway_name)
            if err:
                self.engine.log.info(f"不能取消订单{orderid},因爲{err}")
            else:
                self.engine.log.info(f"已經取消订单{orderid}")
            return False

    def get_security_from_security_code(self, security_code: str) -> Security:
        for gateway_name in self.securities:
            for security in self.securities[gateway_name]:
                if security.code == security_code:
                    return security
        return None
