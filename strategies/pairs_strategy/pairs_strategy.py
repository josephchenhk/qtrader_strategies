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
from time import sleep
from typing import Dict, List, Tuple
from datetime import datetime
from datetime import timedelta
import itertools
import json

import numpy as np
from statsmodels.tsa.stattools import adfuller

from qtrader.core.constants import Direction, Offset, OrderType, OrderStatus
from qtrader.core.data import Bar
from qtrader.core.engine import Engine
from qtrader.core.position import Position
from qtrader.core.security import Security
from qtrader.core.strategy import BaseStrategy
from qtrader_config import TIME_STEP


class PairsStrategy(BaseStrategy):
    """Pairs Trading strategy"""

    def __init__(self,
                 securities: Dict[str, List[Security]],
                 strategy_account: str,
                 strategy_version: str,
                 engine: Engine,
                 strategy_trading_sessions: List[List[datetime]] = None,
                 init_strategy_cash: Dict[str, float] = None,
                 init_strategy_position: Dict[str, Position] = None,
                 **kwargs
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

        # Load strategy parameters
        with open("strategies/pairs_strategy/params.json", "rb") as f:
            self.params = json.load(f)

        # Override default indicator config
        override_indicator_cfg = kwargs.get("override_indicator_cfg")
        if override_indicator_cfg:
            for cfg_name, cfg_value in override_indicator_cfg["params"].items():
                self.params[cfg_name] = cfg_value
        # print(indicator_cfg)
        print("Strategy loaded.")

    def init_strategy(self):

        # initialize strategy parameters
        self.sleep_time = 0
        self.recalibration_date = max(
            self.engine.gateways[gn].market_datetime
            for gn in self.securities
        )
        security_codes = {}
        self.ohlcv = {}
        self.lookback_period = {}
        self.security_pairs = {}
        self.security_pairs_number_of_entry = {}
        self.security_pairs_quantity_of_entry = {}
        for gateway_name in self.securities:
            security_codes[gateway_name] = [
                s.code for s in self.securities[gateway_name]]
            self.ohlcv[gateway_name] = {}
            self.lookback_period[gateway_name] = {}
            self.security_pairs[gateway_name] = list(
                itertools.combinations(security_codes[gateway_name], 2))
            self.security_pairs_number_of_entry[gateway_name] = {
                k: {"long1_short2": 0, "short1_long2": 0}
                for k in self.security_pairs[gateway_name]}
            self.security_pairs_quantity_of_entry[gateway_name] = {
                k: {"long1_short2": [], "short1_long2": []}
                for k in self.security_pairs[gateway_name]}
            for security in self.securities[gateway_name]:
                self.lookback_period[gateway_name][security] = self.params[
                    "lookback_period"]
                self.ohlcv[gateway_name][
                    security] = self.request_historical_ohlcv(
                    gateway_name=gateway_name, security=security)

    def on_bar(self, cur_data: Dict[str, Dict[Security, Bar]]):

        self.engine.log.info(
            "-" * 30
            + "Enter <PairStrategy> on_bar"
            + "-" * 30
        )
        self.engine.log.info(cur_data)
        self._cur_data = cur_data

        cur_datetime = self.get_current_datetime(cur_data)

        # Recalibrate the parameters
        if cur_datetime >= self.recalibration_date:

            # clear position if there is any
            for gateway_name in self.engine.gateways:
                if gateway_name not in cur_data:
                    continue
                position = self.portfolios[gateway_name].position
                position_data_lst = position.get_all_positions()
                for position_data in position_data_lst:
                    if position_data.direction == Direction.SHORT:
                        direction = Direction.LONG
                    elif position_data.direction == Direction.LONG:
                        direction = Direction.SHORT
                    self.send_order(
                        security=position_data.security,
                        quantity=position_data.quantity,
                        direction=direction,
                        offset=Offset.CLOSE,
                        order_type=OrderType.MARKET,
                        gateway_name=gateway_name
                    )
                for security_pair in self.security_pairs[gateway_name]:
                    self.security_pairs_number_of_entry[gateway_name][security_pair][
                        "long1_short2"] = 0
                    self.security_pairs_number_of_entry[gateway_name][security_pair][
                        "short1_long2"] = 0
                    self.security_pairs_quantity_of_entry[gateway_name][security_pair][
                        "long1_short2"] = []
                    self.security_pairs_quantity_of_entry[gateway_name][security_pair][
                        "short1_long2"] = []

            # find out candidate pairs
            self.candidate_security_pairs = []
            for gateway_name in self.engine.gateways:
                if gateway_name not in cur_data:
                    continue
                for security_pair in self.security_pairs[gateway_name]:
                    security1, security2 = list(map(
                        self.get_security_from_security_code, security_pair))
                    corr = np.corrcoef(
                        [np.log(b.close) for b in self.ohlcv[gateway_name][security1]],
                        [np.log(b.close) for b in self.ohlcv[gateway_name][security2]]
                    )[0][1]

                    if corr > self.params["correlation_threshold"]:
                        self.candidate_security_pairs.append(security_pair)
            self.recalibration_date += timedelta(
                days=self.params["recalibration_interval"]
            )

        for gateway_name in self.engine.gateways:

            if gateway_name not in cur_data:
                continue

            self.engine.log.info(
                f"strategy portfolio value: {self.get_strategy_portfolio_value(gateway_name)}\n"
                f"strategy positions: {self.portfolios[gateway_name].position}")

            # Find possible opportunities in every pair
            for security_pair in self.security_pairs[gateway_name]:

                security1, security2 = list(map(
                    self.get_security_from_security_code, security_pair))

                # Collect the ohlcv data
                bar1 = cur_data[gateway_name].get(security1)
                bar2 = cur_data[gateway_name].get(security2)
                if (bar1 is not None and bar1.datetime >
                        self.ohlcv[gateway_name][security1][-1].datetime):
                    self.ohlcv[gateway_name][security1].append(bar1)
                    while (
                        len(self.ohlcv[gateway_name][security1])
                        > self.params["lookback_period"]
                    ):
                        self.ohlcv[gateway_name][security1].pop(0)
                if (bar2 is not None and bar2.datetime >
                        self.ohlcv[gateway_name][security2][-1].datetime):
                    self.ohlcv[gateway_name][security2].append(bar2)
                    while (
                        len(self.ohlcv[gateway_name][security2])
                        > self.params["lookback_period"]
                    ):
                        self.ohlcv[gateway_name][security2].pop(0)

                if (
                    len(self.ohlcv[gateway_name][security1]) < self.params["lookback_period"]
                    or len(self.ohlcv[gateway_name][security2]) < self.params["lookback_period"]
                    or security_pair not in self.candidate_security_pairs
                ):
                    continue

                # Cointegration test
                # log(S1) = gamma * log(S2) + mu
                logS1 = np.array([np.log(b.close)
                                  for b in self.ohlcv[gateway_name][security1]])
                logS2 = np.array([np.log(b.close)
                                  for b in self.ohlcv[gateway_name][security2]])
                A = np.array([logS2, np.ones_like(logS2)])
                w = np.linalg.lstsq(A.T, logS1, rcond=None)[0]
                gamma, mu = w
                if gamma <= 0.1 or gamma >= 10:
                    continue
                # logS1_fit = gamma * logS2 + mu
                spread = logS1 - gamma * logS2
                adf = adfuller(spread, autolag="AIC")
                adf_pvalue = adf[1]
                if adf_pvalue > self.params["cointegration_pvalue_threshold"]:
                    continue

                spread_mean = spread.mean()
                spread_std = spread.std()

                # check maximum number of entries
                can_entry_long1_short2 = (
                    0 <= self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"]
                    < self.params["max_number_of_entry"]
                    and self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"] == 0
                )
                can_entry_short1_long2 = (
                    0 <= self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"]
                    < self.params["max_number_of_entry"]
                    and self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"] == 0
                )
                can_exit_long1_short2 = (
                    self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"] > 0)
                can_exit_short1_long2 = (
                    self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"] < 0)
                entry_short1_long2 = (
                    spread[-1] > spread_mean + spread_std * self.params["entry_threshold"]
                    and spread[-1] < spread_mean + spread_std * self.params["exit_threshold"]
                )
                entry_long1_short2 = (
                    spread[-1] < spread_mean - spread_std * self.params["entry_threshold"]
                    and spread[-1] > spread_mean - spread_std * self.params["exit_threshold"]
                )
                exit_short1_long2 = (
                    spread[-1] > spread_mean + spread_std * self.params["exit_threshold"]
                )
                exit_long1_short2 = (
                    spread[-1] < spread_mean - spread_std * self.params["exit_threshold"]
                )
                take_profit_short1_long2 = (
                    spread[-1] <= spread_mean
                )
                take_profit_long1_short2 = (
                    spread[-1] >= spread_mean
                )

                # Entry if meet the requirements
                if (
                        can_entry_long1_short2
                        and entry_long1_short2
                ):
                    self.engine.log.info("entry_long1_short2:")
                    self.entry_long1_short2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["long1_short2"],
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        security1=security1,
                        bar1=bar1,
                        security2=security2,
                        bar2=bar2,
                        gamma=gamma
                    )
                elif (
                        can_entry_short1_long2
                        and entry_short1_long2
                ):
                    self.engine.log.info("entry_short1_long2")
                    self.entry_short1_long2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["short1_long2"],
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        security1=security1,
                        bar1=bar1,
                        security2=security2,
                        bar2=bar2,
                        gamma=gamma
                    )
                elif (
                        can_exit_long1_short2
                        and exit_long1_short2
                ):
                    self.engine.log.info("exit_long1_short2")
                    self.exit_long1_short2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["long1_short2"],
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        security1=security1,
                        bar1=bar1,
                        security2=security2,
                        bar2=bar2
                    )
                elif (
                        can_exit_long1_short2
                        and take_profit_long1_short2
                ):
                    self.engine.log.info("take_profit_long1_short2")
                    self.take_profit_long1_short2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["long1_short2"],
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        security1=security1,
                        bar1=bar1,
                        security2=security2,
                        bar2=bar2
                    )
                elif (
                        can_exit_short1_long2
                        and exit_short1_long2
                ):
                    self.engine.log.info("exit_short1_long2")
                    self.exit_short1_long2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["short1_long2"],
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        security1=security1,
                        bar1=bar1,
                        security2=security2,
                        bar2=bar2
                    )
                elif (
                        can_exit_short1_long2
                        and take_profit_short1_long2
                ):
                    self.engine.log.info("take_profit_short1_long2")
                    self.take_profit_short1_long2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["short1_long2"],
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        security1=security1,
                        bar1=bar1,
                        security2=security2,
                        bar2=bar2
                    )


    def request_historical_ohlcv(
            self,
            gateway_name: str,
            security: Security
    ) -> List[Bar]:
        """Request the information every day"""
        return self.engine.req_historical_bars(
            security=security,
            gateway_name="Backtest",
            periods=self.lookback_period[gateway_name][security],
            freq=f"{int(TIME_STEP / 60000.)}Min",
            cur_datetime=self.engine.gateways[gateway_name].market_datetime,
            trading_sessions=self.engine.gateways[gateway_name].trading_sessions[
                security.code],
        )

    def get_bar_datetime(self, gateway_name: str) -> datetime:
        bar_datetime = datetime(1970, 1, 1)
        for g in self.engine.gateways:
            if g == gateway_name:
                if gateway_name not in self._cur_data:
                    continue
                for security in self.securities[gateway_name]:
                    if security not in self._cur_data[gateway_name]:
                        continue
                    bar = self._cur_data[gateway_name][security]
                    bar_datetime = max(bar_datetime, bar.datetime)
        return bar_datetime

    def send_order(
            self,
            security: Security,
            quantity: int,
            direction: Direction,
            offset: Offset,
            order_type: OrderType,
            gateway_name: str
    ) -> bool:
        """
        return True if order is successfully fully filled, else False
        """
        order_instruct = dict(
            security=security,
            quantity=quantity,
            direction=direction,
            offset=offset,
            order_type=order_type,
            gateway_name=gateway_name,
        )

        self.engine.log.info(f"Place order: {order_instruct}.")
        orderid = self.engine.send_order(**order_instruct)
        if orderid == "":
            self.engine.log.info("Failed to submit order.")
            return False
        self.engine.log.info(f"Order {orderid} has been submitted.")
        sleep(self.sleep_time)
        order = self.engine.get_order(
            orderid=orderid, gateway_name=gateway_name)
        self.engine.log.info(f"Order {orderid} status:{order}.")

        deals = self.engine.find_deals_with_orderid(
            orderid, gateway_name=gateway_name)
        for deal in deals:
            self.engine.portfolios[gateway_name].update(deal)
            self.portfolios[gateway_name].update(deal)

        if order.status == OrderStatus.FILLED:
            self.engine.log.info(f"Order {orderid} has been filled.")
            return True
        else:
            err = self.engine.cancel_order(
                orderid=orderid, gateway_name=gateway_name)
            if err:
                self.engine.log.info(f"Failed to cancel order {orderid} "
                                     f"due to the following reason: {err}.")
            else:
                self.engine.log.info(f"Order {orderid} has been cancelled.")
            return False

    def get_security_from_security_code(self, security_code: str) -> Security:
        for gateway_name in self.securities:
            for security in self.securities[gateway_name]:
                if security.code == security_code:
                    return security
        return None

    def get_current_datetime(
            self, cur_data: Dict[str, Dict[Security, Bar]]
    ) -> datetime:
        return max(list(itertools.chain(
            *[[cur_data[gn][sec].datetime for sec in cur_data[gn]]
              for gn in cur_data]
        )))

    def entry_long1_short2(
            self,
            no: int,
            gateway_name: str,
            security_pair: Tuple[str],
            security1: Security,
            bar1: Bar,
            security2: Security,
            bar2: Bar,
            gamma: float
    ):
        q1 = int(self.params["capital_per_entry"] / bar1.close)
        q2 = int(self.params["capital_per_entry"] / bar2.close / gamma)
        if q1 <= q2:
            qty1 = q1
            qty2 = int(qty1 * gamma)
        else:
            qty2 = q2
            qty1 = int(qty2 / gamma)
        action = dict(
            no=no,
            gw=gateway_name,
            sec=security1.code,
            qty=qty1,
            side="LONG",
            close=bar1.close,
            offset="OPEN"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled1 = self.send_order(
            security=security1,
            quantity=qty1,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        action = dict(
            no=no,
            gw=gateway_name,
            sec=security2.code,
            qty=qty2,
            side="SHORT",
            close=bar2.close,
            offset="OPEN"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled2 = self.send_order(
            security=security2,
            quantity=qty2,
            direction=Direction.SHORT,
            offset=Offset.OPEN,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        if filled1 and filled2:
            self.security_pairs_number_of_entry[gateway_name][security_pair][
                "long1_short2"] += 1
            self.security_pairs_quantity_of_entry[gateway_name][security_pair][
                "long1_short2"].append(
                (qty1, qty2))

    def exit_long1_short2(
            self,
            no: int,
            gateway_name: str,
            security_pair: Tuple[str],
            security1: Security,
            bar1: Bar,
            security2: Security,
            bar2: Bar
    ):
        self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"] -= 1
        qty1, qty2 = self.security_pairs_quantity_of_entry[gateway_name][security_pair][
            "long1_short2"].pop(0)
        action = dict(
            no=no,
            gw=gateway_name,
            sec=security1.code,
            qty=qty1,
            side="SHORT",
            close=bar1.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled1 = self.send_order(
            security=security1,
            quantity=qty1,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        action = dict(
            no=no,
            gw=gateway_name,
            sec=security2.code,
            qty=qty2,
            side="LONG",
            close=bar2.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled2 = self.send_order(
            security=security2,
            quantity=qty2,
            direction=Direction.LONG,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

    def take_profit_long1_short2(
            self,
            no: int,
            gateway_name: str,
            security_pair: Tuple[str],
            security1: Security,
            bar1: Bar,
            security2: Security,
            bar2: Bar
    ):
        qty1 = 0
        qty2 = 0
        for q1, q2 in self.security_pairs_quantity_of_entry[gateway_name][security_pair][
            "long1_short2"]:
            qty1 += q1
            qty2 += q2
            self.security_pairs_number_of_entry[gateway_name][security_pair][
                "long1_short2"] -= 1
        self.security_pairs_quantity_of_entry[gateway_name][security_pair][
            "long1_short2"] = []
        assert self.security_pairs_number_of_entry[gateway_name][security_pair][
                   "long1_short2"] == 0, (
            "Entry number and quantity mismatch!")
        action = dict(
            no=no,
            gw=gateway_name,
            sec=security1.code,
            qty=qty1,
            side="SHORT",
            close=bar1.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled1 = self.send_order(
            security=security1,
            quantity=qty1,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        action = dict(
            no=no,
            gw=gateway_name,
            sec=security2.code,
            qty=qty2,
            side="LONG",
            close=bar2.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled2 = self.send_order(
            security=security2,
            quantity=qty2,
            direction=Direction.LONG,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

    def entry_short1_long2(
            self,
            no: int,
            gateway_name: str,
            security_pair: Tuple[str],
            security1: Security,
            bar1: Bar,
            security2: Security,
            bar2: Bar,
            gamma: float
    ):
        q1 = int(self.params["capital_per_entry"] / bar1.close)
        q2 = int(self.params["capital_per_entry"] / bar2.close / gamma)
        if q1 <= q2:
            qty1 = q1
            qty2 = int(qty1 * gamma)
        else:
            qty2 = q2
            qty1 = int(qty2 / gamma)
        action = dict(
            no=no,
            gw=gateway_name,
            sec=security1.code,
            qty=qty1,
            side="SHORT",
            close=bar1.close,
            offset="OPEN"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled1 = self.send_order(
            security=security1,
            quantity=qty1,
            direction=Direction.SHORT,
            offset=Offset.OPEN,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        action = dict(
            no=no,
            gw=gateway_name,
            sec=security2.code,
            qty=qty2,
            side="LONG",
            close=bar2.close,
            offset="OPEN"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled2 = self.send_order(
            security=security2,
            quantity=qty2,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        if filled1 and filled2:
            self.security_pairs_number_of_entry[gateway_name][security_pair][
                "short1_long2"] += 1
            self.security_pairs_quantity_of_entry[gateway_name][security_pair][
                "short1_long2"].append((qty1, qty2))

    def exit_short1_long2(
            self,
            no: int,
            gateway_name: str,
            security_pair: Tuple[str],
            security1: Security,
            bar1: Bar,
            security2: Security,
            bar2: Bar
    ):
        self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"] -= 1
        qty1, qty2 = self.security_pairs_quantity_of_entry[gateway_name][security_pair][
            "short1_long2"].pop(0)
        action = dict(
            no=no,
            gw=gateway_name,
            sec=security1.code,
            qty=qty1,
            side="LONG",
            close=bar1.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled1 = self.send_order(
            security=security1,
            quantity=qty1,
            direction=Direction.LONG,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        action = dict(
            no=no,
            gw=gateway_name,
            sec=security2.code,
            qty=qty2,
            side="SHORT",
            close=bar2.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled2 = self.send_order(
            security=security2,
            quantity=qty2,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

    def take_profit_short1_long2(
            self,
            no: int,
            gateway_name: str,
            security_pair: Tuple[str],
            security1: Security,
            bar1: Bar,
            security2: Security,
            bar2: Bar
    ):
        qty1 = 0
        qty2 = 0
        for q1, q2 in self.security_pairs_quantity_of_entry[gateway_name][security_pair][
            "short1_long2"]:
            qty1 += q1
            qty2 += q2
            self.security_pairs_number_of_entry[gateway_name][security_pair][
                "short1_long2"] -= 1
        self.security_pairs_quantity_of_entry[gateway_name][security_pair][
            "long1_short2"] = []
        assert self.security_pairs_number_of_entry[gateway_name][security_pair][
                   "short1_long2"] == 0, (
            "Entry number and quantity mismatch!"
        )
        action = dict(
            no=no,
            gw=gateway_name,
            sec=security1.code,
            qty=qty1,
            side="LONG",
            close=bar1.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled1 = self.send_order(
            security=security1,
            quantity=qty1,
            direction=Direction.LONG,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)

        action = dict(
            no=no,
            gw=gateway_name,
            sec=security2.code,
            qty=qty2,
            side="SHORT",
            close=bar2.close,
            offset="CLOSE"
        )
        self.update_action(
            gateway_name=gateway_name, action=action)
        filled2 = self.send_order(
            security=security2,
            quantity=qty2,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)
