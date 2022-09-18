# -*- coding: utf-8 -*-
# @Time    : 3/9/2022 3:56 PM
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: pairs_strategy.py

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
from copy import copy
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
            for cfg_name, cfg_value in override_indicator_cfg["params"].items(
            ):
                self.params[cfg_name] = cfg_value

        # optimized params will override params
        self.opt_params = kwargs.get("opt_params")

        # pairs are given as params
        self.security_pairs_params = kwargs.get("security_pairs")

        self.engine.log.info("Strategy loaded.")

    def init_strategy(self):

        # initialize strategy parameters
        self.sleep_time = 0
        self.current_candidate_security_pairs = []
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
        self.security_pairs_regression_params = {}

        self.security_pairs_spread = {}

        for gateway_name in self.securities:
            security_codes[gateway_name] = [
                s.code for s in self.securities[gateway_name]]
            self.ohlcv[gateway_name] = {}
            self.lookback_period[gateway_name] = {}

            # If pairs are specified as params, we use them;
            # otherwise list all of the combinations.
            if self.security_pairs_params:
                self.security_pairs[gateway_name] = self.security_pairs_params[:]
            else:
                self.security_pairs[gateway_name] = list(
                    itertools.combinations(security_codes[gateway_name], 2))

            self.security_pairs_number_of_entry[gateway_name] = {
                k: {"long1_short2": 0, "short1_long2": 0}
                for k in self.security_pairs[gateway_name]}
            self.security_pairs_quantity_of_entry[gateway_name] = {
                k: {"long1_short2": [], "short1_long2": []}
                for k in self.security_pairs[gateway_name]}
            self.security_pairs_regression_params[gateway_name] = {
                k: {"gamma": None, "mu": None, "adf_pvalue": None}
                for k in self.security_pairs[gateway_name]}
            self.security_pairs_spread[gateway_name] = {
                k: [] for k in self.security_pairs[gateway_name]}

            for security in self.securities[gateway_name]:
                self.lookback_period[gateway_name][security] = self.params[
                    "lookback_period"]
                self.ohlcv[gateway_name][
                    security] = self.request_historical_ohlcv(
                    gateway_name=gateway_name, security=security)

        # recalibration at the begining
        self.recalibration()

    def recalibration(self):
        # Clear exsiting positions
        for gateway_name in self.engine.gateways:
            for security_pair in self.security_pairs_quantity_of_entry[
                    gateway_name]:

                # # keep the position if it is still a candidate pair
                # if security_pair in self.current_candidate_security_pairs:
                #     continue

                # Clear all positions on recalibration date
                # clear position if it is no longer a candidate pair
                security1, security2 = list(map(
                    self.get_security_from_security_code, security_pair))
                for qty1, qty2 in self.security_pairs_quantity_of_entry[
                        gateway_name][security_pair]["long1_short2"]:
                    self.send_order(
                        security=security1,
                        quantity=qty1,
                        direction=Direction.SHORT,
                        offset=Offset.CLOSE,
                        order_type=OrderType.MARKET,
                        gateway_name=gateway_name
                    )
                    self.send_order(
                        security=security2,
                        quantity=qty2,
                        direction=Direction.LONG,
                        offset=Offset.CLOSE,
                        order_type=OrderType.MARKET,
                        gateway_name=gateway_name
                    )
                self.security_pairs_number_of_entry[gateway_name][
                    security_pair]["long1_short2"] = 0
                self.security_pairs_quantity_of_entry[gateway_name][
                    security_pair]["long1_short2"] = []

                for qty1, qty2 in self.security_pairs_quantity_of_entry[
                        gateway_name][security_pair]["short1_long2"]:
                    self.send_order(
                        security=security1,
                        quantity=qty1,
                        direction=Direction.LONG,
                        offset=Offset.CLOSE,
                        order_type=OrderType.MARKET,
                        gateway_name=gateway_name
                    )
                    self.send_order(
                        security=security2,
                        quantity=qty2,
                        direction=Direction.SHORT,
                        offset=Offset.CLOSE,
                        order_type=OrderType.MARKET,
                        gateway_name=gateway_name
                    )
                self.security_pairs_number_of_entry[gateway_name][
                    security_pair]["short1_long2"] = 0
                self.security_pairs_quantity_of_entry[gateway_name][
                    security_pair]["short1_long2"] = []

                # reset regression params
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["gamma"] = None
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["mu"] = None
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["adf_pvalue"] = None

        # find out candidate pairs
        self.current_candidate_security_pairs = []
        for gateway_name in self.engine.gateways:
            for security_pair in self.security_pairs[gateway_name]:
                security1, security2 = list(map(
                    self.get_security_from_security_code, security_pair))

                # Correlation
                logS1 = np.array([np.log(b.close) for b in
                                  self.ohlcv[gateway_name][security1]])
                logS2 = np.array([np.log(b.close) for b in
                                  self.ohlcv[gateway_name][security2]])
                corr = np.corrcoef(logS1, logS2)[0][1]

                # Cointegration
                # A = np.array([logS2, np.ones_like(logS2)])
                # w = np.linalg.lstsq(A.T, logS1, rcond=None)[0]
                # gamma, mu = w

                w, _, _, _ = np.linalg.lstsq(
                    logS2[:, np.newaxis], logS1, rcond=None)
                gamma = w[0]
                mu = 0

                if len(
                        self.security_pairs_spread[gateway_name][security_pair]) == 0:
                    spread = logS1 - gamma * logS2 - mu
                    self.security_pairs_spread[gateway_name][security_pair] = list(
                        spread)
                else:
                    # we obtain new gamma, thus refreshing the spreads
                    n = self.params["recalibration_interval"]
                    assert n >= 1, "recalibration_interval should >= 1!"
                    self.security_pairs_spread[gateway_name][
                        security_pair][-n:] = (
                        logS1[-n:] - gamma * logS2[-n:] - mu
                    )
                    spread = np.array(
                        self.security_pairs_spread[gateway_name][security_pair])
                adf = adfuller(spread, autolag="AIC")
                adf_pvalue = adf[1]

                self.security_pairs_regression_params[gateway_name][
                    security_pair]["gamma"] = gamma
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["mu"] = mu
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["adf_pvalue"] = adf_pvalue

                if (
                        corr > self.params["correlation_threshold"]
                        and adf_pvalue < self.params["cointegration_pvalue_entry_threshold"]
                        and 0.1 < gamma < 10.0
                ):
                    self.current_candidate_security_pairs.append(
                        security_pair)

        # Set next re-calibration date
        self.recalibration_date = self.recalibration_date + timedelta(
            seconds=self.params["recalibration_interval"] * TIME_STEP / 1000
        )

    def on_bar(self, cur_data: Dict[str, Dict[Security, Bar]]):

        self.engine.log.info(
            "-" * 30
            + "Enter <PairStrategy> on_bar"
            + "-" * 30
        )
        self.engine.log.info(cur_data)
        self._cur_data = cur_data

        # Collect ohlcv data
        for gateway_name in self.engine.gateways:
            if gateway_name not in cur_data:
                continue
            for security_pair in self.security_pairs[gateway_name]:
                security1, security2 = list(map(
                    self.get_security_from_security_code, security_pair))

                # validate current bar timestamps are aligned
                bar1 = cur_data[gateway_name].get(security1)
                bar2 = cur_data[gateway_name].get(security2)
                if bar1 is None and bar2 is None:
                    continue
                elif bar1 is None:
                    bar1 = copy(self.ohlcv[gateway_name][security1][-1])
                    bar1.datetime = bar2.datetime
                    cur_data[gateway_name][security1] = bar1
                elif bar2 is None:
                    bar2 = copy(self.ohlcv[gateway_name][security2][-1])
                    bar2.datetime = bar1.datetime
                    cur_data[gateway_name][security2] = bar2
                elif bar1.datetime != bar2.datetime:
                    raise ValueError(
                        "Current ohlcv timestamp doesn't match: "
                        f"{security1.code}|{security2.code}"
                    )

                # Collect the ohlcv data
                if (
                        bar1 is not None
                        and bar1.datetime > self.ohlcv[gateway_name][
                            security1][-1].datetime
                ):
                    self.ohlcv[gateway_name][security1].append(bar1)
                    while (
                        len(self.ohlcv[gateway_name][security1])
                        > self.params["lookback_period"]
                    ):
                        self.ohlcv[gateway_name][security1].pop(0)
                if (
                        bar2 is not None
                        and bar2.datetime > self.ohlcv[gateway_name][
                            security2][-1].datetime
                ):
                    self.ohlcv[gateway_name][security2].append(bar2)
                    while (
                        len(self.ohlcv[gateway_name][security2])
                        > self.params["lookback_period"]
                    ):
                        self.ohlcv[gateway_name][security2].pop(0)

                # validate historical bars timestamps are aligned
                bar1_first = self.ohlcv[gateway_name][security1][0]
                bar1_last = self.ohlcv[gateway_name][security1][-1]
                bar2_first = self.ohlcv[gateway_name][security2][0]
                bar2_last = self.ohlcv[gateway_name][security2][-1]
                assert (
                    bar1_first.datetime == bar2_first.datetime
                    and bar1_last.datetime == bar2_last.datetime
                ), ("Historical ohlcv timestamp doesn't match: "
                    f"{security1.code}|{security2.code}")

                # Cointegration test
                # log(S1) = gamma * log(S2) + mu + epsilon
                # spread (epsilon) is expected to be mean-reverting
                logS1 = np.log(self.ohlcv[gateway_name][security1][-1].close)
                logS2 = np.log(self.ohlcv[gateway_name][security2][-1].close)

                # logS1_fit = gamma * logS2 + mu
                # gamma, mu were obtained on recalibration date
                gamma = self.security_pairs_regression_params[gateway_name][
                    security_pair]["gamma"]
                mu = self.security_pairs_regression_params[gateway_name][
                    security_pair]["mu"]
                # spread is updated with new prices, but gamma and mu are
                # unchanged
                self.security_pairs_spread[gateway_name][security_pair].append(
                    logS1 - gamma * logS2 - mu
                )
                self.security_pairs_spread[gateway_name][security_pair].pop(0)

        cur_datetime = self.get_current_datetime(cur_data)

        # Recalibrate the parameters
        if cur_datetime >= self.recalibration_date:
            self.recalibration()

        # Find possible opportunities in every pair
        for gateway_name in self.engine.gateways:
            # if gateway_name not in cur_data:
            #     continue

            self.engine.log.info(
                f"strategy portfolio value: {self.get_strategy_portfolio_value(gateway_name)}\n"
                f"strategy positions: {self.portfolios[gateway_name].position}")

            for security_pair in self.security_pairs[gateway_name]:

                security1, security2 = list(map(
                    self.get_security_from_security_code, security_pair))

                if (
                    len(self.ohlcv[gateway_name][security1]) < self.params["lookback_period"]
                    or len(self.ohlcv[gateway_name][security2]) < self.params["lookback_period"]
                    or security_pair not in self.current_candidate_security_pairs
                ):
                    continue

                bar1 = cur_data[gateway_name].get(security1)
                bar2 = cur_data[gateway_name].get(security2)

                gamma = self.security_pairs_regression_params[gateway_name][
                    security_pair]["gamma"]
                spread = np.array(
                    self.security_pairs_spread[gateway_name][security_pair])

                spread_mean = spread.mean()
                spread_std = spread.std()
                spread_zscore = (spread[-1] - spread_mean) / spread_std

                # check various entry/exit conditions
                can_entry_long1_short2 = (
                    0 <= self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"]
                    < self.params["max_number_of_entry"]
                    and self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"] == 0
                    and (self.recalibration_date - cur_datetime).total_seconds() / 3600. > 24
                )
                can_entry_short1_long2 = (
                    0 <= self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"]
                    < self.params["max_number_of_entry"]
                    and self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"] == 0
                    and (self.recalibration_date - cur_datetime).total_seconds() / 3600. > 24
                )

                if self.opt_params:
                    entry_threshold = self.opt_params[security_pair][
                        "entry_threshold"]
                    exit_threshold = self.opt_params[security_pair][
                        "exit_threshold"]
                else:
                    entry_threshold = self.params["entry_threshold"]
                    exit_threshold = self.params["exit_threshold"]
                entry_short1_long2 = (
                    spread_zscore > entry_threshold
                    and spread_zscore < (exit_threshold + entry_threshold) / 2)
                entry_long1_short2 = (
                    spread_zscore < -entry_threshold
                    and spread_zscore > -(exit_threshold + entry_threshold) / 2
                )

                can_exit_long1_short2 = (
                    self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"] > 0)
                can_exit_short1_long2 = (
                    self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"] > 0)
                exit_long1_short2 = (
                    spread_zscore >= 0
                    or spread_zscore < -exit_threshold
                )
                exit_short1_long2 = (
                    spread_zscore <= 0
                    or spread_zscore > exit_threshold
                )

                # Take trades
                if (
                        can_entry_long1_short2
                        and entry_long1_short2
                ):
                    self.engine.log.info(
                        f"entry_long1_short2: {security1.code}|{security2.code}")
                    qty1, qty2 = self.calc_entry_quantities(bar1, bar2, gamma)
                    self.entry_long1_short2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["long1_short2"],
                        gateway_name=gateway_name,
                        security1=security1,
                        bar1=bar1,
                        qty1=qty1,
                        security2=security2,
                        bar2=bar2,
                        qty2=qty2
                    )
                    self.security_pairs_number_of_entry[gateway_name][
                        security_pair][
                        "long1_short2"] += 1
                    self.security_pairs_quantity_of_entry[gateway_name][
                        security_pair][
                        "long1_short2"].append(
                        (qty1, qty2))
                elif (
                        can_entry_short1_long2
                        and entry_short1_long2
                ):
                    self.engine.log.info(
                        f"entry_short1_long2 {security1.code}|{security2.code}")
                    qty1, qty2 = self.calc_entry_quantities(bar1, bar2, gamma)
                    self.entry_short1_long2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["short1_long2"],
                        gateway_name=gateway_name,
                        security1=security1,
                        bar1=bar1,
                        qty1=qty1,
                        security2=security2,
                        bar2=bar2,
                        qty2=qty2
                    )
                    self.security_pairs_number_of_entry[gateway_name][
                        security_pair][
                        "short1_long2"] += 1
                    self.security_pairs_quantity_of_entry[gateway_name][
                        security_pair][
                        "short1_long2"].append(
                        (qty1, qty2))
                elif (
                        can_exit_long1_short2
                        and exit_long1_short2
                ):
                    self.engine.log.info(
                        f"exit_all_long1_short2 {security1.code}|{security2.code}")
                    qty1, qty2 = self.get_existing_quantities(
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        position_side="long1_short2"
                    )
                    self.exit_long1_short2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["long1_short2"],
                        gateway_name=gateway_name,
                        security1=security1,
                        bar1=bar1,
                        qty1=qty1,
                        security2=security2,
                        bar2=bar2,
                        qty2=qty2
                    )
                    self.security_pairs_number_of_entry[gateway_name][
                        security_pair][
                        "long1_short2"] = 0
                    self.security_pairs_quantity_of_entry[gateway_name][
                        security_pair][
                        "long1_short2"] = []
                elif (
                        can_exit_short1_long2
                        and exit_short1_long2
                ):
                    self.engine.log.info(
                        f"exit_all_short1_long2 {security1.code}|{security2.code}")
                    qty1, qty2 = self.get_existing_quantities(
                        gateway_name=gateway_name,
                        security_pair=security_pair,
                        position_side="short1_long2"
                    )
                    self.exit_short1_long2(
                        no=self.security_pairs_number_of_entry[gateway_name][
                            security_pair]["short1_long2"],
                        gateway_name=gateway_name,
                        security1=security1,
                        bar1=bar1,
                        qty1=qty1,
                        security2=security2,
                        bar2=bar2,
                        qty2=qty2
                    )
                    self.security_pairs_number_of_entry[gateway_name][
                        security_pair][
                        "short1_long2"] = 0
                    self.security_pairs_quantity_of_entry[gateway_name][
                        security_pair][
                        "short1_long2"] = []

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

    # def calc_entry_quantities(
    #         self,
    #         bar1: Bar,
    #         bar2: Bar,
    #         gamma: float,
    # ) -> Tuple[int]:
    #     q1 = int(self.params["capital_per_entry"] / bar1.close)
    #     q2 = int(self.params["capital_per_entry"] / bar2.close / gamma)
    #     if q1 <= q2:
    #         qty1 = q1
    #         qty2 = int(q1 * gamma)
    #     else:
    #         qty1 = q2
    #         qty2 = int(q2 * gamma)
    #     return qty1, qty2

    def calc_entry_quantities(
            self,
            bar1: Bar,
            bar2: Bar,
            gamma: float,
    ) -> Tuple[int]:
        assert gamma > 0, "gamma should be positive!"
        if gamma <= 1:
            qty1 = int(self.params["capital_per_entry"] / bar1.close)
            qty2 = int(self.params["capital_per_entry"] * gamma / bar2.close)
        else:
            qty1 = int(self.params["capital_per_entry"] / gamma / bar1.close)
            qty2 = int(self.params["capital_per_entry"] / bar2.close)
        return qty1, qty2

    def get_existing_quantities(
            self,
            gateway_name: str,
            security_pair: str,
            position_side: str
    ) -> Tuple[int]:
        qty1 = 0
        qty2 = 0
        for q1, q2 in self.security_pairs_quantity_of_entry[gateway_name][
                security_pair][position_side]:
            qty1 += q1
            qty2 += q2
        return qty1, qty2

    def entry_long1_short2(
            self,
            no: int,
            gateway_name: str,
            security1: Security,
            bar1: Bar,
            qty1: int,
            security2: Security,
            bar2: Bar,
            qty2: int
    ) -> bool:
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
            return True
        return False

    def exit_long1_short2(
            self,
            no: int,
            gateway_name: str,
            security1: Security,
            bar1: Bar,
            qty1: int,
            security2: Security,
            bar2: Bar,
            qty2: int
    ) -> bool:
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
        if filled1 and filled2:
            return True
        return False

    def entry_short1_long2(
            self,
            no: int,
            gateway_name: str,
            security1: Security,
            bar1: Bar,
            qty1: int,
            security2: Security,
            bar2: Bar,
            qty2: int
    ) -> bool:
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
            return True
        return False

    def exit_short1_long2(
            self,
            no: int,
            gateway_name: str,
            security1: Security,
            bar1: Bar,
            qty1: int,
            security2: Security,
            bar2: Bar,
            qty2: int
    ):
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
        if filled1 and filled2:
            return True
        return False
