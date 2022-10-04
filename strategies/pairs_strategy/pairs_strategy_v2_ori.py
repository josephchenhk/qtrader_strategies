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
import os
from time import sleep
from typing import Dict, List, Tuple
from datetime import datetime
from datetime import timedelta
from copy import copy
import itertools
import json
import pickle

import numpy as np
from statsmodels.tsa.stattools import adfuller
import scipy.stats as st

from qtrader.core.constants import Direction, Offset, OrderType, OrderStatus
from qtrader.core.data import Bar
from qtrader.core.engine import Engine
from qtrader.core.position import Position
from qtrader.core.security import Security
from qtrader.core.strategy import BaseStrategy
from qtrader_config import TIME_STEP
import qtalib.indicators as ta


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
            self.recalibration_interval = int(
                self.params["lookback_period"]
                * self.params["recalibration_lookback_ratio"]
            )

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

            # If pairs are specified as params, we use them as given;
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

        # recalibration at the beginning
        self.recalibration()

    def recalibration(self):
        # find out candidate pairs
        self.current_candidate_security_pairs = []
        for gateway_name in self.engine.gateways:

            # if self.opt_params is None:
            #     shortlisted_security_pairs = self.security_pairs[gateway_name]
            # else:
            #     shortlisted_security_pairs = [
            #         k for k, v in self.opt_params.items()
            #         if v['sharpe_ratio'] > 0.8
            #     ]

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

                # w, _, _, _ = np.linalg.lstsq(
                #     logS2[:, np.newaxis], logS1, rcond=None)
                # gamma = w[0]
                # mu = 0

                a, b, c = line_total_least_squares(logS1, logS2)
                gamma = -b / a
                mu = -c / a
                spread = logS1 - gamma * logS2 - mu
                self.security_pairs_spread[gateway_name][
                    security_pair] = list(spread)

                # adf = adfuller(spread, autolag="AIC")
                # adf_pvalue = adf[1]
                spread_mean = spread.mean()
                spread_std = spread.std()

                spread_ma_short = ta.EMA(
                    spread, self.params['ma_short_length'])
                spread_ma_long = ta.EMA(spread, self.params['ma_long_length'])
                spread_ma_zscore = (
                    spread_ma_short - spread_ma_long) / spread_ma_long.std()
                spread_ma_zscore_mean = spread_ma_zscore.mean()
                spread_ma_zscore_std = spread_ma_zscore.std()
                adf = adfuller(spread_ma_zscore, autolag="AIC")
                adf_pvalue = adf[1]

                entry_mul = st.norm.ppf(self.params['entry_threshold_pct'])
                exit_mul = st.norm.ppf(self.params['exit_threshold_pct'])
                long_entry_threshold = spread_ma_zscore_mean - entry_mul * spread_ma_zscore_std
                long_exit_threshold = spread_ma_zscore_mean - exit_mul * spread_ma_zscore_std
                short_entry_threshold = spread_ma_zscore_mean + entry_mul * spread_ma_zscore_std
                short_exit_threshold = spread_ma_zscore_mean + exit_mul * spread_ma_zscore_std

                self.security_pairs_regression_params[gateway_name][
                    security_pair]["gamma"] = gamma
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["mu"] = mu
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["adf_pvalue"] = adf_pvalue
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["spread_mean"] = spread_mean
                self.security_pairs_regression_params[gateway_name][
                    security_pair]["spread_std"] = spread_std
                self.security_pairs_regression_params[gateway_name][
                    security_pair][
                    "long_entry_threshold"] = long_entry_threshold
                self.security_pairs_regression_params[gateway_name][
                    security_pair][
                    "long_exit_threshold"] = long_exit_threshold
                self.security_pairs_regression_params[gateway_name][
                    security_pair][
                    "short_entry_threshold"] = short_entry_threshold
                self.security_pairs_regression_params[gateway_name][
                    security_pair][
                    "short_exit_threshold"] = short_exit_threshold

                qty_long1_short2 = self.security_pairs_quantity_of_entry[
                    gateway_name][security_pair]["long1_short2"]
                qty_short1_long2 = self.security_pairs_quantity_of_entry[
                    gateway_name][security_pair]["short1_long2"]

                if (
                        (len(qty_long1_short2) == 0 and len(qty_short1_long2) == 0)
                        and corr > self.params["corr_init_threshold"]
                        and adf_pvalue < self.params[
                            "coint_pvalue_init_threshold"]
                        and 0.0 < gamma
                        # and security_pair in shortlisted_security_pairs
                ):
                    self.current_candidate_security_pairs.append(
                        security_pair)
                    # import matplotlib.pyplot as plt
                    # plt.plot((spread - spread_mean) / spread_std)
                    # plt.plot([long_entry_threshold] * len(spread))
                    # plt.plot([long_exit_threshold] * len(spread))
                    # plt.plot([short_entry_threshold] * len(spread))
                    # plt.plot([short_exit_threshold] * len(spread))
                    # plt.plot([0] * len(spread), "--")
                    # plt.show()
                    #
                    # plt.plot(spread_ma_zscore)
                    # plt.plot([long_entry_threshold] * len(spread))
                    # plt.plot([long_exit_threshold] * len(spread))
                    # plt.plot([short_entry_threshold] * len(spread))
                    # plt.plot([short_exit_threshold] * len(spread))
                    # plt.plot([spread_ma_zscore_mean] * len(spread), "--")
                    # plt.show()
                elif (
                        (qty_long1_short2 or qty_short1_long2)
                        and corr > self.params["corr_maintain_threshold"]
                        and adf_pvalue < self.params[
                            "coint_pvalue_maintain_threshold"]
                        and 0.0 < gamma
                        # and security_pair in shortlisted_security_pairs
                ):
                    # Still mark the pairs as a candidate
                    self.current_candidate_security_pairs.append(
                        security_pair)
                    # Rebalance to following quantities
                    bar1 = self.ohlcv[gateway_name][security1][-1]
                    bar2 = self.ohlcv[gateway_name][security2][-1]
                    qty1, qty2 = self.calc_entry_quantities(
                        security1, security2, bar1, bar2, gamma)

                    # Rebalance if long1_short2 position exists
                    for prev_qty1, prev_qty2 in qty_long1_short2:
                        delta_qty1 = qty1 - prev_qty1
                        delta_qty2 = qty2 - prev_qty2
                        if delta_qty1 > 0:
                            self.send_order(
                                security=security1,
                                quantity=delta_qty1,
                                direction=Direction.LONG,
                                price=bar1.close,
                                offset=Offset.OPEN,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                        elif delta_qty1 < 0:
                            self.send_order(
                                security=security1,
                                quantity=abs(delta_qty1),
                                direction=Direction.SHORT,
                                price=bar1.close,
                                offset=Offset.CLOSE,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                        if delta_qty2 > 0:
                            self.send_order(
                                security=security2,
                                quantity=delta_qty2,
                                direction=Direction.SHORT,
                                price=bar2.close,
                                offset=Offset.OPEN,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                        elif delta_qty2 < 0:
                            self.send_order(
                                security=security2,
                                quantity=abs(delta_qty2),
                                direction=Direction.LONG,
                                price=bar2.close,
                                offset=Offset.CLOSE,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                    self.security_pairs_quantity_of_entry[gateway_name][
                        security_pair]["long1_short2"] = [
                        (qty1, qty2) for _ in range(len(qty_long1_short2))
                    ]

                    # Rebalance if short1_long2 position exists
                    for prev_qty1, prev_qty2 in qty_short1_long2:
                        delta_qty1 = qty1 - prev_qty1
                        delta_qty2 = qty2 - prev_qty2
                        if delta_qty1 > 0:
                            self.send_order(
                                security=security1,
                                quantity=delta_qty1,
                                direction=Direction.SHORT,
                                price=bar1.close,
                                offset=Offset.OPEN,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                        elif delta_qty1 < 0:
                            self.send_order(
                                security=security1,
                                quantity=abs(delta_qty1),
                                direction=Direction.LONG,
                                price=bar1.close,
                                offset=Offset.CLOSE,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                        if delta_qty2 > 0:
                            self.send_order(
                                security=security2,
                                quantity=delta_qty2,
                                direction=Direction.LONG,
                                price=bar2.close,
                                offset=Offset.OPEN,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                        elif delta_qty2 < 0:
                            self.send_order(
                                security=security2,
                                quantity=abs(delta_qty2),
                                direction=Direction.SHORT,
                                price=bar2.close,
                                offset=Offset.CLOSE,
                                order_type=OrderType.MARKET,
                                gateway_name=gateway_name
                            )
                    self.security_pairs_quantity_of_entry[gateway_name][
                        security_pair]["short1_long2"] = [
                        (qty1, qty2) for _ in range(len(qty_short1_long2))
                    ]

                else:
                    # Clear all positions of this pair
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

        # Set next re-calibration date
        self.recalibration_date = self.recalibration_date + timedelta(
            seconds=self.recalibration_interval * TIME_STEP / 1000
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
                # spread = epsilon = log(S1) - gamma * log(S2) - mu
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
            self.load_opt_params(cur_datetime)
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
                # spread_mean = self.security_pairs_regression_params[
                #     gateway_name][security_pair]["spread_mean"]
                # spread_std = self.security_pairs_regression_params[
                #     gateway_name][security_pair]["spread_std"]
                long_entry_threshold = self.security_pairs_regression_params[
                    gateway_name][security_pair]["long_entry_threshold"]
                long_exit_threshold = self.security_pairs_regression_params[
                    gateway_name][security_pair]["long_exit_threshold"]
                short_entry_threshold = self.security_pairs_regression_params[
                    gateway_name][security_pair]["short_entry_threshold"]
                short_exit_threshold = self.security_pairs_regression_params[
                    gateway_name][security_pair]["short_exit_threshold"]

                spread = np.array(
                    self.security_pairs_spread[gateway_name][security_pair])
                # prev_spread_zscore = (spread[-2] - spread_mean) / spread_std
                # cur_spread_zscore = (spread[-1] - spread_mean) / spread_std

                # Use long-short term moving average to calculate zsocre
                # instead
                spread_ma_short = ta.EMA(
                    spread, self.params['ma_short_length'])
                spread_ma_long = ta.EMA(spread, self.params['ma_long_length'])
                spread_ma_zscore = (
                    spread_ma_short - spread_ma_long) / spread_ma_long.std()
                prev_spread_ma_zscore = spread_ma_zscore[-2]
                cur_spread_ma_zscore = spread_ma_zscore[-1]
                spread_zscore = (
                    spread - spread_ma_long) / spread_ma_long.std()
                # prev_spread_zscore = spread_zscore[-2]
                cur_spread_zscore = spread_zscore[-1]

                # check various entry/exit conditions
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

                entry_short1_long2 = (
                    cur_spread_ma_zscore < short_entry_threshold < prev_spread_ma_zscore
                    and cur_spread_zscore < cur_spread_ma_zscore
                )
                entry_long1_short2 = (
                    cur_spread_ma_zscore > long_entry_threshold > prev_spread_ma_zscore
                    and cur_spread_zscore > cur_spread_ma_zscore
                )

                # import matplotlib.pyplot as plt
                # plt.plot(spread_ma_zscore)
                # plt.plot([long_entry_threshold] * len(spread))
                # plt.plot([long_exit_threshold] * len(spread))
                # plt.plot([short_entry_threshold] * len(spread))
                # plt.plot([short_exit_threshold] * len(spread))
                # plt.show()

                can_exit_long1_short2 = (
                    self.security_pairs_number_of_entry[gateway_name][security_pair]["long1_short2"] > 0)
                can_exit_short1_long2 = (
                    self.security_pairs_number_of_entry[gateway_name][security_pair]["short1_long2"] > 0)
                exit_long1_short2 = (
                    cur_spread_ma_zscore < long_exit_threshold < prev_spread_ma_zscore)
                exit_short1_long2 = (
                    cur_spread_ma_zscore > short_exit_threshold > prev_spread_ma_zscore)

                # Execute trades
                if (
                        can_entry_long1_short2
                        and entry_long1_short2
                ):
                    self.engine.log.info(
                        f"entry_long1_short2: {security1.code}|{security2.code}")
                    qty1, qty2 = self.calc_entry_quantities(
                        security1, security2, bar1, bar2, gamma)
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
                    qty1, qty2 = self.calc_entry_quantities(
                        security1, security2, bar1, bar2, gamma)
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
            gateway_name: str,
            price: float = None,
    ) -> bool:
        """
        return True if order is successfully fully filled, else False
        """
        order_instruct = dict(
            security=security,
            quantity=quantity,
            price=price,
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

    def load_opt_params(self, cur_datetime: datetime):
        opt_params_files = os.listdir("strategies/pairs_strategy/opt_params")
        opt_params_files = [f for f in opt_params_files if ".pkl" in f]
        opt_params_files = sorted(opt_params_files)
        opt_params_file = None
        for i in range(1, len(opt_params_files)):
            prev_opt_params_file = opt_params_files[i - 1]
            cur_opt_params_file = opt_params_files[i]
            _, _, _, prev_end_str = prev_opt_params_file.replace(
                ".pkl", "").split("_")
            prev_end = datetime.strptime(prev_end_str, "%Y%m%d")
            _, _, _, cur_end_str = cur_opt_params_file.replace(
                ".pkl", "").split("_")
            cur_end = datetime.strptime(cur_end_str, "%Y%m%d")
            if prev_end <= cur_datetime < cur_end:
                opt_params_file = prev_opt_params_file
                break
            elif cur_datetime >= cur_end and i == len(opt_params_files) - 1:
                opt_params_file = cur_opt_params_file
                break
        assert opt_params_file is not None, (
            f"cur_datetime = {cur_datetime} couldn't find opt_params file!")
        with open(f"strategies/pairs_strategy/opt_params/{opt_params_file}", "rb") as f:
            opt_params = pickle.load(f)
        recalibration_lookback_ratios = [
            v['recalibration_lookback_ratio'] for k, v in opt_params.items()]
        ma_short_lengths = [v['ma_short_length']
                            for k, v in opt_params.items()]
        recalibration_lookback_ratio_mean = sum(
            recalibration_lookback_ratios) / len(recalibration_lookback_ratios)
        ma_short_length_mean = sum(ma_short_lengths) / len(ma_short_lengths)
        # print(
        #     cur_datetime,
        #     recalibration_lookback_ratio_mean,
        #     ma_short_length_mean)

        # update params
        self.params["recalibration_lookback_ratio"] = round(
            recalibration_lookback_ratio_mean, 3
        )
        self.params["ma_short_length"] = int(ma_short_length_mean)
        # update opt_params
        self.opt_params = {}
        for k, v in opt_params.items():
            v['recalibration_lookback_ratio'] = self.params["recalibration_lookback_ratio"]
            v['ma_short_length'] = self.params["ma_short_length"]
            self.opt_params[k] = v
        # update recalibration interval
        self.recalibration_interval = int(
            self.params["lookback_period"]
            * self.params["recalibration_lookback_ratio"]
        )

    def get_current_datetime(
            self, cur_data: Dict[str, Dict[Security, Bar]]
    ) -> datetime:
        return max(list(itertools.chain(
            *[[cur_data[gn][sec].datetime for sec in cur_data[gn]]
              for gn in cur_data]
        )))

    def calc_entry_quantities(
            self,
            security1: Security,
            security2: Security,
            bar1: Bar,
            bar2: Bar,
            gamma: float,
    ) -> Tuple[int]:
        assert gamma > 0, "gamma should be positive!"
        if gamma <= 1:
            qty1 = int(
                self.params["capital_per_entry"] /
                bar1.close /
                security1.lot_size)
            qty2 = int(
                self.params["capital_per_entry"] *
                gamma /
                bar2.close /
                security2.lot_size)
        else:
            qty1 = int(
                self.params["capital_per_entry"] /
                gamma /
                bar1.close /
                security1.lot_size)
            qty2 = int(
                self.params["capital_per_entry"] /
                bar2.close /
                security2.lot_size)
        return qty1, qty2

    def get_existing_quantities(
            self,
            gateway_name: str,
            security_pair: Tuple[str],
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
            price=bar1.close,
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
            price=bar2.close,
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
            price=bar1.close,
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
            price=bar2.close,
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
            price=bar1.close,
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
            price=bar2.close,
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
            price=bar1.close,
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
            price=bar2.close,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            gateway_name=gateway_name)
        if filled1 and filled2:
            return True
        return False


def line_total_least_squares(x, y):
    n = len(x)

    x_m = np.sum(x) / n
    y_m = np.sum(y) / n

    # Calculate the x~ and y~
    x1 = x - x_m
    y1 = y - y_m

    # Create the matrix array
    X = np.vstack((x1, y1))
    X_t = np.transpose(X)

    # Finding A_T_A and it's Find smallest eigenvalue::
    prd = np.dot(X, X_t)
    W, V = np.linalg.eig(prd)
    small_eig_index = W.argmin()
    a, b = V[:, small_eig_index]

    # Compute C:
    c = (-1 * a * x_m) + (-1 * b * y_m)

    return a, b, c


# def hurst_exponent(ts):
#     """Returns the Hurst Exponent of the time series vector ts
#     Ref:
#     1. https://en.wikipedia.org/wiki/Hurst_exponent
#     """
#     # Convert into np.array if the original ts is not a np.array
#     ts = np.array(ts)
#
#     N = len(ts)
#     min_lag = 10
#     max_lag = int(np.floor(N / 2))
#     lags = range(min_lag, max_lag + 1)
#
#     # Calculate different rescaled ranges
#     ARS = []
#     for lag in lags:
#         # divide ts into different samples
#         num = int(np.floor(N / lag))
#         RS = []
#         for n in range(num):
#             X = ts[n * lag:(n + 1) * lag]
#             # [Important]: take lagged differences
#             X_diff = X[1:] - X[:-1]  # X_diff = np.diff(X)
#             Y = X_diff - X_diff.mean()
#             Z = np.cumsum(Y)
#             R = Z.max() - Z.min()
#             S = Y.std(ddof=1)
#             if abs(S) < 1e-8:
#                 continue
#             RS.append(R / S)
#
#         ARS.append(sum(RS) / len(RS))
#
#     # Note we are estimating rescaled range (RS), instead of
#     # RS^2, so the factor 2 disappear here.
#     poly = np.polyfit(np.log(lags), np.log(ARS), 1)
#     return poly[0]
#
# def half_life(ts):
#     """
#     Calculates the half life of a mean reversion
#     """
#     # make sure we are working with an array, convert if necessary
#     ts = np.asarray(ts)
#
#     # delta = p(t) - p(t-1)
#     delta_ts = np.diff(ts)
#
#     # calculate the vector of lagged values. lag = 1
#     lag_ts = np.vstack([ts[:-1], np.ones(len(ts[:-1]))]).T
#
#     # calculate the slope of the deltas vs the lagged values
#     # Ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
#     lambda_, const = np.linalg.lstsq(lag_ts, delta_ts, rcond=None)[0]
#
#     # compute and return half life
#     # negative sign to turn half life to a positive value
#     return - np.log(2) / lambda_
