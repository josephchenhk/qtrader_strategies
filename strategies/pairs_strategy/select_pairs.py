# -*- coding: utf-8 -*-
# @Time    : 9/10/2022 8:21 am
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: temp.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""

from datetime import datetime

import numpy as np

from qtrader.core.security import Stock, Currency
from qtrader.core.constants import Exchange
from qtrader.core.data import _get_data


# securities = [
#     Stock(code='HK.00175', lot_size=1000, security_name='吉利汽车'),
#     Stock(code='HK.00305', lot_size=10000, security_name='五菱汽车'),
#     Stock(code='HK.00489', lot_size=2000, security_name='东风集团股份'),
#     Stock(code='HK.00708', lot_size=500, security_name='恒大汽车'),
#     Stock(code='HK.01114', lot_size=2000, security_name='华晨中国'),
#     Stock(code='HK.01122', lot_size=2000, security_name='庆铃汽车股份'),
#     Stock(code='HK.01211', lot_size=500, security_name='比亚迪股份'),
#     Stock(code='HK.01958', lot_size=500, security_name='北京汽车'),
#     Stock(code='HK.02015', lot_size=100, security_name='理想汽车-W'),
#     Stock(code='HK.02238', lot_size=2000, security_name='广汽集团'),
#     Stock(code='HK.02333', lot_size=500, security_name='长城汽车'),
#     # Stock(code='HK.09863', lot_size=100, security_name='零跑汽车'),
#     # Stock(code='HK.09866', lot_size=10, security_name='蔚来-SW'),
#     Stock(code='HK.09868', lot_size=100, security_name='小鹏汽车-W')
# ]

# securities = [
#     Stock(code='HK.00005', lot_size=400, security_name='汇丰控股'),
#     Stock(code='HK.00011', lot_size=100, security_name='恒生银行'),
#     Stock(code='HK.00023', lot_size=200, security_name='东亚银行'),
#     Stock(code='HK.02356', lot_size=400, security_name='大新银行集团'),
#     Stock(code='HK.02388', lot_size=500, security_name='中银香港'),
#     Stock(code='HK.02888', lot_size=50, security_name='渣打集团')
# ]

# securities = [
#     Stock(code='HK.00002', lot_size=500, security_name='中电控股'),
#     Stock(code='HK.00003', lot_size=1000, security_name='香港中华煤气'),
#     Stock(code='HK.00006', lot_size=500, security_name='电能实业'),
#     Stock(code='HK.02638', lot_size=500, security_name='港灯-SS'),
# ]

# securities = [
#     Stock(code='HK.00151', lot_size=1000, security_name='中国旺旺'),
#     Stock(code='HK.00220', lot_size=1000, security_name='统一企业中国'),
#     Stock(code='HK.00322', lot_size=2000, security_name='康师傅控股'),
#     Stock(code='HK.00345', lot_size=2000, security_name='维他奶国际'),
#     Stock(code='HK.00359', lot_size=4000, security_name='海升果汁'),
#     Stock(code='HK.00506', lot_size=2000, security_name='中国食品'),
#     Stock(code='HK.01068', lot_size=1000, security_name='雨润食品'),
#     Stock(code='HK.01262', lot_size=1000, security_name='蜡笔小新食品'),
#     Stock(code='HK.01458', lot_size=500, security_name='周黑鸭'),
#     Stock(code='HK.01583', lot_size=1000, security_name='亲亲食品'),
#     Stock(code='HK.01610', lot_size=1000, security_name='中粮家佳康'),
#     Stock(code='HK.03799', lot_size=500, security_name='达利食品'),
#     # Stock(code='HK.06183', lot_size=1000, security_name='中国绿宝')
# ]

securities = [
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

# stocks
# start = datetime(2021, 3, 1, 0, 0)
# end = datetime(2022, 3, 1, 0, 0)

# crypto
start = datetime(2021, 1, 1, 0, 0)
end = datetime(2022, 1, 1, 0, 0)

data_dct = {}
for security in securities:
    data = _get_data(
        security=security,
        start=start,
        end=end,
        dfield='kline',
        dtype=['time_key', 'open', 'high', 'low', 'close', 'volume'])
    data_dct[security.code] = data

corr_df = None
for i, security in enumerate(securities):
    pair_name = security.code
    df = data_dct[pair_name]
    df = df[["time_key", "close"]].set_index("time_key")
    df["close"] = np.log(df["close"] * security.lot_size)
    df.columns = [pair_name]
    if corr_df is None:
        corr_df = df
    else:
        corr_df = corr_df.join(df, how="outer")

corr_df = corr_df.ffill().bfill()

corr_table = corr_df.corr()
security_codes = set()
for i in range(corr_table.shape[0] - 1):
    for j in range(i + 1, corr_table.shape[0]):
        security_code1 = corr_table.index[i]
        security_code2 = corr_table.columns[j]
        corr = corr_table.iloc[i, j]
        if corr > 0.7:
            print(f"(\"{security_code1}\", \"{security_code2}\"),  # {corr}")
            security_codes.add(security_code1)
            security_codes.add(security_code2)
print(security_codes)
